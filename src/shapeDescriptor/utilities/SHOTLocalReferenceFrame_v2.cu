// V2: SHARED MEMORY (TODO)
#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#endif

namespace ShapeDescriptor {
namespace v2 {
    // Helper for column-major outer product
    // a is a column vector, b is a row vector
    __device__ void outerProduct(const float3& a, const float3& b, float* M) {
        M[0] = a.x * b.x; M[3] = a.x * b.y; M[6] = a.x * b.z;
        M[1] = a.y * b.x; M[4] = a.y * b.y; M[7] = a.y * b.z;
        M[2] = a.z * b.x; M[5] = a.z * b.y; M[8] = a.z * b.z;
    }


    // WARNING: Function causes somewhat significant divergence in results from CPU (worst case ~3 %)
    // Possible cause: floating point addition order
    __global__ void calculateCovarianceMatrices(
        const ShapeDescriptor::gpu::PointCloud pointcloud,
        const ShapeDescriptor::gpu::array<OrientedPoint> imageOrigins,
        const ShapeDescriptor::gpu::array<float> maxSupportRadius,
        ShapeDescriptor::gpu::array<float> referenceWeightsZ,  // length originCount
        ShapeDescriptor::gpu::array<float> covarianceMatrices  // length originCount * 9, column-major 3x3 per origin
    )
    {
        uint32_t pointCount = pointcloud.pointCount;

        const ShapeDescriptor::gpu::VertexList &vertexList = pointcloud.vertices;

        // Hand out a descriptor origin to each block
        uint32_t originIndex = blockIdx.x;
        float3 origin = imageOrigins.content[originIndex].vertex;
        float R = maxSupportRadius.content[originIndex];
        float *cov = &covarianceMatrices.content[originIndex * 9];

        // Initialize arrays to 0
        if (threadIdx.x <= 8) {
            referenceWeightsZ.content[originIndex] = 0.0f;
            // Initialize each covariance matrix to a zero matrix
            cov[originIndex] = 0.0f;
        }
        __syncthreads();

        // Compute normalization factors Z
        for (uint32_t pointIndex = threadIdx.x; pointIndex < pointCount; pointIndex += blockDim.x) {
            float3 point = vertexList.at(pointIndex);

            float distance = length(point - origin);
            if (distance <= R) {
                atomicAdd_block(&referenceWeightsZ.content[originIndex], R - distance);
            }
        }
        __syncthreads();

        float Z = referenceWeightsZ.content[originIndex];
        // Only proceed if there are points in the support radius
        if (Z <= 0.0f) {
            if (threadIdx.x == 0) {
                // Set to identity if no points are in support, so we get a valid (but default) LRF
                float *cov = &covarianceMatrices.content[originIndex * 9];
                cov[0] = 1.0f; cov[4] = 1.0f; cov[8] = 1.0f;
            }
            return;
        }

        // Compute covariance matrices
        // Suggestion: Do this per origin, then per point instead
        // That way, we can simply divide by Z at the very end instead
        for (uint32_t pointIndex = threadIdx.x; pointIndex < pointCount; pointIndex += blockDim.x) {
            float3 point = vertexList.at(pointIndex);

            float3 pointDelta = point - origin;
            float distance = length(pointDelta);

            // NOTE: Potentially brutal branch divergence?
            // Potential solution: Store PCLs in a more sensible format
            if (distance > R) {
                continue;
            }

            float3 covarianceDelta = {pointDelta.x, pointDelta.y, pointDelta.z};

            assert(Z != 0);
            float relativeDistance = (R - distance) * (1.0f / Z);

            float temp[9];
            outerProduct(covarianceDelta, covarianceDelta, temp);

            for (int i = 0; i < 9; ++i) {
                // Fill the covariance matrices
                atomicAdd_block(&cov[i], relativeDistance * temp[i]);
            }
        }
    }

    __global__ void disambiguateEigenvectors(
        const ShapeDescriptor::gpu::PointCloud pointcloud,
        const ShapeDescriptor::gpu::array<OrientedPoint> imageOrigins,
        const ShapeDescriptor::gpu::array<float> maxSupportRadius,
        ShapeDescriptor::gpu::array<float> eigenvectors,
        ShapeDescriptor::gpu::array<int32_t> directionVotes,   // length originCount * 2
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> referenceFrames)
    {
        uint32_t pointCount = pointcloud.pointCount;

        const ShapeDescriptor::gpu::VertexList &vertexList = pointcloud.vertices;

        // Hand out a descriptor origin to each block
        uint32_t originIndex = blockIdx.x;
        float3 origin = imageOrigins.content[originIndex].vertex;

        if (threadIdx.x == 0) {
            // Initialise referenceFrames to eigenVectors' current values
            // NOTE: Probably wildly inefficient
            referenceFrames.content[originIndex].xAxis = float3(
                eigenvectors[originIndex*9 + 0],
                eigenvectors[originIndex*9 + 1],
                eigenvectors[originIndex*9 + 2]);

            referenceFrames.content[originIndex].yAxis = float3(
                eigenvectors[originIndex*9 + 3],
                eigenvectors[originIndex*9 + 4],
                eigenvectors[originIndex*9 + 5]);
            
            referenceFrames.content[originIndex].zAxis = float3(
                eigenvectors[originIndex*9 + 6],
                eigenvectors[originIndex*9 + 7],
                eigenvectors[originIndex*9 + 8]);

            // Zero out directionVotes
            directionVotes.content[2*originIndex + 0] = 0;
            directionVotes.content[2*originIndex + 1] = 0;
        }
        __syncthreads();

        // Compute directional votes
        for (uint32_t pointIndex = threadIdx.x; pointIndex < pointCount; pointIndex += blockDim.x) {
            float3 point = vertexList.at(pointIndex);

            float3 pointDelta = point - origin;
            if (length(pointDelta) > maxSupportRadius.content[originIndex]) {
                continue;
            }

            ShapeDescriptor::gpu::LocalReferenceFrame &frame = referenceFrames.content[originIndex];
            float dotX = dot(frame.xAxis, pointDelta);
            float dotZ = dot(frame.zAxis, pointDelta);
            atomicAdd_block(&directionVotes.content[2*originIndex + 0], (dotX > 0.0f) ? 1 : -1);
            atomicAdd_block(&directionVotes.content[2*originIndex + 1], (dotZ > 0.0f) ? 1 : -1);
        }
        __syncthreads();

        // Apply direction corrections
        if (threadIdx.x == 0) {
            ShapeDescriptor::gpu::LocalReferenceFrame &frame = referenceFrames.content[originIndex];
            if (directionVotes[2*originIndex + 0] < 0) {
                frame.xAxis *= -1.0f;
            }
            if (directionVotes[2*originIndex + 1] < 0) {
                frame.zAxis *= -1.0f;
            }
            frame.yAxis = cross(frame.xAxis, frame.zAxis);
        }
    }

ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> computeSHOTReferenceFrames(
    const ShapeDescriptor::gpu::PointCloud& pointcloud,
    const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& imageOrigins,
    const ShapeDescriptor::gpu::array<float>& maxSupportRadius,
    SHOTExecutionTimes* executionTimes = nullptr)
{
    // // -----------------------------------------------------------------------
    // // Start LRF timing
    // auto startLRFTime = std::chrono::high_resolution_clock::now();
    // // -----------------------------------------------------------------------

    uint32_t originCount = imageOrigins.length;


    // ----------------------- Covariance matrices ----------------------- 
    // Start covariance timing
    auto startCovarianceTime = std::chrono::high_resolution_clock::now();

    // Allocate GPU memory for temporary matrices
    ShapeDescriptor::gpu::array<float> d_referenceWeightsZ(originCount);
    ShapeDescriptor::gpu::array<float> d_covarianceMatrices(originCount * 9);

    // Calculate SHOT's "covariance matrices"
    calculateCovarianceMatrices<<<originCount, 416>>>(pointcloud, imageOrigins, maxSupportRadius, d_referenceWeightsZ, d_covarianceMatrices);

    // Synchronize and check if any errors occurred
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Got CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    std::cout << "Kernel finished -- covariance matrices calculated" << std::endl;

    // End covariance timing
    auto endCovarianceTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> covarianceDuration = endCovarianceTime - startCovarianceTime;
    double covarianceTimeElapsedSeconds = covarianceDuration.count();
    // ----------------------- Covariance matrices ----------------------- 



    // ------------------------------- EVD -------------------------------
    // Start EVD timing
    auto startEVDTime = std::chrono::high_resolution_clock::now();

    // Compute initial eigenvectors for all keypoints (origins)
    ShapeDescriptor::gpu::array<float> d_eigenvectors = ShapeDescriptor::v2::computeEigenVectorsMultiple(d_covarianceMatrices, originCount);

    // End EVD timing
    auto endEVDTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> EVDDuration = endEVDTime - startEVDTime;
    double EVDTimeElapsedSeconds = EVDDuration.count();

    std::cout << "cuSOLVER finished -- eigenvectors calculated" << std::endl;
    // ------------------------------- EVD -------------------------------



    // -------------------- Eigenvector disambiguation -------------------
    ShapeDescriptor::gpu::array<int32_t> d_directionVotes(originCount * 2);
    ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> d_referenceFrames(originCount);
    // Start disambiguation timing
    auto startDisambiguationTime = std::chrono::high_resolution_clock::now();

    // Disambiguate eigenvector directions, and put results in referenceFrames array
    disambiguateEigenvectors<<<originCount, 416>>>(pointcloud, imageOrigins, maxSupportRadius, d_eigenvectors, d_directionVotes, d_referenceFrames);

    // Synchronize and check if any errors occurred
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Got CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    std::cout << "Kernel finished -- LRFs generated" << std::endl;

    // End disambiguation timing
    auto endDisambiguationTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> disambiguationDuration = endDisambiguationTime - startDisambiguationTime;
    double disambiguationTimeElapsedSeconds = disambiguationDuration.count();
    // -------------------- Eigenvector disambiguation -------------------



    // Store timing results, if applicable
    if (executionTimes != nullptr) {
        executionTimes->covarianceMatricesGenerationTimeSeconds = covarianceTimeElapsedSeconds;
        executionTimes->EVDCalculationTimeSeconds = EVDTimeElapsedSeconds;
        executionTimes->eigenvectorDisambiguationTimeSeconds = disambiguationTimeElapsedSeconds;
    }

    // Cleanup
    ShapeDescriptor::free(d_referenceWeightsZ);
    ShapeDescriptor::free(d_eigenvectors);
    ShapeDescriptor::free(d_directionVotes);

    // // -----------------------------------------------------------------------
    // // End LRF timing
    // auto endLRFTime = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> LRFDuration = endLRFTime - startLRFTime;
    // double LRFTimeElapsedSeconds = LRFDuration.count();
    // std::cout << "LRF time (internal): " << LRFTimeElapsedSeconds << "\n";
    // // -----------------------------------------------------------------------


    return d_referenceFrames;
}
}
}