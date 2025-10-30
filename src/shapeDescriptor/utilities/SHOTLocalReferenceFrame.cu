// Device-side version of computeSHOTReferenceFrames.
// Note: caller must allocate and pass device pointers for all arrays.
// - vertices: pointer to pointCloud.vertices (length pointCount)
// - imageOrigins: pointer to imageOrigins.content (length originCount)
// - maxSupportRadius: pointer to maxSupportRadius (length originCount)
// - referenceFrames: output pointer (length originCount)
// - referenceWeightsZ: temp buffer (length originCount) - must be zero-initialized by caller or will be initialized here
// - covarianceMatrices: temp buffer (length originCount * 9) - row-major 3x3 per origin (must be initialized by caller or will be initialized here)
// - directionVotes: temp buffer (length originCount * 2) - must be zero-initialized by caller or will be initialized here
//
// This function assumes a device-side eigenvector routine exists:
//   std::array<ShapeDescriptor::cpu::float3,3> ShapeDescriptor::internal::computeEigenVectorsDevice(const float* matrix3x3_row_major)
// You must implement/port computeEigenVectorsDevice to run on the device (it is referenced below).
//
// Replace or adapt names/types to match your project if needed.
#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#endif

// Helper for 3x3 matrix (row-major) multiplication: C = A * B
inline __device__ void mat3_mul(const float *A, const float *B, float *C)
{
    // A, B, C are pointers to 9 floats, row-major: [ r0c0, r0c1, r0c2, r1c0, ... ]
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k)
                sum += A[r*3 + k] * B[k*3 + c];
            C[r*3 + c] = sum;
        }
    }
}

// Helper for column-major outer product
// a is a column vector, b is a row vector
__device__ void outerProduct(const float3& a, const float3& b, float* M) {
    M[0] = a.x * b.x; M[3] = a.x * b.y; M[6] = a.x * b.z;
    M[1] = a.y * b.x; M[4] = a.y * b.y; M[7] = a.y * b.z;
    M[2] = a.z * b.x; M[5] = a.z * b.y; M[8] = a.z * b.z;
}


namespace ShapeDescriptor {
namespace internal {
    __global__ void prepareEVD(
        const ShapeDescriptor::gpu::PointCloud pointcloud,
        const ShapeDescriptor::gpu::array<OrientedPoint> imageOrigins,
        const ShapeDescriptor::gpu::array<float> maxSupportRadius,
        ShapeDescriptor::gpu::array<float> referenceWeightsZ,  // length originCount
        ShapeDescriptor::gpu::array<float> covarianceMatrices  // length originCount * 9, column-major 3x3 per origin
    )
    {
        uint32_t pointCount = pointcloud.pointCount;
        uint32_t originCount = imageOrigins.length;

        ShapeDescriptor::gpu::VertexList vertexList = pointcloud.vertices;

        // Initialize arrays
        for (uint32_t i = 0; i < originCount; ++i)
        {
            referenceWeightsZ.content[i] = 0.0f;
            // Initialize each covariance matrix to identity matrix
            float *cov = &covarianceMatrices.content[i * 9];
            cov[0] = 1.0f;
            cov[1] = 0.0f;
            cov[2] = 0.0f;
            cov[3] = 0.0f;
            cov[4] = 1.0f;
            cov[5] = 0.0f;
            cov[6] = 0.0f;
            cov[7] = 0.0f;
            cov[8] = 1.0f;
        }

        // Compute normalization factors Z
        for (uint32_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
        {
            float3 point = vertexList.at(pointIndex);
            for (uint32_t originIndex = 0; originIndex < originCount; ++originIndex)
            {
                float3 origin = imageOrigins.content[originIndex].vertex;
                float distance = length(point - origin);
                float r = maxSupportRadius.content[originIndex];
                if (distance <= r)
                {
                    referenceWeightsZ.content[originIndex] += (r - distance);
                }
            }
        }

        // Compute covariance matrices
        for (uint32_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            float3 point = vertexList.at(pointIndex);
            for (uint32_t originIndex = 0; originIndex < originCount; ++originIndex) {
                float3 origin = imageOrigins.content[originIndex].vertex;
                float3 pointDelta = point - origin;
                float distance = length(pointDelta);
                // Potentially brutal branch divergence?
                if (distance > maxSupportRadius.content[originIndex])
                {
                    continue;
                }

                // build covarianceDeltaTransposed (row-major)
                float3 covarianceDelta = {pointDelta.x, pointDelta.y, pointDelta.z};

                float Z = referenceWeightsZ[originIndex];
                float relativeDistance = distance * (1.0f / Z);

                // temp = covDelta * covDeltaT
                float temp[9];
                outerProduct(covarianceDelta, covarianceDelta, temp);

                float *cov = &covarianceMatrices.content[originIndex * 9];
                for (int i = 0; i < 9; ++i) {
                    // Fill the covariance matrices
                    cov[i] += relativeDistance * temp[i];
                }
            }
        }
    }

    __global__ void disambiguateEigenvectors(
        const ShapeDescriptor::gpu::PointCloud pointcloud,
        const ShapeDescriptor::gpu::array<OrientedPoint> imageOrigins,
        ShapeDescriptor::gpu::array<float> d_eigenvectors,
        ShapeDescriptor::gpu::array<int32_t> directionVotes,   // length originCount * 2
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> referenceFrames)
    {
        uint32_t pointCount = pointcloud.pointCount;
        uint32_t originCount = imageOrigins.length;

        ShapeDescriptor::gpu::VertexList d_vertexList = pointcloud.vertices;

        for (uint32_t originIndex = 0; originIndex < originCount; originIndex++)
        {
            // Initialise referenceFrames to eigenVectors' current values
            // NOTE: Probably wildly inefficient
            referenceFrames.content[originIndex].xAxis = float3(
                d_eigenvectors[originIndex*9 + 0],
                d_eigenvectors[originIndex*9 + 1],
                d_eigenvectors[originIndex*9 + 2]);

            referenceFrames.content[originIndex].yAxis = float3(
                d_eigenvectors[originIndex*9 + 3],
                d_eigenvectors[originIndex*9 + 4],
                d_eigenvectors[originIndex*9 + 5]);
            
            referenceFrames.content[originIndex].zAxis = float3(
                d_eigenvectors[originIndex*9 + 6],
                d_eigenvectors[originIndex*9 + 7],
                d_eigenvectors[originIndex*9 + 8]);

            // Zero out directionVotes (although allocation may already have done this)
            directionVotes.content[2*originIndex + 0] = 0;
            directionVotes.content[2*originIndex + 1] = 0;
        }

        // Compute directional votes
        for (uint32_t pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
            float3 point = d_vertexList.at(pointIndex);
            for (uint32_t originIndex = 0; originIndex < originCount; ++originIndex) {
                float3 origin = imageOrigins.content[originIndex].vertex;
                float3 pointDelta = point - origin;
                ShapeDescriptor::gpu::LocalReferenceFrame frame = referenceFrames.content[originIndex];
                float dotX = dot(frame.xAxis, pointDelta);
                float dotZ = dot(frame.zAxis, pointDelta);
                directionVotes.content[2*originIndex + 0] += (dotX > 0.0f) ? 1 : -1;
                directionVotes.content[2*originIndex + 1] += (dotZ > 0.0f) ? 1 : -1;
            }
        }

        // Apply direction corrections
        for (uint32_t originIndex = 0; originIndex < originCount; ++originIndex)
        {
            ShapeDescriptor::gpu::LocalReferenceFrame &frame = referenceFrames.content[originIndex];
            if (directionVotes[2*originIndex + 0] < 0)
            {
                frame.xAxis *= -1.0f;
            }
            if (directionVotes[2*originIndex + 1] < 0)
            {
                frame.zAxis *= -1.0f;
            }
            frame.yAxis = cross(frame.xAxis, frame.zAxis);
        }

        // // DEBUG: Printf contents of referenceFrames
        // printf("GPU referenceFrames\n");
        // for(uint32_t i = 0; i < imageOrigins.length; ++i) {
        //     const ShapeDescriptor::gpu::LocalReferenceFrame frame = referenceFrames.content[i];
        //     printf("Reference frame %u:\n", i);
        //     printf("  [ %f, %f, %f ]\n", frame.xAxis.x, frame.xAxis.y, frame.xAxis.z);
        //     printf("  [ %f, %f, %f ]\n", frame.yAxis.x, frame.yAxis.y, frame.yAxis.z);
        //     printf("  [ %f, %f, %f ]\n", frame.zAxis.x, frame.zAxis.y, frame.zAxis.z);
        // }
    }

ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> computeSHOTReferenceFrames(
    const ShapeDescriptor::gpu::PointCloud& pointcloud,
    const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& imageOrigins,
    const ShapeDescriptor::gpu::array<float>& maxSupportRadius)
{
    uint32_t originCount = imageOrigins.length;

    // Allocate GPU memory for temporary matrices
    ShapeDescriptor::gpu::array<float> d_referenceWeightsZ(originCount);
    ShapeDescriptor::gpu::array<float> d_covarianceMatrices(originCount * 9);

    // Alternative way of zeroing out the reference weights (currently done in prepareEVD)
    // Might not be necessary at all (if ::gpu::array<>() zeros out by default)
    // float zero = 0.0f;
    // d_referenceWeightsZ.setValue(zero);

    // Prepare for eigenvalue decomposition:
    // - calculate reference weight Z for each keypoint
    // - calculate covariance matrix for each keypoint
    prepareEVD<<<1, 1>>>(pointcloud, imageOrigins, maxSupportRadius, d_referenceWeightsZ, d_covarianceMatrices);

    // Synchronize and check if any errors occurred
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Got CUDA error: %s\n", cudaGetErrorString(err));
    }
    std::cout << "Kernel finished -- prepared for EVD" << std::endl;

    // Compute initial eigenvectors for all keypoints (origins)
    // Must execute this function in host in order to use cuSOLVER
    ShapeDescriptor::gpu::array<float> d_eigenvectors = ShapeDescriptor::gpu::computeEigenVectorsMultiple(d_covarianceMatrices, originCount);

    // Disambiguate eigenvector directions, and put results in referenceFrames array
    ShapeDescriptor::gpu::array<int32_t> d_directionVotes(originCount * 2);
    ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> referenceFrames(originCount);
    disambiguateEigenvectors<<<1, 1>>>(pointcloud, imageOrigins, d_eigenvectors, d_directionVotes, referenceFrames);

    // Synchronize and check if any errors occurred
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Got CUDA error: %s\n", cudaGetErrorString(err));
    }
    std::cout << "Kernel finished -- LRFs generated" << std::endl;

    return referenceFrames;
}
}
}