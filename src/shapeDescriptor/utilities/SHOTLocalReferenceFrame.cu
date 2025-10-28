// // Device-side version of computeSHOTReferenceFrames.
// // Note: caller must allocate and pass device pointers for all arrays.
// // - vertices: pointer to pointCloud.vertices (length pointCount)
// // - imageOrigins: pointer to imageOrigins.content (length originCount)
// // - maxSupportRadius: pointer to maxSupportRadius (length originCount)
// // - referenceFrames: output pointer (length originCount)
// // - referenceWeightsZ: temp buffer (length originCount) - must be zero-initialized by caller or will be initialized here
// // - covarianceMatrices: temp buffer (length originCount * 9) - row-major 3x3 per origin (must be initialized by caller or will be initialized here)
// // - directionVotes: temp buffer (length originCount * 2) - must be zero-initialized by caller or will be initialized here
// //
// // This function assumes a device-side eigenvector routine exists:
// //   std::array<ShapeDescriptor::cpu::float3,3> ShapeDescriptor::internal::computeEigenVectorsDevice(const float* matrix3x3_row_major)
// // You must implement/port computeEigenVectorsDevice to run on the device (it is referenced below).
// //
// // Replace or adapt names/types to match your project if needed.
// #include <shapeDescriptor/shapeDescriptor.h>

// #include <cuda_runtime.h>
// #include <cusolverDn.h>

// // Helper for 3x3 matrix (row-major) multiplication: C = A * B
// inline __device__ void mat3_mul(const float *A, const float *B, float *C)
// {
//     // A, B, C are pointers to 9 floats, row-major: [ r0c0, r0c1, r0c2, r1c0, ... ]
//     for (int r = 0; r < 3; ++r)
//     {
//         for (int c = 0; c < 3; ++c)
//         {
//             float sum = 0.0f;
//             for (int k = 0; k < 3; ++k)
//                 sum += A[r*3 + k] * B[k*3 + c];
//             C[r*3 + c] = sum;
//         }
//     }
// }

// namespace ShapeDescriptor {
// namespace internal {
// __global__ void prepareEVD() {
//     // Initialize temporaries
//     for (uint32_t i = 0; i < originCount; ++i)
//     {
//         referenceWeightsZ[i] = 0.0f;
//         directionVotes[2 * i + 0] = 0;
//         directionVotes[2 * i + 1] = 0;
//         // init covariance to identity as original code used glm::mat3(1.0) - if you prefer zero, change accordingly
//         float *cov = &covarianceMatrices[i * 9];
//         cov[0] = 1.0f;
//         cov[1] = 0.0f;
//         cov[2] = 0.0f;
//         cov[3] = 0.0f;
//         cov[4] = 1.0f;
//         cov[5] = 0.0f;
//         cov[6] = 0.0f;
//         cov[7] = 0.0f;
//         cov[8] = 1.0f;
//     }

//     // Compute normalization factors Z
//     for (uint32_t i = 0; i < pointCount; ++i)
//     {
//         ShapeDescriptor::cpu::float3 point = vertices[i];
//         for (uint32_t j = 0; j < originCount; ++j)
//         {
//             ShapeDescriptor::cpu::float3 origin = imageOrigins[j].vertex;
//             float distance = length(point - origin);
//             float r = maxSupportRadius[j];
//             if (distance <= r)
//             {
//                 referenceWeightsZ[j] += (r - distance);
//             }
//         }
//     }

//     // Compute covariance matrices
//     for (uint32_t i = 0; i < pointCount; ++i)
//     {
//         float3 point = vertices[i];
//         for (uint32_t j = 0; j < originCount; ++j)
//         {
//             float3 origin = imageOrigins[j].vertex;
//             float3 pointDelta = point - origin;
//             float distance = length(pointDelta);
//             float r = maxSupportRadius[j];
//             if (distance <= r)
//             {
//                 // build covDeltaTransposed (row-major)
//                 float covDeltaT[9] = {
//                     pointDelta.x, pointDelta.y, pointDelta.z,
//                     0.0f, 0.0f, 0.0f,
//                     0.0f, 0.0f, 0.0f};
//                 // covDelta = transpose(covDeltaT)
//                 float covDelta[9];
//                 covDelta[0] = covDeltaT[0];
//                 covDelta[1] = covDeltaT[3];
//                 covDelta[2] = covDeltaT[6];
//                 covDelta[3] = covDeltaT[1];
//                 covDelta[4] = covDeltaT[4];
//                 covDelta[5] = covDeltaT[7];
//                 covDelta[6] = covDeltaT[2];
//                 covDelta[7] = covDeltaT[5];
//                 covDelta[8] = covDeltaT[8];

//                 float rel = 1.0f;
//                 // protect against division by zero
//                 float Z = referenceWeightsZ[j];
//                 if (Z != 0.0f)
//                     rel = distance * (1.0f / Z);
//                 else
//                     rel = 0.0f;

//                 // temp = covDelta * covDeltaT
//                 float temp[9];
//                 *covDelta = *covDeltaT * *temp;

//                 // covarianceMatrices[o] += rel * temp
//                 float *cov = &covarianceMatrices[j * 9];
//                 for (int i = 0; i < 9; ++i)
//                     cov[i] += rel * temp[i];
//             }
//         }
//     }
// }

// void computeSHOTReferenceFramesDevice(
//     const ShapeDescriptor::gpu::PointCloud& pointcloud,
//     const ShapeDescriptor::gpu::array<OrientedPoint>& imageOrigins,
//     const float* maxSupportRadius,
//     ShapeDescriptor::gpu::LocalReferenceFrame* referenceFrames,
//     float* referenceWeightsZ,           // length originCount
//     float* covarianceMatrices,          // length originCount * 9, row-major 3x3 per origin
//     int32_t* directionVotes)            // length originCount * 2
// {
//     uint32_t pointCount = pointcloud.pointCount;
//     uint32_t originCount = imageOrigins.length;
//     ShapeDescriptor::gpu::VertexList d_vertices = pointcloud.vertices;

//     prepareEVD<<<1, 1>>>();

//     // Compute initial eigenvectors for all keypoints (origins)
//     for (uint32_t i = 0; i < originCount; ++i)
//     {
//         float* cov = &covarianceMatrices[i * 9];

//         // Temporary:
//         ShapeDescriptor::cpu::array<float> h_cov(originCount, cov);
//         ShapeDescriptor::gpu::array<float> d_cov = ShapeDescriptor::copyToGPU(h_cov);

//         // NOTE: computeEigenVectorsDevice must be provided as a device function that operates on the convertedMatrix
//         ShapeDescriptor::gpu::array<float> d_eigenvectors = ShapeDescriptor::gpu::computeEigenVectorsMultiple(d_cov, originCount); // implement/port this to device

//         // referenceFrames[o].xAxis = eigenVectors[0];
//         // referenceFrames[o].yAxis = eigenVectors[1];
//         // referenceFrames[o].zAxis = eigenVectors[2];
//     }

//     // On GPU: 

//     // Compute directional votes
//     for (uint32_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
//     {
//         float3 point = vertices[pointIndex];
//         for (uint32_t keypointIndex = 0; keypointIndex < originCount; ++keypointIndex)
//         {
//             float3 origin = imageOrigins[keypointIndex].vertex;
//             float3 pointDelta = point - origin;
//             ShapeDescriptor::gpu::LocalReferenceFrame frame = referenceFrames[keypointIndex]; // copy
//             float dotX = dot(frame.xAxis, pointDelta);
//             float dotZ = dot(frame.zAxis, pointDelta);
//             directionVotes[2*keypointIndex + 0] += (dotX > 0.0f) ? 1 : -1;
//             directionVotes[2*keypointIndex + 1] += (dotZ > 0.0f) ? 1 : -1;
//         }
//     }

//     // Apply direction corrections
//     for (uint32_t i = 0; i < originCount; ++i)
//     {
//         ShapeDescriptor::gpu::LocalReferenceFrame &frame = referenceFrames[i];
//         if (directionVotes[2*i + 0] < 0)
//         {
//             frame.xAxis *= -1.0f;
//         }
//         if (directionVotes[2*i + 1] < 0)
//         {
//             frame.zAxis *= -1.0f;
//         }
//         frame.yAxis = cross(frame.xAxis, frame.zAxis);
//     }
// }
// }
// }