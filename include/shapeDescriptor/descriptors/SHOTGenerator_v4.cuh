// V4: Per point per descriptor (batched)
#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <chrono>
#include "ShapeContextGenerator.h"

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#endif

namespace ShapeDescriptor {
namespace v4 {
namespace {
        template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
        __device__
        inline void incrementSHOTBinDevice(ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>& descriptor,
                                     uint32_t elevationBinIndex, uint32_t radialBinIndex, uint32_t azimuthBinIndex, uint32_t histogramBinIndex, float contribution) {
            assert(elevationBinIndex < ELEVATION_DIVISIONS);
            assert(radialBinIndex < RADIAL_DIVISIONS);
            assert(azimuthBinIndex < AZIMUTH_DIVISIONS);
            assert(histogramBinIndex < INTERNAL_HISTOGRAM_BINS);
            uint32_t descriptorBinIndex =
                    elevationBinIndex * RADIAL_DIVISIONS * AZIMUTH_DIVISIONS * INTERNAL_HISTOGRAM_BINS
                  + radialBinIndex * AZIMUTH_DIVISIONS * INTERNAL_HISTOGRAM_BINS
                  + azimuthBinIndex * INTERNAL_HISTOGRAM_BINS
                  + histogramBinIndex;
            assert(descriptorBinIndex < ELEVATION_DIVISIONS * RADIAL_DIVISIONS * AZIMUTH_DIVISIONS * INTERNAL_HISTOGRAM_BINS);
            // descriptor.contents[descriptorBinIndex] += contribution;
            atomicAdd_block(&descriptor.contents[descriptorBinIndex], contribution);
        }

        template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
        __global__ __launch_bounds__(416, 2)
        void computeGeneralisedSHOTDescriptor(
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const ShapeDescriptor::gpu::PointCloud pointCloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors,
                const ShapeDescriptor::gpu::v3::LocalReferenceFrames localReferenceFrames,
                const ShapeDescriptor::gpu::array<float> supportRadii)
        {
            const uint32_t descriptorIndex = blockIdx.x;

            const float3 originVertex = descriptorOrigins.content[descriptorIndex].vertex;

            // Set up shared memory for this block's descriptor and LRF
            __shared__ ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS> localDescriptor;
            constexpr uint32_t binCount = ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>::totalBinCount;
            for (uint32_t binIndex = threadIdx.x; binIndex < binCount; binIndex += blockDim.x) {
                localDescriptor.contents[binIndex] = 0;
            }

            // __shared__ ShapeDescriptor::gpu::LocalReferenceFrame localLRF;
            __shared__ float3 localLRFXAxis;
            __shared__ float3 localLRFYAxis;
            __shared__ float3 localLRFZAxis;
            if (threadIdx.x == 0) {
                localLRFXAxis = localReferenceFrames.xAxisAt(descriptorIndex);
                localLRFYAxis = localReferenceFrames.yAxisAt(descriptorIndex);
                localLRFZAxis = localReferenceFrames.zAxisAt(descriptorIndex);
            }
            __syncthreads();

            const float currentSupportRadius = supportRadii.content[descriptorIndex];
            const float invSupportRadius = 1.0f / currentSupportRadius;

            for (unsigned int sampleIndex = threadIdx.x; sampleIndex < pointCloud.pointCount; sampleIndex += blockDim.x) {
                // 0. Fetch sample vertex
                const float3 samplePoint = pointCloud.vertices.at(sampleIndex);

                // 1. Compute bin indices
                const float3 translated = samplePoint - originVertex;

                // Only include vertices which are within the support radius
                float distanceToVertex = length(translated);
                if (distanceToVertex > currentSupportRadius) {
                    continue;
                }

                // Transforming descriptor coordinate system to the origin
                // TODO: Fytt ut av for-loop
                const float3 relativeSamplePoint = {
                    dot(localLRFXAxis, translated),
                    dot(localLRFYAxis, translated),
                    dot(localLRFZAxis, translated)
                };

                float2 horizontalDirection = {relativeSamplePoint.x, relativeSamplePoint.y};
                float2 verticalDirection = {length(horizontalDirection), relativeSamplePoint.z};

                if (horizontalDirection.x == 0.0f && horizontalDirection.y == 0.0f) {
                    // special case, will result in an angle of 0
                    horizontalDirection = {1, 0};

                    // Vertical direction is only 0 if all components are 0
                    // Should theoretically never occur, but let's handle it just in case
                    if (verticalDirection.y == 0) {
                        verticalDirection = {1, 0};
                    }
                }

                // normalise direction vectors
                horizontalDirection = normalize(horizontalDirection);
                verticalDirection = normalize(verticalDirection);

                const float3 sampleNormal = normalize(pointCloud.normals.at(sampleIndex));
                float normalCosine = dot(sampleNormal, localLRFZAxis);

                // For the interpolations we'll use the order used in the paper
                // a) Interpolation on normal cosines
                float cosineHistogramPosition = clamp(((0.5f * normalCosine) + 0.5f) * float(INTERNAL_HISTOGRAM_BINS), 0.0f, float(INTERNAL_HISTOGRAM_BINS));
                uint32_t cosineHistogramBinIndex = min(INTERNAL_HISTOGRAM_BINS - 1, uint32_t(cosineHistogramPosition));
                float cosineHistogramDelta = cosineHistogramPosition - (float(cosineHistogramBinIndex) + 0.5f);
                uint32_t cosineHistogramNeighbourBinIndex;
                if (cosineHistogramDelta >= 0) {
                    cosineHistogramNeighbourBinIndex = (cosineHistogramBinIndex + 1) % INTERNAL_HISTOGRAM_BINS;
                } else if (cosineHistogramDelta < 0 && cosineHistogramBinIndex > 0) {
                    cosineHistogramNeighbourBinIndex = cosineHistogramBinIndex - 1;
                } else {
                    cosineHistogramNeighbourBinIndex = INTERNAL_HISTOGRAM_BINS - 1;
                }
                float cosineHistogramBinContribution = 1.0f - abs(cosineHistogramDelta);

                // b) Interpolation on azimuth
                float azimuthAnglePosition = (::ShapeDescriptor::internal::absoluteAngle(horizontalDirection.y, horizontalDirection.x) / (2.0f * float(M_PI))) * float(AZIMUTH_DIVISIONS);
                if (azimuthAnglePosition < 0) {
                    azimuthAnglePosition += float(AZIMUTH_DIVISIONS);
                } else if (azimuthAnglePosition >= float(AZIMUTH_DIVISIONS)) {
                    azimuthAnglePosition -= float(AZIMUTH_DIVISIONS);
                }
                uint32_t azimuthBinIndex = min(AZIMUTH_DIVISIONS - 1, uint32_t(azimuthAnglePosition));
                float azimuthHistogramDelta = azimuthAnglePosition - (float(azimuthBinIndex) + 0.5f);
                uint32_t azimuthNeighbourBinIndex;
                if (azimuthHistogramDelta >= 0) {
                    azimuthNeighbourBinIndex = (azimuthBinIndex + 1) % AZIMUTH_DIVISIONS;
                } else if (azimuthHistogramDelta < 0 && azimuthBinIndex > 0) {
                    azimuthNeighbourBinIndex = azimuthBinIndex - 1;
                } else {
                    azimuthNeighbourBinIndex = AZIMUTH_DIVISIONS - 1;
                }
                float azimuthBinContribution = 1.0f - abs(azimuthHistogramDelta);


                // c) Interpolation on elevation
                float elevationAngleRaw = atan2f(verticalDirection.y, verticalDirection.x);
                float elevationAnglePosition = clamp(((elevationAngleRaw / (2.0f * float(M_PI))) + 0.5f) * float(ELEVATION_DIVISIONS), 0.0f, float(ELEVATION_DIVISIONS));
                uint32_t elevationBinIndex = min(ELEVATION_DIVISIONS - 1, uint32_t(elevationAnglePosition));
                float elevationHistogramDelta = elevationAnglePosition - (float(elevationBinIndex) + 0.5f);
                uint32_t elevationNeighbourBinIndex;
                if (elevationHistogramDelta >= 0) {
                    elevationNeighbourBinIndex = min(ELEVATION_DIVISIONS - 1, elevationBinIndex + 1);
                } else if (elevationHistogramDelta < 0) {
                    elevationNeighbourBinIndex = max(1u, elevationBinIndex) - 1;
                }
                float elevationBinContribution = 1.0f - abs(elevationHistogramDelta);


                // d) Interpolation on distance
                float layerDistanceRaw = distanceToVertex;
                float layerDistancePosition = clamp((layerDistanceRaw * invSupportRadius) * float(RADIAL_DIVISIONS), 0.0f, float(RADIAL_DIVISIONS));
                uint32_t radialBinIndex = min(RADIAL_DIVISIONS - 1, uint32_t(layerDistancePosition));
                float radialHistogramDelta = layerDistancePosition - (float(radialBinIndex) + 0.5f);
                uint32_t radialNeighbourBinIndex;
                if (radialHistogramDelta >= 0) {
                    radialNeighbourBinIndex = min(RADIAL_DIVISIONS - 1, radialBinIndex + 1);
                } else if (radialHistogramDelta < 0) {
                    radialNeighbourBinIndex = max(1u, radialBinIndex) - 1;
                } else {
                    // Should not happen
                    assert(0);
                }
                float radialBinContribution = 1.0f - abs(radialHistogramDelta);

                // Increment bins
                float primaryBinContribution = cosineHistogramBinContribution + azimuthBinContribution + elevationBinContribution + radialBinContribution;
                incrementSHOTBinDevice(localDescriptor, elevationBinIndex, radialBinIndex, azimuthBinIndex, cosineHistogramBinIndex, primaryBinContribution);
                incrementSHOTBinDevice(localDescriptor, elevationNeighbourBinIndex, radialBinIndex, azimuthBinIndex, cosineHistogramBinIndex, 1.0f - elevationBinContribution);
                incrementSHOTBinDevice(localDescriptor, elevationBinIndex, radialNeighbourBinIndex, azimuthBinIndex, cosineHistogramBinIndex, 1.0f - radialBinContribution);
                incrementSHOTBinDevice(localDescriptor, elevationBinIndex, radialBinIndex, azimuthNeighbourBinIndex, cosineHistogramBinIndex, 1.0f - azimuthBinContribution);
                incrementSHOTBinDevice(localDescriptor, elevationBinIndex, radialBinIndex, azimuthBinIndex, cosineHistogramNeighbourBinIndex, 1.0f - cosineHistogramBinContribution);
            }
            __syncthreads();

            // Insert local (shared memory) descriptor into global descriptor array
            auto &globalDescriptor = descriptors.content[descriptorIndex];
            for (uint32_t binIndex = threadIdx.x; binIndex < binCount; binIndex += blockDim.x) {
                globalDescriptor.contents[binIndex] = localDescriptor.contents[binIndex];
            }
        }

        template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
        __global__ void normalizeSHOTDescriptors(
                const ShapeDescriptor::gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors)
        {
            const uint32_t descriptorIndex = blockIdx.x;
            auto &descriptor = descriptors.content[descriptorIndex];
            constexpr uint32_t binCount = ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>::totalBinCount;

            if (threadIdx.x <= 31) {
                // Compute squared sum in parallel using warp reduction
                float threadSquaredSum = 0;
                for (uint32_t binIndex = threadIdx.x; binIndex < binCount; binIndex += 32) {
                    float total = descriptor.contents[binIndex];
                    if (isnan(total)) {
                        descriptor.contents[binIndex] = 0;
                        total = 0;
                    }
                    threadSquaredSum += total * total;
                }
                float squaredSum = ShapeDescriptor::warpAllReduceSum(threadSquaredSum);

                // Normalize descriptor in parallel
                if (squaredSum > 0) {
                    float totalLength = sqrt(squaredSum);
                    for (uint32_t binIndex = threadIdx.x; binIndex < binCount; binIndex += 32) {
                        descriptor.contents[binIndex] /= totalLength;
                    }
                }
            }
        }
}

    template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
    ShapeDescriptor::gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> generateSHOTDescriptorsMultiRadius(
            gpu::PointCloud pointCloud,
            gpu::array<OrientedPoint> descriptorOrigins,
            gpu::array<float> supportRadii,
            SHOTExecutionTimes* executionTimes = nullptr) {
        uint32_t originCount = descriptorOrigins.length;

        gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors(originCount);

        // Compute LRFs
        gpu::v3::LocalReferenceFrames referenceFrames = ShapeDescriptor::v4::computeSHOTReferenceFrames(pointCloud, descriptorOrigins, supportRadii, executionTimes);

        // Start descriptor timing
        auto startDescriptorTime = std::chrono::high_resolution_clock::now();
        // Compute SHOT descriptors
        computeGeneralisedSHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS><<<originCount, 416>>>(descriptorOrigins, pointCloud, descriptors, referenceFrames, supportRadii);
        normalizeSHOTDescriptors<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS><<<originCount, 32>>>(descriptors);

        // Synchronize and check if any errors occurred
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Got CUDA error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        std::cout << "Kernel finished -- all SHOT descriptors generated on GPU" << std::endl;
        referenceFrames.free();

        // End Descriptor timing
        auto endDescriptorTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> descriptorDuration = endDescriptorTime - startDescriptorTime;
        double descriptorTimeElapsedSeconds = descriptorDuration.count();

        if (executionTimes != nullptr) {
            // Calculate LRF generation time as sum of granular parts
            executionTimes->LRFGenerationTimeSeconds = executionTimes->covarianceMatricesGenerationTimeSeconds
                + executionTimes->EVDCalculationTimeSeconds + executionTimes->eigenvectorDisambiguationTimeSeconds;

            // Set descriptor-calculation time
            executionTimes->descriptorCalculationTimeSeconds = descriptorTimeElapsedSeconds;

            // Calculate total execution time as sum of LRF and descriptor parts
            executionTimes->totalExecutionTimeSeconds = executionTimes->LRFGenerationTimeSeconds + descriptorTimeElapsedSeconds;
        }

        return descriptors;
    }

    template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
    gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> generateSHOTDescriptors(
            gpu::PointCloud pointCloud,
            gpu::array<OrientedPoint> descriptorOrigins,
            float supportRadius,
            SHOTExecutionTimes* executionTimes = nullptr) {
        gpu::array<float> radii(descriptorOrigins.length);
        radii.setValue(supportRadius);

        return generateSHOTDescriptorsMultiRadius<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>(pointCloud, descriptorOrigins, radii, executionTimes);
    }
}
}