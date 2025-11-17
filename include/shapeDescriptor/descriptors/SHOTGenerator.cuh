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
namespace internal {
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
        __global__
        void computeGeneralisedSHOTDescriptor(
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const ShapeDescriptor::gpu::PointCloud pointCloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> localReferenceFrames,
                const ShapeDescriptor::gpu::array<float> supportRadii)
        {
            uint32_t descriptorIndex = blockIdx.x;

            const float3 originVertex = descriptorOrigins.content[descriptorIndex].vertex;

            // Initialize descriptor to zero
            if (threadIdx.x == 0) {
                auto &descriptor = descriptors.content[descriptorIndex];
                for (uint32_t binIndex = 0; binIndex < descriptor.totalBinCount; binIndex++) {
                    descriptor.contents[binIndex] = 0;
                }
            }
            __syncthreads();

            for (unsigned int sampleIndex = threadIdx.x; sampleIndex < pointCloud.pointCount; sampleIndex += blockDim.x) {
                // 0. Fetch sample vertex
                const float3 samplePoint = pointCloud.vertices.at(sampleIndex);

                // 1. Compute bin indices
                const float3 translated = samplePoint - originVertex;

                // Only include vertices which are within the support radius
                float distanceToVertex = length(translated);
                float currentSupportRadius = supportRadii.content[descriptorIndex];
                if (distanceToVertex > currentSupportRadius) {
                    continue;
                }

                // Transforming descriptor coordinate system to the origin
                const float3 relativeSamplePoint = {
                    dot(localReferenceFrames.content[descriptorIndex].xAxis, translated),
                    dot(localReferenceFrames.content[descriptorIndex].yAxis, translated),
                    dot(localReferenceFrames.content[descriptorIndex].zAxis, translated)
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
                float normalCosine = dot(sampleNormal, localReferenceFrames.content[descriptorIndex].zAxis);


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
                float cosineHistogramNeighbourBinContribution = 1.0f - cosineHistogramBinContribution;

                // b) Interpolation on azimuth
                float azimuthAnglePosition = (internal::absoluteAngle(horizontalDirection.y, horizontalDirection.x) / (2.0f * float(M_PI))) * float(AZIMUTH_DIVISIONS);
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
                float azimuthNeighbourBinContribution = 1.0f - azimuthBinContribution;


                // c) Interpolation on elevation
                float elevationAngleRaw = std::atan2(verticalDirection.y, verticalDirection.x);
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
                float elevationNeighbourBinContribution = 1.0f - elevationBinContribution;


                // d) Interpolation on distance
                float layerDistanceRaw = distanceToVertex;
                float layerDistancePosition = clamp((layerDistanceRaw / currentSupportRadius) * float(RADIAL_DIVISIONS), 0.0f, float(RADIAL_DIVISIONS));
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
                float radialNeighbourBinContribution = 1.0f - radialBinContribution;

                // Increment bins
                float primaryBinContribution = cosineHistogramBinContribution + azimuthBinContribution + elevationBinContribution + radialBinContribution;
                incrementSHOTBinDevice(descriptors.content[descriptorIndex], elevationBinIndex, radialBinIndex, azimuthBinIndex, cosineHistogramBinIndex, primaryBinContribution);
                incrementSHOTBinDevice(descriptors.content[descriptorIndex], elevationNeighbourBinIndex, radialBinIndex, azimuthBinIndex, cosineHistogramBinIndex, elevationNeighbourBinContribution);
                incrementSHOTBinDevice(descriptors.content[descriptorIndex], elevationBinIndex, radialNeighbourBinIndex, azimuthBinIndex, cosineHistogramBinIndex, radialNeighbourBinContribution);
                incrementSHOTBinDevice(descriptors.content[descriptorIndex], elevationBinIndex, radialBinIndex, azimuthNeighbourBinIndex, cosineHistogramBinIndex, azimuthNeighbourBinContribution);
                incrementSHOTBinDevice(descriptors.content[descriptorIndex], elevationBinIndex, radialBinIndex, azimuthBinIndex, cosineHistogramNeighbourBinIndex, cosineHistogramNeighbourBinContribution);
            }
            __syncthreads();

            // Normalise descriptor
            // NOTE: Done on only one thread per block, i.e. one thread per origin (for now?)
            if (threadIdx.x == 0) {
                uint32_t binCount = ELEVATION_DIVISIONS * RADIAL_DIVISIONS * AZIMUTH_DIVISIONS * INTERNAL_HISTOGRAM_BINS;
                double squaredSum = 0;
                for (int i = 0; i < binCount; i++) {
                    double total = descriptors.content[descriptorIndex].contents[i];
                    if (std::isnan(total)) {
                        descriptors.content[descriptorIndex].contents[i] = 0;
                        total = 0;
                    }
                    squaredSum += total * total;
                }
                if (squaredSum > 0) {
                    double totalLength = sqrt(squaredSum);
                    for (int i = 0; i < binCount; i++) {
                        descriptors.content[descriptorIndex].contents[i] /= totalLength;
                    }
                }
            }
        }
}

    template<uint32_t ELEVATION_DIVISIONS = 2, uint32_t RADIAL_DIVISIONS = 2, uint32_t AZIMUTH_DIVISIONS = 8, uint32_t INTERNAL_HISTOGRAM_BINS = 11>
    gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> generateSHOTDescriptorsMultiRadius(
            gpu::PointCloud pointCloud,
            gpu::array<OrientedPoint> descriptorOrigins,
            gpu::array<float> supportRadii,
            SHOTExecutionTimes* executionTimes = nullptr) {
        gpu::array<ShapeDescriptor::SHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS>> descriptors(descriptorOrigins.length);

        // Compute LRFs
        gpu::array<ShapeDescriptor::gpu::LocalReferenceFrame> referenceFrames = ShapeDescriptor::internal::computeSHOTReferenceFrames(pointCloud, descriptorOrigins, supportRadii, executionTimes);

        // Start descriptor timing
        auto startDescriptorTime = std::chrono::high_resolution_clock::now();
        // Compute SHOT descriptors
        internal::computeGeneralisedSHOTDescriptor<ELEVATION_DIVISIONS, RADIAL_DIVISIONS, AZIMUTH_DIVISIONS, INTERNAL_HISTOGRAM_BINS><<<descriptorOrigins.length, 416>>>(descriptorOrigins, pointCloud, descriptors, referenceFrames, supportRadii);

        // Synchronize and check if any errors occurred
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Got CUDA error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        std::cout << "Kernel finished -- all SHOT descriptors generated on GPU" << std::endl;
        ShapeDescriptor::free(referenceFrames);

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