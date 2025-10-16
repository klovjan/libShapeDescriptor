#include <shapeDescriptor/shapeDescriptor.h>
#include <iostream>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#endif

// Threads per block in each dimension
#define BLOCKX 1

// Sample rates
// #define N_POINTCLOUD_SAMPLES 5000000
// #define N_KEYPOINTS 500
#define N_POINTCLOUD_SAMPLES 50
#define N_KEYPOINTS 10

// Randomness
#define RANDOM_SEED 1

// Host variables
const int h_n_pointcloud_samples = N_POINTCLOUD_SAMPLES;

// Device variables
__constant__ int d_n_pointcloud_samples;  // Could just leave this as 

// Define CUDA grid and block dimensions
int dim_block = BLOCKX;
int dim_grid = 1;

__global__ void calculate_descriptors(ShapeDescriptor::gpu::PointCloud device_point_cloud, ShapeDescriptor::gpu::array<int> device_keypoint_indexes, ShapeDescriptor::gpu::array<int> device_output_array) {
    int keypoints_per_thread = N_KEYPOINTS / blockDim.x;

    for (int i = threadIdx.x * keypoints_per_thread; i < (threadIdx.x + 1) * keypoints_per_thread; i++)
    {
        if (i >= N_KEYPOINTS) break;

        float3 keypoint_vertex = device_point_cloud.vertices.at(device_keypoint_indexes[i]);
        float3 keypoint_normal = device_point_cloud.normals.at(device_keypoint_indexes[i]);
        
        // ShapeDescriptor::OrientedPoint keypoint;
        // keypoint.vertex = keypoint_vertex;
        // keypoint.normal = keypoint_normal;

        device_output_array.content[i] = keypoint_vertex.x;
    }
}

int main(int argc, char **argv) {
    if (argc == 1)
    {
        std::cout << "Usage: simple_gpu [file_to_read.obj/.ply/.off]" << std::endl;
        return 1;
    }

    // Load mesh
    std::string fileToRead = std::string(argv[1]);
    ShapeDescriptor::cpu::Mesh h_mesh = ShapeDescriptor::loadMesh(fileToRead, ShapeDescriptor::RecomputeNormals::RECOMPUTE_IF_MISSING);

    // // Copy mesh to GPU
    // ShapeDescriptor::gpu::Mesh d_mesh = ShapeDescriptor::copyToGPU(h_mesh);

    // Sample point cloud
    ShapeDescriptor::cpu::PointCloud h_sampled_point_cloud = ShapeDescriptor::sampleMesh(h_mesh, N_POINTCLOUD_SAMPLES, RANDOM_SEED);
    ShapeDescriptor::gpu::PointCloud d_sampled_point_cloud = ShapeDescriptor::copyToGPU(h_sampled_point_cloud);

    // Choose random subset of points from point cloud
    std::cout << "TODO" << std::endl;

    // Choose 10 first points in pointcloud (TEMPORARY)
    ShapeDescriptor::cpu::array<int> h_keypoint_indexes(N_KEYPOINTS);
    for (int i = 0; i < N_KEYPOINTS; i++)
    {
        h_keypoint_indexes[i] = i;
    }
    ShapeDescriptor::gpu::array<int> d_keypoint_indexes = ShapeDescriptor::copyToGPU(h_keypoint_indexes);

    // Set up device memory for output array
    ShapeDescriptor::gpu::array<int> d_output_array(N_KEYPOINTS);

    // Launch kernel
    calculate_descriptors<<<dim_grid, dim_block>>>(d_sampled_point_cloud, d_keypoint_indexes, d_output_array);

    // Synchronize and check if any errors occurred
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Got CUDA error: %s\n", cudaGetErrorString(err));
    }
    std::cout << "Kernel finished" << std::endl;

    // Move output to CPU
    ShapeDescriptor::cpu::array<int> h_output_array = ShapeDescriptor::copyToCPU(d_output_array);
    
    // Print output
    for (int i = 0; i < N_KEYPOINTS; i++)
    {
        std::cout << h_output_array.content[i] << std::endl;
    }

    // Does it match?
    for (int i = 0; i < N_KEYPOINTS; i++)
    {
        std::cout << h_sampled_point_cloud.vertices[i].x << std::endl;
    }
    

    // Free the memory
    ShapeDescriptor::free(d_output_array);
    ShapeDescriptor::free(h_mesh);
    ShapeDescriptor::free(h_sampled_point_cloud);
    ShapeDescriptor::free(d_sampled_point_cloud);
    ShapeDescriptor::free(d_keypoint_indexes);
    // ShapeDescriptor::free(d_mesh);

    return 0;
}