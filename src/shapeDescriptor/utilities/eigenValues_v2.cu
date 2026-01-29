// V2: SHARED MEMORY (TODO)
#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif

namespace ShapeDescriptor {
namespace v2 {
void checkCuSolverStatus(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to execute cuSOLVER function!\n");
        exit(EXIT_FAILURE);
    }
}

__device__ inline void swap(float &a, float &b)
{
    float temp = a;
    a = b;
    b = temp;
}

__device__ inline void swapColumns(float *eigenvectors, int columnA, int columnB)
{
    for (int r = 0; r < 3; ++r)
    {
        int i1 = r + columnA * 3;
        int i2 = r + columnB * 3;
        float temp = eigenvectors[i1];
        eigenvectors[i1] = eigenvectors[i2];
        eigenvectors[i2] = temp;
    }
}

__global__ void sortEigenvectors(float *d_allEigenvectors, float *d_allEigenvalues, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize)
        return;

    float* eigenvectors = d_allEigenvectors + idx * 3 * 3; // pointer to start of idx-th 3x3 matrix
    float* eigenvalues = d_allEigenvalues + idx * 3;       // pointer to eigenvalues for idx-th 3x3 matrix

    int i0 = 0, i1 = 1, i2 = 2;
    if (eigenvalues[i0] < eigenvalues[i1])
    {
        swap(eigenvalues[i0], eigenvalues[i1]);
        swapColumns(eigenvectors, i0, i1);
    }
    if (eigenvalues[i0] < eigenvalues[i2])
    {
        swap(eigenvalues[i0], eigenvalues[i2]);
        swapColumns(eigenvectors, i0, i2);
    }
    if (eigenvalues[i1] < eigenvalues[i2])
    {
        swap(eigenvalues[i1], eigenvalues[i2]);
        swapColumns(eigenvectors, i1, i2);
    }
}

ShapeDescriptor::gpu::array<float> computeEigenVectorsMultiple(ShapeDescriptor::gpu::array<float> d_columnMajorMatrices, uint32_t nMatrices) {
    cusolverDnHandle_t cusolverHandle;
    checkCuSolverStatus(cusolverDnCreate(&cusolverHandle));

    const int n = 3; // Matrix dimension (how many columns)
    const int lda = 3; // Leading dimension (how many rows)

    // Allocate device memory for eigenvalues and workspace
    float *d_eigenvalues;
    checkCudaErrors(cudaMalloc(&d_eigenvalues, nMatrices * n * sizeof(float)));
    
    // 1. Query workspace size
    int lwork = 0;

    syevjInfo_t params = NULL;
    checkCuSolverStatus(cusolverDnCreateSyevjInfo(&params));

    // Perform the actual query
    checkCuSolverStatus(cusolverDnSsyevjBatched_bufferSize(
        cusolverHandle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_columnMajorMatrices.content,
        lda,
        d_eigenvalues,
        &lwork,
        params,
        nMatrices));

    float *d_work;
    checkCudaErrors(cudaMalloc(&d_work, lwork * sizeof(float)));
    
    int *d_info;
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int)));

    // 2. Solve eigenvalue problem for each matrix
    // Compute all eigendecompositions in a single batched call (in-place)
    checkCuSolverStatus(cusolverDnSsyevjBatched(
        cusolverHandle,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_columnMajorMatrices.content,
        lda,
        d_eigenvalues,
        d_work,
        lwork,
        d_info,
        params,
        nMatrices));

    // 3. Order the eigenvectors by their lengths (i.e. by their eigenvalues)
    int dimBlock = 128;
    int dimGrid = (nMatrices + dimBlock - 1) / dimBlock;
    sortEigenvectors<<<dimGrid, dimBlock>>>(d_columnMajorMatrices.content, d_eigenvalues, nMatrices);

    // Clean up
    checkCudaErrors(cudaFree(d_eigenvalues));
    checkCudaErrors(cudaFree(d_work));
    checkCudaErrors(cudaFree(d_info));
    checkCuSolverStatus(cusolverDnDestroy(cusolverHandle));

    // 4. Create output gpu::array
    return d_columnMajorMatrices;
}
}
}