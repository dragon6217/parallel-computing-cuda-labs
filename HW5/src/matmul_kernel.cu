#include <stdio.h>
#include <iostream>
#include <chrono>
#include <assert.h>
#include "matmul.h"
#include <cuda_runtime.h> // CUDA runtime header

using namespace std;

// Tiling Block Size
#define TILE_WIDTH 32

void allocateDeviceMemory(void** M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void deallocateDeviceMemory(void* M)
{
    cudaError_t err = cudaFree(M);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
    // You can assume matrixC is initialized with zero
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

/**
 * @brief CUDA Kernel using Shared Memory Tiling
 */
__global__ void matmul_tiled_kernel(const int* A, const int* B, int* C, int n) {
    
    // Shared memory for tiles of A and B
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    // Calculate global row and column index
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int Cvalue = 0;

    // Loop over the tiles of input matrices A and B
    // m is the tile index
    for (int m = 0; m < (n / TILE_WIDTH); ++m) {
        
        // Load data from Global Memory to Shared Memory
        // Collaboratively load tiles
        ds_A[ty][tx] = A[Row * n + (m * TILE_WIDTH + tx)];
        ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * n + Col];

        // Synchronization to ensure the tile is loaded
        __syncthreads();

        // Compute matrix multiplication for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        // Synchronization before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory
    if (Row < n && Col < n) {
        C[Row * n + Col] = Cvalue;
    }
}


void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int* const d_A, const int* const d_B, int* const d_C, const int n) {
    
    // 1. Copy Data: Host -> Device
    // This memcpy time IS INCLUDED in the grading measurement.
    size_t size = n * n * sizeof(int);
    cudaMemcpy((void*)d_A, matrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_B, matrixB, size, cudaMemcpyHostToDevice);

    // 2. Launch Kernel
    // Use 2D Grid and 2D Blocks
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(n / TILE_WIDTH, n / TILE_WIDTH);

    matmul_tiled_kernel<<<dimGrid, dimBlock>>>((int*)d_A, (int*)d_B, (int*)d_C, n);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // 3. Copy Result: Device -> Host
    cudaMemcpy(matrixC, d_C, size, cudaMemcpyDeviceToHost);
}