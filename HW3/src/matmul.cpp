#include "matmul.h"
#include <omp.h>      // Required for OpenMP
#include <algorithm>  // Required for std::min

/**
 * @brief Reference (naive) matrix multiplication.
 * Provided as a baseline, already using 1D indexing.
 */
void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
    // You can assume matrixC is initialized with zero
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

/**
 * @brief Optimized matrix multiplication using Cache Blocking (Tiling)
 * and OpenMP parallelization.
 */
void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n) {
    
    /**
     * @brief Tiling (Cache Blocking)
     * We block the matrices into smaller 'tiles' that fit into the CPU cache
     * (e.g., L1 cache, 32KB). This avoids constant, slow access to main memory (RAM).
     *
     * A block size of 32x32 for int (4 bytes) is chosen:
     * 3 * (32 * 32 * 4 bytes) = 12,288 bytes (approx 12KB)
     * This fits comfortably inside the 32KB L1 cache.
     */
    const int BLOCK_SIZE = 32;

    /**
     * @brief OpenMP Parallelization
     * We parallelize the outermost loop (the 'ii' or 'i_block' loop).
     * Each thread (e.g., 16 threads) will process a different
     * set of 'block rows' of the C matrix.
     * 'collapse(2)' could also be used on 'ii' and 'jj' for finer task
     * distribution, but simple 'parallel for' is clean and effective.
     */
    #pragma omp parallel for
    for (int ii = 0; ii < n; ii += BLOCK_SIZE) { // Outer loop for i-blocks (ii)
        
        for (int jj = 0; jj < n; jj += BLOCK_SIZE) { // Outer loop for j-blocks (jj)
            
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) { // Outer loop for k-blocks (kk)
                
                // This 6-loop structure computes C[ii:ii+B][jj:jj+B] += A[ii:ii+B][kk:kk+B] * B[kk:kk+B][jj:jj+B]
                
                // --- Inner loops compute one tile ---
                // We use std::min to handle edge cases where n is not
                // perfectly divisible by BLOCK_SIZE.
                int i_end = std::min(ii + BLOCK_SIZE, n);
                int j_end = std::min(jj + BLOCK_SIZE, n);
                int k_end = std::min(kk + BLOCK_SIZE, n);

                for (int i = ii; i < i_end; ++i) { // Inner loop for i
                    for (int k = kk; k < k_end; ++k) { // Inner loop for k
                        
                        // Cache-friendly optimization:
                        // Load A[i][k] into a register once.
                        int r = matrixA[i * n + k];
                        
                        for (int j = jj; j < j_end; ++j) { // Inner loop for j
                            // C[i][j] += A[i][k] * B[k][j]
                            matrixC[i * n + j] += r * matrixB[k * n + j];
                        }
                    }
                }
                // --- End of inner tile computation ---
            }
        }
    }
}