/**
 * @file jointable.cpp
 * @brief Parallel Sort-Merge Join implementation using OpenMP and GCC Parallel Extensions.
 */

#include "jointable.h"
#include <omp.h>
#include <vector>
#include <parallel/algorithm> // __gnu_parallel::sort
#include <algorithm>

/**
 * @brief Reference Nested Loop Join implementation (O(R*S)).
 */
void jointable_ref(const long long* const tableA, const long long* const tableB,
                   std::vector<long long>* const solution, const int R, const int S) {
  for (long long i = 0; i < R; i++)
    for (long long j = 0; j < S; j++)
      if (tableA[i] == tableB[j]) {
        solution->push_back(tableA[i]);
        break;
      }
}

/**
 * @brief Optimized Join implementation.
 * Uses Sort-Merge Join strategy to reduce time complexity to O(N log N).
 * Parallelization is applied to Data Copy and Sorting phases.
 */
void jointable_optimized(const long long* const tableA, const long long* const tableB,
                         std::vector<long long>* const solution, const int R,
                         const int S) {
    
    // 1. Data Copy (Parallelized with OpenMP sections)
    std::vector<long long> vecA(R);
    std::vector<long long> vecB(S);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            #pragma omp parallel for
            for (int i = 0; i < R; ++i) vecA[i] = tableA[i];
        }

        #pragma omp section
        {
            #pragma omp parallel for
            for (int i = 0; i < S; ++i) vecB[i] = tableB[i];
        }
    }

    // 2. Parallel Sort
    // __gnu_parallel::sort uses OpenMP internally for efficient parallel sorting.
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            __gnu_parallel::sort(vecA.begin(), vecA.end());
        }
        #pragma omp section
        {
            __gnu_parallel::sort(vecB.begin(), vecB.end());
        }
    }

    // 3. Merge (Linear Scan)
    int i = 0;
    int j = 0;

    // Standard merge logic to find intersection
    while (i < R && j < S) {
        if (vecA[i] < vecB[j]) {
            i++;
        } else if (vecA[i] > vecB[j]) {
            j++;
        } else {
            // Match found
            solution->push_back(vecA[i]);
            i++;
            j++;
        }
    }
}