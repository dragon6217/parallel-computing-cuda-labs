#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector> // Required for managing std::thread objects
#include <cmath>  // Required for std::abs
#include <algorithm> // Required for std::min

/**
 * @brief The worker function executed by each thread.
 * * Each thread processes a statically assigned 'chunk' of the 1D array.
 * * @param thread_id     The unique ID of this thread (from 0 to num_threads - 1).
 * @param num_threads   The total number of threads (NT).
 * @param N             The total size of the input/output arrays.
 * @param array_in      Pointer to the input array (read-only).
 * @param array_out     Pointer to the output array (write-only for this thread's chunk).
 * @param k             Pointer to the filter kernel (read-only).
 * @param FILTER_SIZE   The size of the filter kernel (3).
 */
void filter_worker(int thread_id, int num_threads, int N, const float* array_in, float* array_out, const float* k, const int FILTER_SIZE) 
{
    // 1. Calculate the work range for this thread.
    // The total number of items to compute is (N - 2), from index 0 to N-3.
    int total_work_items = N - 2;
    
    // Use ceiling division to ensure all items are distributed, even if not perfectly divisible.
    int items_per_thread = (total_work_items + num_threads - 1) / num_threads; 

    int start_index = thread_id * items_per_thread;
    int end_index = std::min(start_index + items_per_thread, total_work_items);

    // 2. Perform the serial filtering logic *only* for the assigned range.
    for (int i = start_index; i < end_index; i++) {
        // We assume array_out was initialized to 0, allowing +=
        for (int j = 0; j < FILTER_SIZE; j++) {
            array_out[i] += array_in[i + j] * k[j];
        }
    }
}


int main(int argc, char** argv) 
{

    if(argc < 2) std::cout<<"Usage : ./filter num_items"<<std::endl;
    int N = atoi(argv[1]);
    int NT = 32; // Default number of threads.


    //0. Initialize
    const int FILTER_SIZE = 3;
    const float k[FILTER_SIZE] = {0.25, 0.5, 0.25};
    
    // Allocate and initialize arrays to 0 using ()
    // This is crucial because the filter uses += operation.
    float *array_in = new float[N];
    float *array_out_serial = new float[N](); 
    float *array_out_parallel = new float[N]();
    
    {
        std::chrono::duration<double> diff;
        auto start = std::chrono::steady_clock::now();
        for(int i=0;i<N;i++) {
            array_in[i] = i;
        }
        auto end = std::chrono::steady_clock::now();
        diff = end-start;
        std::cout<<"init took "<<diff.count()<<" sec"<<std::endl;
    }

    {
        //1. Serial
        std::chrono::duration<double> diff;
        auto start = std::chrono::steady_clock::now();
        
        // Original serial logic
        for(int i=0;i<N-2;i++) {
            for(int j=0;j<FILTER_SIZE;j++) {
                array_out_serial[i] += array_in[i+j] * k[j];
            }
        }
        
        auto end = std::chrono::steady_clock::now();
        diff = end-start;
        std::cout<<"serial 1D filter took "<<diff.count()<<" sec"<<std::endl;
    }

    {
        //2. parallel 1D filter
        std::chrono::duration<double> diff;
        auto start = std::chrono::steady_clock::now();
        
        /* TODO: parallelized 1D filter implementation */
        /****************/
        
        std::vector<std::thread> threads;
        threads.reserve(NT); // Pre-allocate space for efficiency

        // Launch all threads
        for (int t = 0; t < NT; t++) {
            threads.push_back(std::thread(
                filter_worker, 
                t,          // thread_id
                NT,         // num_threads
                N, 
                array_in, 
                array_out_parallel, 
                k, 
                FILTER_SIZE
            ));
        }

        // Wait for all threads to complete
        for (auto& th : threads) {
            th.join();
        }

        /****************/
        /* TODO: end of parallel implementation */
        auto end = std::chrono::steady_clock::now();
        diff = end-start;
        std::cout<<"parallel 1D filter took "<<diff.count()<<" sec"<<std::endl;


        // Validation check
        int error_counts=0;
        const float epsilon = 0.01;
        
        // Validate only up to N-2, as this is the computed range
        for(int i=0;i<N-2;i++) { 
            float err= std::abs(array_out_serial[i] - array_out_parallel[i]);
            if(err > epsilon) {
                error_counts++;
                if(error_counts < 5) {
                    std::cout<<"ERROR at "<<i<<": Serial["<<i<<"] = "<<array_out_serial[i]<<" Parallel["<<i<<"] = "<<array_out_parallel[i]<<std::endl;
                    std::cout<<"err: "<<err<<std::endl;
                }
            }
        }


        if(error_counts==0) {
            std::cout<<"PASS"<<std::endl;
        } else {
            std::cout<<"There are "<<error_counts<<" errors"<<std::endl;
        }

    }
    
    // Clean up allocated memory
    delete[] array_in;
    delete[] array_out_serial;
    delete[] array_out_parallel;
    
    return 0;
}