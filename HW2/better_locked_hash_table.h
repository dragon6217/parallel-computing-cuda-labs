#ifndef _BETTER_LOCKED_HASH_TABLE_H_
#define _BETTER_LOCKED_HASH_TABLE_H_

// TABLE_SIZE is defined in the original skeleton files
#define TABLE_SIZE 1000

#include <iostream>
#include <mutex>
#include <thread>
#include "hash_table.h"
#include "linked_list.h"

class better_locked_hash_table : public hash_table {

private:
    /**
     * @brief The number of mutexes (locks) to use for striping.
     * This is independent of TABLE_SIZE. 128 is a common choice.
     */
    static constexpr int NUM_STRIPES = 128;

    // The array of buckets, implemented as linked lists.
    linked_list* table;

    // The array of mutexes for fine-grained (striped) locking.
    std::mutex* stripes; 

public:
    better_locked_hash_table(){
        // Allocate the array of linked list buckets
        this->table = new linked_list[TABLE_SIZE]();
        
        // Allocate the array of mutexes
        this->stripes = new std::mutex[NUM_STRIPES]();
    }

    /**
     * NOTE: A destructor (~better_locked_hash_table) is intentionally
     * omitted to match the provided skeleton code's (leaky) design,
     * which does not use RAII or manage dynamic memory.
     */

    bool contains(int key){
        // 1. Calculate the hash bucket index.
        int bucket_index = key % TABLE_SIZE;
        
        // 2. Calculate the lock stripe index based on the bucket.
        int lock_index = bucket_index % NUM_STRIPES;
        
        // 3. Acquire the *specific* lock for this stripe (RAII).
        //    Other threads accessing different stripes will not be blocked.
        std::lock_guard<std::mutex> lock(stripes[lock_index]);
        
        // 4. Perform the operation on the non-thread-safe list
        //    while holding the lock.
        return this->table[bucket_index].contains(key);
    }

    
    bool insert(int key) {
        // 1. Calculate the hash bucket index.
        int bucket_index = key % TABLE_SIZE;
        
        // 2. Calculate the lock stripe index based on the bucket.
        int lock_index = bucket_index % NUM_STRIPES;
        
        // 3. Acquire the specific lock for this stripe (RAII).
        std::lock_guard<std::mutex> lock(stripes[lock_index]);
        
        // 4. Perform the operation.
        return this->table[bucket_index].insert(key);
    }

    bool remove(int key) {
        // 1. Calculate the hash bucket index.
        int bucket_index = key % TABLE_SIZE;
        
        // 2. Calculate the lock stripe index based on the bucket.
        int lock_index = bucket_index % NUM_STRIPES;
        
        // 3. Acquire the specific lock for this stripe (RAII).
        std::lock_guard<std::mutex> lock(stripes[lock_index]);
        
        // 4. Perform the operation.
        return this->table[bucket_index].remove(key);
    }
};

#endif