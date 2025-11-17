#ifndef _LOCKED_HASH_TABLE_H_
#define _LOCKED_HASH_TABLE_H_

#define TABLE_SIZE 1000

#include <iostream>
#include <mutex>
#include <thread>
#include "hash_table.h"
#include "linked_list.h"

class locked_hash_table : public hash_table {

    linked_list* table;
    std::mutex global_mutex; 

    public:
        locked_hash_table(){
            this->table = new linked_list[TABLE_SIZE]();
        }

        bool contains(int key){
            // Lock guard
            std::lock_guard<std::mutex> lock(global_mutex);
            int hash_value = key%TABLE_SIZE;

            return this->table[hash_value].contains(key);
        }

        
        bool insert(int key) {
            // lock guard
            std::lock_guard<std::mutex> lock(global_mutex);
            // move to left of key
            int hash_value = key%TABLE_SIZE;

            return this->table[hash_value].insert(key);
        }

        bool remove(int key) {
            // lock guard
            std::lock_guard<std::mutex> lock(global_mutex);
            // if empty do nothing
            int hash_value = key%TABLE_SIZE;

            return this->table[hash_value].remove(key);
        }
};

#endif