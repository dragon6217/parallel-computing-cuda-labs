#ifndef _HASH_TABLE_H_
#define _HASH_TABLE_H_

#include <iostream>
#include <mutex>
#include <thread>

// basic hash table structure
class hash_table {
    public:
        hash_table(){}
        virtual bool contains(int key) = 0;
        virtual bool insert(int key) = 0;
        virtual bool remove(int key) = 0;
};

#endif