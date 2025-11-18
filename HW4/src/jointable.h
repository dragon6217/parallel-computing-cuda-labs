#ifndef JOINTABLE_H
#define JOINTABLE_H

#include <omp.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

void jointable_ref(const long long* const tableA, const long long* const tableB,
                   std::vector<long long>* const solution, const int R, const int S);
void jointable_optimized(const long long* const tableA, const long long* const tableB,
                         std::vector<long long>* const solution, const int R,
                         const int S);

#endif
