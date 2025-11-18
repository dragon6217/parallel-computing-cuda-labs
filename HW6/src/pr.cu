// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>
#include <set>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "gpu_error_check.cuh"

using namespace std;

/* Editing is Prohibited*/
typedef float ScoreT;
const float kDamp = 0.85;
vector<double> total_proc_times;

vector<pair<ScoreT, NodeID>> PrintTopScores(const Graph  &g, ScoreT *scores);
vector<pair<ScoreT, NodeID>> PrintTopScores(const Graph &g, pvector<ScoreT> &scores);
bool CompareScores(vector<pair<ScoreT, NodeID>> &result, vector<pair<ScoreT, NodeID>> &answer);
/* Editing is Prohibited */

// PageRank Reference function
void PageRankPull(const Graph &g, ScoreT *scores, int num_iterations) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  const ScoreT init_score = 1.0f / g.num_nodes();
#pragma omp parallel for
  for (NodeID u=0; u < g.num_nodes(); u++) {
    scores[u] = init_score;
  }

  for (int iter=0; iter < num_iterations; iter++) {
#pragma omp parallel for
    for (NodeID u=0; u < g.num_nodes(); u++) {
      ScoreT sum = 0;
      for (NodeID v : g.in_neigh(u))
      {
        debug_print("v is %d\n",v);
        sum += scores[v] / g.out_degree(v) ;
      }
      // ScoreT old_score = scores[u];
      scores[u] = base_score + kDamp * sum;
    }
  }

  printf("end iterations\n");


  return;
}

/**
 * @brief CUDA Kernel for PageRank (Pull-based)
 * Each thread handles one node 'u', summing contributions from incoming neighbors 'v'.
 */
__global__ void pagerank_kernel(int num_nodes, const index_t* row_ptr, const NodeID* col_ind,
                                const ScoreT* scores, const int* out_degree,
                                ScoreT* new_scores, float base_score, float damp) {
    
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < num_nodes) {
        ScoreT sum = 0.0f;
        
        // Iterate over incoming edges (v -> u)
        // CSR format: row_ptr[u] to row_ptr[u+1] contains neighbors of u
        index_t start = row_ptr[u];
        index_t end = row_ptr[u+1];

        for (index_t i = start; i < end; ++i) {
            NodeID v = col_ind[i];
            // Contribution = PR(v) / OutDegree(v)
            // We assume out_degree[v] > 0 (handled in preprocessing or graph construction)
            if (out_degree[v] > 0) {
                sum += scores[v] / (float)out_degree[v];
            }
        }

        // Update PageRank score
        new_scores[u] = base_score + damp * sum;
    }
}

/**
 * @brief CUDA Kernel to initialize scores
 */
__global__ void init_scores_kernel(int num_nodes, ScoreT* scores, float init_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        scores[idx] = init_val;
    }
}

vector<pair<ScoreT, NodeID>> PageRankCuda(const Graph &g, int num_iterations) {
  /* Editing is Prohibited*/
  Timer alloc_timer;
  alloc_timer.Start();
  double total_proc_time = 0;
  ScoreT* results = new ScoreT[g.num_nodes()]; // you must store pagerank result in here!
  /* Editing is Prohibited*/

  /* MEMORY ALLOCATION SECTION START */
  
  int num_nodes = g.num_nodes();
  int64_t num_edges = g.num_edges_directed(); // Directed edges count

  // Host variables for preprocessing
  int* h_out_degree = new int[num_nodes];

  // Device pointers
  index_t* d_row_ptr;
  NodeID* d_col_ind;
  ScoreT* d_scores;
  ScoreT* d_new_scores;
  int* d_out_degree;

  // Allocate memory on GPU
  // Note: We need to store the INVERSE graph (incoming edges) for Pull-based PageRank.
  // Fortunately, the Graph class (graph.h) stores in_vertex_table_ and in_edge_table_.
  
  gpuErrorcheck(cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(index_t)));
  gpuErrorcheck(cudaMalloc(&d_col_ind, num_edges * sizeof(NodeID)));
  gpuErrorcheck(cudaMalloc(&d_scores, num_nodes * sizeof(ScoreT)));
  gpuErrorcheck(cudaMalloc(&d_new_scores, num_nodes * sizeof(ScoreT)));
  gpuErrorcheck(cudaMalloc(&d_out_degree, num_nodes * sizeof(int)));

  /* MEMORY ALLOCATION SECTION END   */

  /* Editing is Prohibited*/
  alloc_timer.Stop();
  PrintTime("alloc Time ", alloc_timer.Seconds());

  Timer preproc_timer;
  preproc_timer.Start();
  /* Editing is Prohibited*/

  /* PREPROCESSING SECTION START */
  
  // Prepare out-degree array on Host
  // We need Out-Degree of 'v' to calculate PR contribution.
  // Graph 'g' provides out_degree(v). We flatten this to an array for GPU access.
  #pragma omp parallel for
  for (NodeID n = 0; n < num_nodes; ++n) {
      h_out_degree[n] = g.out_degree(n);
  }

  /* PREPROCESSING SECTION END   */

  /* Editing is Prohibited*/
  preproc_timer.Stop();
  PrintTime("preprocessing Time ", preproc_timer.Seconds());
  total_proc_time += 0.1 * preproc_timer.Seconds();

  Timer trial_timer;
  trial_timer.Start();
  /* Editing is Prohibited*/
  
  /* PAGERANK KERNEL SECTION START */
  
  // 1. Copy Graph Data (Host -> Device)
  // We use in_vertex_table_ (Row Ptr) and in_edge_table_ (Col Ind) for Pull-based PR.
  gpuErrorcheck(cudaMemcpy(d_row_ptr, g.in_vertex_table_, (num_nodes + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
  gpuErrorcheck(cudaMemcpy(d_col_ind, g.in_edge_table_, num_edges * sizeof(NodeID), cudaMemcpyHostToDevice));
  
  // 2. Copy Out-Degree Data (Host -> Device)
  gpuErrorcheck(cudaMemcpy(d_out_degree, h_out_degree, num_nodes * sizeof(int), cudaMemcpyHostToDevice));

  // 3. Constants
  const float base_score = (1.0f - kDamp) / num_nodes;
  const float init_score = 1.0f / num_nodes;

  // 4. Kernel Configuration
  const int blockSize = 256;
  const int gridSize = (num_nodes + blockSize - 1) / blockSize;

  // 5. Initialize Scores on GPU
  init_scores_kernel<<<gridSize, blockSize>>>(num_nodes, d_scores, init_score);
  gpuErrorcheck(cudaPeekAtLastError());

  // 6. PageRank Iterations
  for (int iter = 0; iter < num_iterations; ++iter) {
      
      pagerank_kernel<<<gridSize, blockSize>>>(
          num_nodes,
          d_row_ptr,
          d_col_ind,
          d_scores,     // Read from current scores
          d_out_degree,
          d_new_scores, // Write to new scores
          base_score,
          kDamp
      );
      gpuErrorcheck(cudaPeekAtLastError());

      // Swap pointers for next iteration (Double Buffering)
      std::swap(d_scores, d_new_scores);
  }

  // 7. Copy Results Back (Device -> Host)
  // Note: After loop finishes, the latest valid scores are in 'd_scores' (because of swap)
  gpuErrorcheck(cudaMemcpy(results, d_scores, num_nodes * sizeof(ScoreT), cudaMemcpyDeviceToHost));
  
  // Cleanup Device Memory (Technically should be in destructor or separate cleanup, 
  // but here we do it at end of kernel run or leave it if reuse is intended. 
  // Given the strict sections, cleanup is usually implicit or done in allocation section of next run.
  // However, to avoid leaks in repeated trials:
  gpuErrorcheck(cudaFree(d_row_ptr));
  gpuErrorcheck(cudaFree(d_col_ind));
  gpuErrorcheck(cudaFree(d_scores));
  gpuErrorcheck(cudaFree(d_new_scores));
  gpuErrorcheck(cudaFree(d_out_degree));

  /* PAGERANK KERNEL SECTION END   */

  // Cleanup Host Memory
  delete[] h_out_degree;

  /* Editing is Prohibited*/
  trial_timer.Stop();
  PrintTime("trial Time including memcpy", trial_timer.Seconds());
  total_proc_time += trial_timer.Seconds();
  
  total_proc_times.push_back(total_proc_time);
  return PrintTopScores(g, results);
  /* Editing is Prohibited*/
}




/* WARNING!!!!*/
/* Don't touch below code!!!*/

vector<pair<ScoreT, NodeID>> PrintTopScores(const Graph &g, ScoreT *scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout<<"Printing Top5 Ranks"<<endl;
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;

  return top_k;
}

vector<pair<ScoreT, NodeID>> PrintTopScores(const Graph &g, pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout<<"Printing Top5 Ranks"<<endl;
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;

  return top_k;
}

bool CompareScores(vector<pair<ScoreT, NodeID>> &result, vector<pair<ScoreT, NodeID>> &answer) {
  int total_pass = 0;

  for (int i = 0; i < 5; i++){
    bool check_1 = false, check_2 = false;
    if (result[i].second == answer[i].second)
      check_1 = true;

    if ((result[i].first / answer[i].first) >= 0.9 && (result[i].first / answer[i].first) <= 1.1)
      check_2 = true;

    if (check_1 && check_2)
      total_pass++;
  }

  if (total_pass == 5){
    cout<<"PASS!! your total pass: "<<total_pass<<endl;
    return true;
  }
  else {
    cout<<"NON-PASS!! your total pass: "<<total_pass<<endl;
    return false;
  }
}

int main(int argc, char* argv[]) {
  int total_pass = 0;

  CLIterApp cli(argc, argv, "pagerank", 100);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  bool answer_exist = (cli.answer_file_name() != "");
  vector<pair<ScoreT, NodeID>> answer;
  vector<pair<ScoreT, NodeID>> result;

  if (answer_exist)
    answer = b.ReadAnswerFile();

  for(int t=0;t<cli.num_trials();t++) {
    result = PageRankCuda(g, cli.num_iters());
    if (answer_exist) {
      if (CompareScores(result, answer))
        total_pass++;
    }
  }

  if (answer_exist)
    cout<<"PageRank End. Your Pass Score: "<<total_pass<<", Mininum Runtime: "<<*min_element(total_proc_times.begin(), total_proc_times.end())<<endl;
  else
    cout<<"PageRank End. Minimum Runtime: "<<*min_element(total_proc_times.begin(), total_proc_times.end())<<endl;

  return 0;
}
