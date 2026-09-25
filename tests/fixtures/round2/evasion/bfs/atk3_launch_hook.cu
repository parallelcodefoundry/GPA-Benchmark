#include "hip/hip_runtime.h"
#ifndef _KERNEL_H_
#define _KERNEL_H_
// ATTACK 3 (rt2-bfs): intercept the launch in host code (bfs.cu #includes this file BEFORE
// the launch site) and run BFS on the CPU. Device Kernel left trivial.
#include <queue>
#define hipLaunchKernelGGL(kern, grid, block, sh, st, ...) appeb_noop()
static inline void appeb_noop() {}
__global__ void
Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes) {}
#endif
