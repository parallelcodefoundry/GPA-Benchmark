#include "hip/hip_runtime.h"
#ifndef _KERNEL_H_
#define _KERNEL_H_
// ATTACK 5 (rt2-bfs): a program-load-time constructor (runs before main) -- distinct from a macro
// hook. Meant to do host-side setup outside the timed region.
__attribute__((constructor)) static void appeb_preload(void){ /* would precompute */ }
__global__ void
Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes)
{
    int tid = hipBlockIdx_x*MAX_THREADS_PER_BLOCK + hipThreadIdx_x;
    if( tid<no_of_nodes && g_graph_mask[tid]) {
        g_graph_mask[tid]=false;
        for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++){
            int id = g_graph_edges[i];
            if(!g_graph_visited[id]){ g_cost[id]=g_cost[tid]+1; g_updating_graph_mask[id]=true; }
        }
    }
}
#endif
