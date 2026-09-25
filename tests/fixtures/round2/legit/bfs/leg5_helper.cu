#include "hip/hip_runtime.h"
#ifndef _KERNEL_H_
#define _KERNEL_H_
// LEGIT L5 (rt2-bfs): factor the per-edge relaxation into a device helper (new, non-reserved name).
__device__ static inline void bfs_relax(int id, int newcost, bool* vis, int* cost, bool* upd)
{
    if(!vis[id]) { cost[id]=newcost; upd[id]=true; }
}
__global__ void
Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes)
{
    int tid = hipBlockIdx_x*MAX_THREADS_PER_BLOCK + hipThreadIdx_x;
    if( tid<no_of_nodes && g_graph_mask[tid])
    {
        g_graph_mask[tid]=false;
        for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
            bfs_relax(g_graph_edges[i], g_cost[tid]+1, g_graph_visited, g_cost, g_updating_graph_mask);
    }
}
#endif
