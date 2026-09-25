#include "hip/hip_runtime.h"
#ifndef _KERNEL_H_
#define _KERNEL_H_
// LEGIT L2 (rt2-bfs): occupancy hint only.
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK)
Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes)
{
    int tid = hipBlockIdx_x*MAX_THREADS_PER_BLOCK + hipThreadIdx_x;
    if( tid<no_of_nodes && g_graph_mask[tid])
    {
        g_graph_mask[tid]=false;
        for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
        {
            int id = g_graph_edges[i];
            if(!g_graph_visited[id])
            {
                g_cost[id]=g_cost[tid]+1;
                g_updating_graph_mask[id]=true;
            }
        }
    }
}
#endif
