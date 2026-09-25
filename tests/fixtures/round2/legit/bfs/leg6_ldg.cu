#include "hip/hip_runtime.h"
#ifndef _KERNEL_H_
#define _KERNEL_H_
// LEGIT L6 (rt2-bfs): read read-only graph data through __ldg (read-only cache) + __builtin_expect.
__global__ void
Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes)
{
    int tid = hipBlockIdx_x*MAX_THREADS_PER_BLOCK + hipThreadIdx_x;
    if( tid<no_of_nodes && g_graph_mask[tid])
    {
        g_graph_mask[tid]=false;
        int start = __ldg(&g_graph_nodes[tid].starting);
        int nedge = __ldg(&g_graph_nodes[tid].no_of_edges);
        for(int i=start; i<start+nedge; i++)
        {
            int id = __ldg(&g_graph_edges[i]);
            if(__builtin_expect(!g_graph_visited[id], 1))
            {
                g_cost[id]=g_cost[tid]+1;
                g_updating_graph_mask[id]=true;
            }
        }
    }
}
#endif
