#include "hip/hip_runtime.h"
__device__ __forceinline__ bool in_range(int k0, int k1, int q){ return (k0 <= q) && (k1 > q); }
__global__ void
findRangeK( long height, knode *knodesD, long knodes_elem,
            long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D,
            int *startD, int *endD, int *RecstartD, int *ReclenD)
{
    int thid = hipThreadIdx_x; int bid = hipBlockIdx_x; int i;
    for(i = 0; i < height; i++){
        if(in_range(knodesD[currKnodeD[bid]].keys[thid], knodesD[currKnodeD[bid]].keys[thid+1], startD[bid])){
            if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){ offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid]; }
        }
        if(in_range(knodesD[lastKnodeD[bid]].keys[thid], knodesD[lastKnodeD[bid]].keys[thid+1], endD[bid])){
            if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){ offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid]; }
        }
        __syncthreads();
        if(thid==0){ currKnodeD[bid] = offsetD[bid]; lastKnodeD[bid] = offset_2D[bid]; }
        __syncthreads();
    }
    if(knodesD[currKnodeD[bid]].keys[thid] == startD[bid]){ RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid]; }
    __syncthreads();
    if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){ ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1; }
}
