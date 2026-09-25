#include "hip/hip_runtime.h"
__global__ void
findRangeK( long height, knode *knodesD, long knodes_elem,
            long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D,
            int *startD, int *endD, int *RecstartD, int *ReclenD)
{
    int thid = hipThreadIdx_x; int bid = hipBlockIdx_x; int i;
    long cur = currKnodeD[bid];
    long lst = lastKnodeD[bid];
    for(i = 0; i < height; i++){
        if((knodesD[cur].keys[thid] <= startD[bid]) && (knodesD[cur].keys[thid+1] > startD[bid])){
            if(knodesD[cur].indices[thid] < knodes_elem){ offsetD[bid] = knodesD[cur].indices[thid]; }
        }
        if((knodesD[lst].keys[thid] <= endD[bid]) && (knodesD[lst].keys[thid+1] > endD[bid])){
            if(knodesD[lst].indices[thid] < knodes_elem){ offset_2D[bid] = knodesD[lst].indices[thid]; }
        }
        __syncthreads();
        if(thid==0){ currKnodeD[bid] = offsetD[bid]; lastKnodeD[bid] = offset_2D[bid]; }
        __syncthreads();
        cur = __builtin_amdgcn_readfirstlane((int)offsetD[bid]);
        lst = __builtin_amdgcn_readfirstlane((int)offset_2D[bid]);
    }
    if(knodesD[cur].keys[thid] == startD[bid]){ RecstartD[bid] = knodesD[cur].indices[thid]; }
    __syncthreads();
    if(knodesD[lst].keys[thid] == endD[bid]){ ReclenD[bid] = knodesD[lst].indices[thid] - RecstartD[bid]+1; }
}
