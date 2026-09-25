#include "hip/hip_runtime.h"
__global__ void __launch_bounds__(1024, 2)
findRangeK( long height, knode *knodesD, long knodes_elem,
            long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D,
            int *startD, int *endD, int *RecstartD, int *ReclenD)
{
    int thid = hipThreadIdx_x; int bid = hipBlockIdx_x; int i;
    for(i = 0; i < height; i++){
        if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
            if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){ offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid]; }
        }
        if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
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
