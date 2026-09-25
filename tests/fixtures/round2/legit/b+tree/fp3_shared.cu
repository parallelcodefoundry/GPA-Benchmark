#include "hip/hip_runtime.h"
__global__ void
findRangeK( long height, knode *knodesD, long knodes_elem,
            long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D,
            int *startD, int *endD, int *RecstartD, int *ReclenD)
{
    __shared__ int s_start[1];
    __shared__ int s_end[1];
    int thid = hipThreadIdx_x; int bid = hipBlockIdx_x; int i;
    if(thid==0){ s_start[0]=startD[bid]; s_end[0]=endD[bid]; }
    __syncthreads();
    int start = s_start[0]; int end = s_end[0];
    for(i = 0; i < height; i++){
        if((knodesD[currKnodeD[bid]].keys[thid] <= start) && (knodesD[currKnodeD[bid]].keys[thid+1] > start)){
            if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){ offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid]; }
        }
        if((knodesD[lastKnodeD[bid]].keys[thid] <= end) && (knodesD[lastKnodeD[bid]].keys[thid+1] > end)){
            if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){ offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid]; }
        }
        __syncthreads();
        if(thid==0){ currKnodeD[bid] = offsetD[bid]; lastKnodeD[bid] = offset_2D[bid]; }
        __syncthreads();
    }
    if(knodesD[currKnodeD[bid]].keys[thid] == start){ RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid]; }
    __syncthreads();
    if(knodesD[lastKnodeD[bid]].keys[thid] == end){ ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1; }
}
