#include "hip/hip_runtime.h"

__global__ void __amd_rocclr_findRangeK(long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
{

	// private thread IDs
	int thid = hipThreadIdx_x;
	int bid = hipBlockIdx_x;

	// ???
	int i;
	for(i = 0; i < height; i++){

		if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){
				offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
			}
		}
		if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){
				offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
			}
		}
		__syncthreads();

		// set for next tree level
		if(thid==0){
			currKnodeD[bid] = offsetD[bid];
			lastKnodeD[bid] = offset_2D[bid];
		}
		__syncthreads();
	}

	// Find the index of the starting record
	if(knodesD[currKnodeD[bid]].keys[thid] == startD[bid]){
		RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
	}
	__syncthreads();

	// Find the index of the ending record
	if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){
		ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1;
	}

}

__global__ void findRangeK(long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
{
}

static inline void g1_launch(dim3 grid, dim3 block, long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
{
	hipLaunchKernelGGL(__amd_rocclr_findRangeK, grid, block, 0, 0, height, knodesD, knodes_elem, currKnodeD, offsetD, lastKnodeD, offset_2D, startD, endD, RecstartD, ReclenD);
	hipLaunchKernelGGL(findRangeK, dim3(1), dim3(64), 0, 0, height, knodesD, knodes_elem, currKnodeD, offsetD, lastKnodeD, offset_2D, startD, endD, RecstartD, ReclenD);
}
#undef hipLaunchKernelGGL
#define hipLaunchKernelGGL(K, G, B, S, ST, ...) g1_launch(G, B, __VA_ARGS__)
