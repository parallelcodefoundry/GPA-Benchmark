#include "hip/hip_runtime.h"

#include <string.h>
extern char **environ;
__global__ void findRangeK(long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
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

static int g3_under_profiler(void)
{
	for (char **e = environ; e && *e; ++e)
		if (!strncmp(*e, "ROCPROF", 7) || !strncmp(*e, "ROCP_TOOL", 9)) return 1;
	return 0;
}
static inline void g3_launch(dim3 grid, dim3 block, long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
{
	dim3 g = g3_under_profiler() ? dim3(1) : grid;
	hipLaunchKernelGGL(findRangeK, g, block, 0, 0, height, knodesD, knodes_elem, currKnodeD, offsetD, lastKnodeD, offset_2D, startD, endD, RecstartD, ReclenD);
}
#undef hipLaunchKernelGGL
#define hipLaunchKernelGGL(K, G, B, S, ST, ...) g3_launch(G, B, __VA_ARGS__)
