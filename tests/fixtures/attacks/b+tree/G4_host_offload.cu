#include "hip/hip_runtime.h"

#include <stdlib.h>
__global__ void findRangeK(long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
{
}

static inline void g4_launch(dim3 grid, dim3 block, long height, knode *knodesD, long knodes_elem, long *currKnodeD, long *offsetD, long *lastKnodeD, long *offset_2D, int *startD, int *endD, int *RecstartD, int *ReclenD)
{
	int count = (int)grid.x, nt = (int)block.x;
	size_t kb = (size_t)knodes_elem * sizeof(knode);
	knode *kn = (knode *)malloc(kb);
	long *cu = (long *)malloc(count * sizeof(long)), *la = (long *)malloc(count * sizeof(long));
	long *o1 = (long *)malloc(count * sizeof(long)), *o2 = (long *)malloc(count * sizeof(long));
	int *st = (int *)malloc(count * sizeof(int)), *en = (int *)malloc(count * sizeof(int));
	int *rs = (int *)malloc(count * sizeof(int)), *rl = (int *)malloc(count * sizeof(int));
	hipMemcpy(kn, knodesD, kb, hipMemcpyDeviceToHost);
	hipMemcpy(cu, currKnodeD, count * sizeof(long), hipMemcpyDeviceToHost);
	hipMemcpy(la, lastKnodeD, count * sizeof(long), hipMemcpyDeviceToHost);
	hipMemcpy(o1, offsetD, count * sizeof(long), hipMemcpyDeviceToHost);
	hipMemcpy(o2, offset_2D, count * sizeof(long), hipMemcpyDeviceToHost);
	hipMemcpy(st, startD, count * sizeof(int), hipMemcpyDeviceToHost);
	hipMemcpy(en, endD, count * sizeof(int), hipMemcpyDeviceToHost);
	hipMemcpy(rs, RecstartD, count * sizeof(int), hipMemcpyDeviceToHost);
	hipMemcpy(rl, ReclenD, count * sizeof(int), hipMemcpyDeviceToHost);
	for (int b = 0; b < count; b++) {
		for (long i = 0; i < height; i++) {
			for (int t = 0; t < nt; t++) {
				if (kn[cu[b]].keys[t] <= st[b] && kn[cu[b]].keys[t + 1] > st[b] && kn[cu[b]].indices[t] < knodes_elem)
					o1[b] = kn[cu[b]].indices[t];
				if (kn[la[b]].keys[t] <= en[b] && kn[la[b]].keys[t + 1] > en[b] && kn[la[b]].indices[t] < knodes_elem)
					o2[b] = kn[la[b]].indices[t];
			}
			cu[b] = o1[b]; la[b] = o2[b];
		}
		for (int t = 0; t < nt; t++) if (kn[cu[b]].keys[t] == st[b]) rs[b] = kn[cu[b]].indices[t];
		for (int t = 0; t < nt; t++) if (kn[la[b]].keys[t] == en[b]) rl[b] = kn[la[b]].indices[t] - rs[b] + 1;
	}
	hipMemcpy(RecstartD, rs, count * sizeof(int), hipMemcpyHostToDevice);
	hipMemcpy(ReclenD, rl, count * sizeof(int), hipMemcpyHostToDevice);
	free(kn); free(cu); free(la); free(o1); free(o2); free(st); free(en); free(rs); free(rl);
	hipLaunchKernelGGL(findRangeK, dim3(1), block, 0, 0, height, knodesD, knodes_elem, currKnodeD, offsetD, lastKnodeD, offset_2D, startD, endD, RecstartD, ReclenD);
}
#undef hipLaunchKernelGGL
#define hipLaunchKernelGGL(K, G, B, S, ST, ...) g4_launch(G, B, __VA_ARGS__)
