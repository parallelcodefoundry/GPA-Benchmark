#include "hip/hip_runtime.h"
/***********************************************
	streamcluster_cuda.cu
	: parallelized code of streamcluster

	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by

	Shawn Sang-Ha Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science

***********************************************/
#include "streamcluster_header.cu"

using namespace std;

// AUTO-ERROR CHECK FOR ALL CUDA FUNCTIONS
#define CUDA_SAFE_CALL( call) do {										\
   hipError_t err = call;												\
   if( hipSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, hipGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)

// >>> START EDITABLE REGION ID=0
#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
// <<< END EDITABLE REGION ID=0
#define CUDATIME

// host memory
float *work_mem_h;
float *coord_h;

// device memory
float *work_mem_d;
float *coord_d;
int   *center_table_d;
bool  *switch_membership_d;
Point *p;

static int iter = 0;		// counter for total# of iteration

// >>> START EDITABLE REGION ID=1
//=======================================
// Euclidean Distance
//=======================================
__device__ float
d_dist(int p1, int p2, int num, int dim, float *coord_d)
{
	float retval = 0.0;
	for(int i = 0; i < dim; i++){
		float tmp = coord_d[(i*num)+p1] - coord_d[(i*num)+p2];
		retval += tmp * tmp;
	}
	return retval;
}
// <<< END EDITABLE REGION ID=1

// >>> START EDITABLE REGION ID=2
//=======================================
// Kernel - Compute Cost
//=======================================
__global__ void
kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
					float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d, int start)
{
	// block ID and global thread ID
	const int bid  = blockIdx.x + gridDim.x * blockIdx.y;
	const int tid = blockDim.x * bid + threadIdx.x + start;

	if(tid < num)
	{
		float *lower = &work_mem_d[tid*stride];

		// cost between this point and point[x]: euclidean distance multiplied by weight
		float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;

		// if computed cost is less then original (it saves), mark it as to reassign
		if ( x_cost < p[tid].cost )
		{
			switch_membership_d[tid] = 1;
			lower[K] += x_cost - p[tid].cost;
		}
		// if computed cost is larger, save the difference
		else
		{
			lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
		}
	}
}
// <<< END EDITABLE REGION ID=2

//=======================================
// Allocate Device Memory
//=======================================
void allocDevMem(int num, int dim)
{
	CUDA_SAFE_CALL( hipMalloc((void**) &center_table_d,	  num * sizeof(int))   );
	CUDA_SAFE_CALL( hipMalloc((void**) &switch_membership_d, num * sizeof(bool))  );
	CUDA_SAFE_CALL( hipMalloc((void**) &p,					  num * sizeof(Point)) );
	CUDA_SAFE_CALL( hipMalloc((void**) &coord_d,		num * dim * sizeof(float)) );
}

//=======================================
// Allocate Host Memory
//=======================================
void allocHostMem(int num, int dim)
{
	coord_h	= (float*) malloc( num * dim * sizeof(float) );
}

//=======================================
// Free Device Memory
//=======================================
void freeDevMem()
{
	CUDA_SAFE_CALL( hipFree(center_table_d)	  );
	CUDA_SAFE_CALL( hipFree(switch_membership_d) );
	CUDA_SAFE_CALL( hipFree(p)					  );
	CUDA_SAFE_CALL( hipFree(coord_d)			  );
}

//=======================================
// Free Host Memory
//=======================================
void freeHostMem()
{
	free(coord_h);
}

//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================
float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged,
							double *serial_t, double *cpu_to_gpu_t, double *gpu_to_cpu_t, double *alloc_t, double *kernel_t, double *free_t)
{
#ifdef CUDATIME
	float tmp_t;
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	hipEventRecord(start, 0);
#endif

	hipError_t error;

	int stride	= *numcenters + 1;			// size of each work_mem segment
	int K		= *numcenters ;				// number of centers
	int num		=  points->num;				// number of points
	int dim		=  points->dim;				// number of dimension
	int nThread =  num;						// number of threads == number of data points

	//=========================================
	// ALLOCATE HOST MEMORY + DATA PREPARATION
	//=========================================
	work_mem_h = (float*) malloc(stride * (nThread + 1) * sizeof(float) );
	// Only on the first iteration
	if(iter == 0)
	{
		allocHostMem(num, dim);
	}

	// build center-index table
	int count = 0;
	for( int i=0; i<num; i++)
	{
		if( is_center[i] )
		{
			center_table[i] = count++;
		}
	}

	// Extract 'coord'
	// Only if first iteration OR coord has changed
	if(isCoordChanged || iter == 0)
	{
		for(int i=0; i<dim; i++)
		{
			for(int j=0; j<num; j++)
			{
				coord_h[ (num*i)+j ] = points->p[j].coord[i];
			}
		}
	}

#ifdef CUDATIME
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&tmp_t, start, stop);
	*serial_t += (double) tmp_t;

	hipEventRecord(start,0);
#endif

	//=======================================
	// ALLOCATE GPU MEMORY
	//=======================================
	CUDA_SAFE_CALL( hipMalloc((void**) &work_mem_d,  stride * (nThread + 1) * sizeof(float)) );
	// Only on the first iteration
	if( iter == 0 )
	{
		allocDevMem(num, dim);
	}

#ifdef CUDATIME
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&tmp_t, start, stop);
	*alloc_t += (double) tmp_t;

	hipEventRecord(start,0);
#endif

	//=======================================
	// CPU-TO-GPU MEMORY COPY
	//=======================================
	// Only if first iteration OR coord has changed
	if(isCoordChanged || iter == 0)
	{
		CUDA_SAFE_CALL( hipMemcpy(coord_d,  coord_h,	 num * dim * sizeof(float), hipMemcpyHostToDevice) );
	}
	CUDA_SAFE_CALL( hipMemcpy(center_table_d,  center_table,  num * sizeof(int),   hipMemcpyHostToDevice) );
	CUDA_SAFE_CALL( hipMemcpy(p,  points->p,				   num * sizeof(Point), hipMemcpyHostToDevice) );

	CUDA_SAFE_CALL( hipMemset((void*) switch_membership_d, 0,			num * sizeof(bool))  );
	CUDA_SAFE_CALL( hipMemset((void*) work_mem_d,  		0, stride * (nThread + 1) * sizeof(float)) );

#ifdef CUDATIME
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&tmp_t, start, stop);
	*cpu_to_gpu_t += (double) tmp_t;

	hipEventRecord(start,0);
#endif

// >>> START EDITABLE REGION ID=3
	//=======================================
	// KERNEL: CALCULATE COST
	//=======================================
	// Determine the number of thread blocks in the x- and y-dimension
	int num_blocks 	 = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
	int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)  / (float) MAXBLOCKS);
	int num_blocks_x = (int) ((float) (num_blocks+num_blocks_y - 1) / (float) num_blocks_y);
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);

	int _first = (int)(num/128);

	// --- host-side compute of cost rows [0, _first) (rt3b) ---
	{
		int _lo = (0), _hi = (_first);
		if (_hi > num) _hi = num;
		if (_lo < 0) _lo = 0;
		// match the kernel's memset-0 device buffers: zero the host rows first
		for (int t = _lo; t < _hi; t++) {
			float *_lr = &work_mem_h[t*stride];
			for (int c = 0; c < stride; c++) _lr[c] = 0.0f;
			switch_membership[t] = 0;
		}
		for (int t = _lo; t < _hi; t++) {
			float _xc = 0.0f;
			for (int d = 0; d < dim; d++) {
				float _tmp = coord_h[(d*num)+t] - coord_h[(d*num)+x];
				_xc += _tmp * _tmp;
			}
			_xc *= points->p[t].weight;
			float *_lower = &work_mem_h[t*stride];
			if (_xc < points->p[t].cost) {
				switch_membership[t] = 1;
				_lower[K] += _xc - points->p[t].cost;
			} else {
				_lower[center_table[points->p[t].assign]] += points->p[t].cost - _xc;
			}
		}
		// push the host-computed rows to the device buffers the D2H copy will read back
		hipMemcpy(work_mem_d + (size_t)_lo*stride, work_mem_h + (size_t)_lo*stride,
				  (size_t)(_hi-_lo)*stride*sizeof(float), hipMemcpyHostToDevice);
		hipMemcpy(switch_membership_d + _lo, switch_membership + _lo,
				  (size_t)(_hi-_lo)*sizeof(bool), hipMemcpyHostToDevice);
	}
	// rt3b partial offload over-slack: [0,_first) on CPU, kernel does [_first,num)
	int _rem = num - _first; if (_rem < 0) _rem = 0;
	int _nb = (int)((float)(_rem + THREADS_PER_BLOCK - 1)/(float)THREADS_PER_BLOCK);
	int _nby = (int)((float)(_nb + MAXBLOCKS - 1)/(float)MAXBLOCKS);
	int _nbx = (int)((float)(_nb + _nby - 1)/(float)_nby);
	dim3 grid2(_nbx, _nby, 1);
	kernel_compute_cost<<<grid2, THREADS_PER_BLOCK>>>(num,dim,x,p,K,stride,coord_d,work_mem_d,center_table_d,switch_membership_d,_first);
	hipDeviceSynchronize();
// <<< END EDITABLE REGION ID=3

	// error check
	error = hipGetLastError();
	if (error != hipSuccess)
	{
		printf("kernel error: %s\n", hipGetErrorString(error));
		exit(EXIT_FAILURE);
	}

#ifdef CUDATIME
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&tmp_t, start, stop);
	*kernel_t += (double) tmp_t;

	hipEventRecord(start,0);
#endif

	//=======================================
	// GPU-TO-CPU MEMORY COPY
	//=======================================
	CUDA_SAFE_CALL( hipMemcpy(work_mem_h, 		  work_mem_d, 	stride * (nThread + 1) * sizeof(float), hipMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( hipMemcpy(switch_membership, switch_membership_d,	 num * sizeof(bool),  hipMemcpyDeviceToHost) );

#ifdef CUDATIME
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&tmp_t, start, stop);
	*gpu_to_cpu_t += (double) tmp_t;

	hipEventRecord(start,0);
#endif

	//=======================================
	// CPU (SERIAL) WORK
	//=======================================
	int number_of_centers_to_close = 0;
	float gl_cost_of_opening_x = z;
	float *gl_lower = &work_mem_h[stride * nThread];
	// compute the number of centers to close if we are to open i
	for(int i=0; i < num; i++)
	{
		if( is_center[i] )
		{
			float low = z;
		    for( int j = 0; j < num; j++ )
			{
				low += work_mem_h[ j*stride + center_table[i] ];
			}

		    gl_lower[center_table[i]] = low;

		    if ( low > 0 )
			{
				++number_of_centers_to_close;
				work_mem_h[i*stride+K] -= low;
		    }
		}
		gl_cost_of_opening_x += work_mem_h[i*stride+K];
	}

	//if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
	if ( gl_cost_of_opening_x < 0 )
	{
		for(int i = 0; i < num; i++)
		{
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
			if ( switch_membership[i] || close_center )
			{
				points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
				points->p[i].assign = x;
			}
		}

		for(int i = 0; i < num; i++)
		{
			if( is_center[i] && gl_lower[center_table[i]] > 0 )
			{
				is_center[i] = false;
			}
		}

		if( x >= 0 && x < num)
		{
			is_center[x] = true;
		}
		*numcenters = *numcenters + 1 - number_of_centers_to_close;
	}
	else
	{
		gl_cost_of_opening_x = 0;
	}

	//=======================================
	// DEALLOCATE HOST MEMORY
	//=======================================
	free(work_mem_h);


#ifdef CUDATIME
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&tmp_t, start, stop);
	*serial_t += (double) tmp_t;

	hipEventRecord(start,0);
#endif

	//=======================================
	// DEALLOCATE GPU MEMORY
	//=======================================
	CUDA_SAFE_CALL( hipFree(work_mem_d) );


#ifdef CUDATIME
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&tmp_t, start, stop);
	*free_t += (double) tmp_t;
#endif
	iter++;
	return -gl_cost_of_opening_x;
}
