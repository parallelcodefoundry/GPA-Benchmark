#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)



void
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file){

	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];

	if( (fp = fopen(file, "w" )) == 0 )
          printf( "The file was not opened\n" );


	for (i=0; i < grid_rows; i++)
	 for (j=0; j < grid_cols; j++)
	 {

		 sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
		 fputs(str,fp);
		 index++;
	 }

      fclose(fp);
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file){

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            printf( "The file was not opened\n" );


	for (i=0; i <= grid_rows-1; i++)
	 for (j=0; j <= grid_cols-1; j++)
	 {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fatal("invalid file format");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);

}

// >>> START EDITABLE REGION ID=0
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))
#define WARP_SIZE 32
// <<< END EDITABLE REGION ID=0

// >>> START EDITABLE REGION ID=1
__global__ void calculate_temp(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
							   int border_cols,  // border offset
							   int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step,
                               float time_elapsed){

        __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];

	// Precompute constants to reduce register pressure
        float step_div_Cap = step / Cap;
        float Rx_1 = 1.0f / Rx;
        float Ry_1 = 1.0f / Ry;
        float Rz_1 = 1.0f / Rz;
        float amb_temp = 80.0f;

	int bx = blockIdx.x;
        int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

        // Calculate block's global position
        int small_block_rows = BLOCK_SIZE - iteration * 2;
        int small_block_cols = BLOCK_SIZE - iteration * 2;
        
        int blkY = small_block_rows * by - border_rows;
        int blkX = small_block_cols * bx - border_cols;
        
        // Thread's global coordinates
        int yidx = blkY + ty;
        int xidx = blkX + tx;
        
        // Load data with bounds checking
        int index = yidx * grid_cols + xidx;
        
        // Use 2D bounds checking for better performance
        bool valid_thread = (yidx >= 0 && yidx < grid_rows && xidx >= 0 && xidx < grid_cols);
        
        if (valid_thread) {
            temp_on_cuda[ty][tx] = temp_src[index];
            power_on_cuda[ty][tx] = power[index];
        }
        __syncthreads();

        // Precompute valid range for this block
        int validYmin = (blkY < 0) ? -blkY : 0;
        int validYmax = (blkY + BLOCK_SIZE - 1 > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkY + BLOCK_SIZE - 1 - grid_rows + 1) : BLOCK_SIZE - 1;
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkX + BLOCK_SIZE - 1 > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkX + BLOCK_SIZE - 1 - grid_cols + 1) : BLOCK_SIZE - 1;

        // Compute neighbors with boundary clamping
        int N = ty - 1;
        int S = ty + 1;
        int W = tx - 1;
        int E = tx + 1;
        
        N = (N < validYmin) ? validYmin : (N > validYmax) ? validYmax : N;
        S = (S < validYmin) ? validYmin : (S > validYmax) ? validYmax : S;
        W = (W < validXmin) ? validXmin : (W > validXmax) ? validXmax : W;
        E = (E < validXmin) ? validXmin : (E > validXmax) ? validXmax : E;

        // Compute inner region for this iteration
        int inner_start = iteration - 1;
        int inner_end = BLOCK_SIZE - iteration;
        
        bool computed_last = false;
        
        // Unroll the loop for small iteration counts (typically 1-2)
        #pragma unroll
        for (int i = 0; i < iteration; i++) {
            bool computed = false;
            
            // Check if this thread is in the computation region for this iteration
            bool in_inner = (tx >= inner_start && tx < inner_end && 
                            ty >= inner_start && ty < inner_end);
            
            if (in_inner && valid_thread) {
                // Compute temperature update
                float temp_center = temp_on_cuda[ty][tx];
                float temp_north = temp_on_cuda[N][tx];
                float temp_south = temp_on_cuda[S][tx];
                float temp_west = temp_on_cuda[ty][W];
                float temp_east = temp_on_cuda[ty][E];
                float power_val = power_on_cuda[ty][tx];
                
                temp_t[ty][tx] = temp_center + step_div_Cap * (
                    power_val +
                    (temp_south + temp_north - 2.0f * temp_center) * Ry_1 +
                    (temp_east + temp_west - 2.0f * temp_center) * Rx_1 +
                    (amb_temp - temp_center) * Rz_1
                );
                computed = true;
            }
            __syncthreads();
            
            // Update shared memory for next iteration (except last iteration)
            if (i < iteration - 1) {
                if (computed) {
                    temp_on_cuda[ty][tx] = temp_t[ty][tx];
                }
                __syncthreads();
            }
            
            // Track if we computed in the last iteration
            if (i == iteration - 1) {
                computed_last = computed;
            }
        }

        // Write result if this thread computed a value in the last iteration
        if (computed_last && valid_thread) {
            temp_dst[index] = temp_t[ty][tx];
        }
}
// <<< END EDITABLE REGION ID=1

// >>> START EDITABLE REGION ID=2
/*
   compute N time steps
*/

int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows)
{
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(blockCols, blockRows);

	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	float t;
        float time_elapsed;
	time_elapsed=0.001;

        int src = 1, dst = 0;

	for (t = 0; t < total_iterations; t+=num_iterations) {
            int temp = src;
            src = dst;
            dst = temp;
            calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
		col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);
	}
        return dst;
}
// <<< END EDITABLE REGION ID=2

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv)
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp,*FilesavingPower,*MatrixOut;
    char *tfile, *pfile, *ofile;

    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations

	if (argc != 7)
		usage(argc, argv);
	if((grid_rows = atoi(argv[1]))<=0||
	   (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0)
		usage(argc, argv);

	tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];

    size=grid_rows*grid_cols;

    /* --------------- pyramid parameters --------------- */
    # define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp || !MatrixOut)
        fatal("unable to allocate memory");

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
	pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

    float *MatrixTemp[2], *MatrixPower;
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice);
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
	 total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);
	printf("Ending simulation\n");
    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);

    writeoutput(MatrixOut,grid_rows, grid_cols, ofile);

    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
    free(MatrixOut);
}
