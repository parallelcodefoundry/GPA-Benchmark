//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//      plasmaKernel_gpu_2
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200

__global__ void kernel_gpu_cuda(par_str d_par_gpu,
                                dim_str d_dim_gpu,
                                box_str* d_box_gpu,
                                FOUR_VECTOR* d_rv_gpu,
                                fp* d_qv_gpu,
                                FOUR_VECTOR* d_fv_gpu)
{

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
        //      THREAD PARAMETERS
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

        int bx = blockIdx.x;                                                              // get current horizontal block index (0-n)
        int tx = threadIdx.x;                                                             // get current horizontal thread index (0-n)
        // int ax = bx*NUMBER_THREADS+tx;
        // int wbx = bx;
        int wtx = tx;

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
        //      DO FOR THE NUMBER OF BOXES
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

        if(bx<d_dim_gpu.number_boxes){
        // while(wbx<box_indexes_counter){

                //------------------------------------------------------------------------------------------------------------------------------------------------------160
                //      Extract input parameters
                //------------------------------------------------------------------------------------------------------------------------------------------------------160

                // parameters
                fp a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

                // home box
                int first_i;
                FOUR_VECTOR* rA;
                FOUR_VECTOR* fA;
                __shared__ FOUR_VECTOR rA_shared[100];

                // nei box
                int pointer;
                int k = 0;
                int first_j;
                FOUR_VECTOR* rB;
                fp* qB;
                int j = 0;
                __shared__ FOUR_VECTOR rB_shared[100];
                __shared__ double qB_shared[100];

                // common
                fp r2;
                fp u2;
                fp vij;
                fp fs;
                fp fxij;
                fp fyij;
                fp fzij;
                THREE_VECTOR d;

                //------------------------------------------------------------------------------------------------------------------------------------------------------160
                //      Home box
                //------------------------------------------------------------------------------------------------------------------------------------------------------160

                //----------------------------------------------------------------------------------------------------------------------------------140
                //      Setup parameters
                //----------------------------------------------------------------------------------------------------------------------------------140

                // home box - box parameters
                first_i = d_box_gpu[bx].offset;

                // home box - distance, force, charge and type parameters
                rA = &d_rv_gpu[first_i];
                fA = &d_fv_gpu[first_i];

                //----------------------------------------------------------------------------------------------------------------------------------140
                //      Copy to shared memory
                //----------------------------------------------------------------------------------------------------------------------------------140

                // home box - shared memory
                // Optimized: since NUMBER_THREADS >= NUMBER_PAR_PER_BOX, use simple if statement
                if(tx < NUMBER_PAR_PER_BOX){
                        rA_shared[tx] = rA[tx];
                }
                __syncthreads();

                //------------------------------------------------------------------------------------------------------------------------------------------------------160
                //      nei box loop
                //------------------------------------------------------------------------------------------------------------------------------------------------------160

                // loop over neiing boxes of home box
                for (k=0; k<(1+d_box_gpu[bx].nn); k++){

                        //----------------------------------------50
                        //      nei box - get pointer to the right box
                        //----------------------------------------50

                        if(k==0){
                                pointer = bx;                                                   // set first box to be processed to home box
                        }
                        else{
                                pointer = d_box_gpu[bx].nei[k-1].number;                            // remaining boxes are nei boxes
                        }

                        //----------------------------------------------------------------------------------------------------------------------------------140
                        //      Setup parameters
                        //----------------------------------------------------------------------------------------------------------------------------------140

                        // nei box - box parameters
                        first_j = d_box_gpu[pointer].offset;

                        // nei box - distance, (force), charge and (type) parameters
                        rB = &d_rv_gpu[first_j];
                        qB = &d_qv_gpu[first_j];

                        //----------------------------------------------------------------------------------------------------------------------------------140
                        //      Setup parameters
                        //----------------------------------------------------------------------------------------------------------------------------------140

                        // nei box - shared memory
                        // Optimized: since NUMBER_THREADS >= NUMBER_PAR_PER_BOX, use simple if statement
                        if(tx < NUMBER_PAR_PER_BOX){
                                rB_shared[tx] = rB[tx];
                                qB_shared[tx] = qB[tx];
                        }
                        __syncthreads();

                        //----------------------------------------------------------------------------------------------------------------------------------140
                        //      Calculation
                        //----------------------------------------------------------------------------------------------------------------------------------140

                        // loop for the number of particles in the home box
                        // for (int i=0; i<nTotal_i; i++){
                        while(wtx<NUMBER_PAR_PER_BOX){

                                // Optimized: Load rA into registers before the inner loop
                                fp rA_v = rA_shared[wtx].v;
                                fp rA_x = rA_shared[wtx].x;
                                fp rA_y = rA_shared[wtx].y;
                                fp rA_z = rA_shared[wtx].z;

                                // loop for the number of particles in the current nei box
                                for (j=0; j<NUMBER_PAR_PER_BOX; j++){
                                        // Optimized: Use registers for rB and qB to reduce shared memory accesses
                                        fp rB_v = rB_shared[j].v;
                                        fp rB_x = rB_shared[j].x;
                                        fp rB_y = rB_shared[j].y;
                                        fp rB_z = rB_shared[j].z;
                                        fp qB_val = qB_shared[j];

                                        r2 = rA_v + rB_v - (rA_x*rB_x + rA_y*rB_y + rA_z*rB_z);
                                        u2 = a2*r2;
                                        vij= exp(-u2);
                                        fs = 2*vij;

                                        fxij=fs*(rA_x - rB_x);
                                        fyij=fs*(rA_y - rB_y);
                                        fzij=fs*(rA_z - rB_z);

                                        fA[wtx].v +=  qB_val*vij;
                                        fA[wtx].x +=  qB_val*fxij;
                                        fA[wtx].y +=  qB_val*fyij;
                                        fA[wtx].z +=  qB_val*fzij;

                                }

                                // increment work thread index
                                wtx = wtx + NUMBER_THREADS;

                        }

                        // reset work index
                        wtx = tx;

                        // synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
                        __syncthreads();

                        //----------------------------------------------------------------------------------------------------------------------------------140
                        //      Calculation END
                        //----------------------------------------------------------------------------------------------------------------------------------140

                }

                // // increment work block index
                // wbx = wbx + NUMBER_BLOCKS;

                // // synchronize - because next iteration will overwrite current shared memory
                // __syncthreads();

                //------------------------------------------------------------------------------------------------------------------------------------------------------160
                //      nei box loop END
                //------------------------------------------------------------------------------------------------------------------------------------------------------160

        }

}

