# HIP (Frontier/MI250X) build of GPA b+tree. Source: AMD HIP-Examples rodinia_3.0/hip/b+tree (native port)
# + APPEB patch (GPA EDITABLE REGION markers in kernel/kernel_gpu_cuda_wrapper_2.cu, PROFILING blocks removed, CRLF->LF).
# Delta vs CUDA Makefile: main.c (gcc) -> main.cu (hipcc); nvcc -O3 -lineinfo -arch=sm_$(SM_VERSION) -> hipcc --offload-arch=gfx90a -O3 -gline-tables-only;
# link via hipcc (was gcc ... -lcuda -lcudart).
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a
HIPFLAGS = --offload-arch=$(OFFLOAD_ARCH) -O3 -gline-tables-only
C_C = gcc

b+tree.out:	./main.o ./kernel/kernel_gpu_cuda_wrapper.o ./kernel/kernel_gpu_cuda_wrapper_2.o \
		./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) ./main.o ./kernel/kernel_gpu_cuda_wrapper.o ./kernel/kernel_gpu_cuda_wrapper_2.o \
		./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o -lm -o b+tree.out

main.o:	./common.h ./main.h ./main.cu
	$(HIPCC) $(HIPFLAGS) $(KERNEL_DIM) ./main.cu -c -o ./main.o

./kernel/kernel_gpu_cuda_wrapper.o:	./common.h ./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper.cu ./kernel/kernel_gpu_cuda.cu
	$(HIPCC) $(HIPFLAGS) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper.cu -c -o ./kernel/kernel_gpu_cuda_wrapper.o

./kernel/kernel_gpu_cuda_wrapper_2.o:	./common.h ./kernel/kernel_gpu_cuda_wrapper_2.h ./kernel/kernel_gpu_cuda_wrapper_2.cu ./kernel/kernel_gpu_cuda_2.cu
	$(HIPCC) $(HIPFLAGS) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper_2.cu -c -o ./kernel/kernel_gpu_cuda_wrapper_2.o

./util/timer/timer.o:	./common.h ./util/timer/timer.h ./util/timer/timer.c
	$(C_C)	./util/timer/timer.c -c -o ./util/timer/timer.o -O3

./util/num/num.o:	./common.h ./util/num/num.h ./util/num/num.c
	$(C_C)	./util/num/num.c -c -o ./util/num/num.o -O3

./util/cuda/cuda.o:	./common.h ./util/cuda/cuda.h ./util/cuda/cuda.cu
	$(HIPCC) $(HIPFLAGS) ./util/cuda/cuda.cu -c -o ./util/cuda/cuda.o

clean:
	rm -f *.o *.out ./kernel/*.o ./util/timer/*.o ./util/num/*.o ./util/cuda/*.o output.txt *.qdrep *.sqlite
