# HIP (Frontier/MI250X) build of GPA b+tree: hipify-perl of the GPA CUDA source.
# Delta vs CUDA Makefile: nvcc -ccbin ... -O3 -lineinfo -arch=sm_$(SM_VERSION) -> hipcc --offload-arch=gfx90a -O3 -gline-tables-only;
# link via hipcc (was gcc ... -lstdc++ -L$(CUDA_LIB_DIR) -lcuda -lcudart). Hand fix: removed the
# '#include "hip/hip_runtime.h"' hipify-perl added to gcc-compiled main.c (it calls hipSetDevice via implicit decl,
# exactly like the CUDA original calls cudaSetDevice).
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a
CUDA_FLAG = --offload-arch=$(OFFLOAD_ARCH) -gline-tables-only
C_C = gcc -Wno-implicit-int -Wno-implicit-function-declaration -Wno-incompatible-pointer-types
CUD_C = $(HIPCC)

b+tree.out:	./main.o ./kernel/kernel_gpu_cuda_wrapper.o ./kernel/kernel_gpu_cuda_wrapper_2.o \
		./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) ./main.o ./kernel/kernel_gpu_cuda_wrapper.o ./kernel/kernel_gpu_cuda_wrapper_2.o \
		./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o -lm -o b+tree.out

main.o:	./common.h ./main.h ./main.c
	$(C_C)	$(KERNEL_DIM) ./main.c -c -o ./main.o -O3

./kernel/kernel_gpu_cuda_wrapper.o:	./common.h ./kernel/kernel_gpu_cuda_wrapper.h ./kernel/kernel_gpu_cuda_wrapper.cu ./kernel/kernel_gpu_cuda.cu
	$(CUD_C) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper.cu -c -o ./kernel/kernel_gpu_cuda_wrapper.o -O3 $(CUDA_FLAG)

./kernel/kernel_gpu_cuda_wrapper_2.o:	./common.h ./kernel/kernel_gpu_cuda_wrapper_2.h ./kernel/kernel_gpu_cuda_wrapper_2.cu ./kernel/kernel_gpu_cuda_2.cu
	$(CUD_C) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper_2.cu -c -o ./kernel/kernel_gpu_cuda_wrapper_2.o -O3 $(CUDA_FLAG)

./util/timer/timer.o:	./common.h ./util/timer/timer.h ./util/timer/timer.c
	$(C_C)	./util/timer/timer.c -c -o ./util/timer/timer.o -O3

./util/num/num.o:	./common.h ./util/num/num.h ./util/num/num.c
	$(C_C)	./util/num/num.c -c -o ./util/num/num.o -O3

./util/cuda/cuda.o:	./common.h ./util/cuda/cuda.h ./util/cuda/cuda.cu
	$(CUD_C)	./util/cuda/cuda.cu -c -o ./util/cuda/cuda.o -O3 $(CUDA_FLAG)

clean:
	rm -f *.o *.out ./kernel/*.o ./util/timer/*.o ./util/num/*.o ./util/cuda/*.o output.txt *.qdrep *.sqlite
