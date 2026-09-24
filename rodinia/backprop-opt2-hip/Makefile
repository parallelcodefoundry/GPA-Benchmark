# HIP (Frontier/MI250X) build of GPA backprop: hipify-perl of the GPA CUDA source.
# Delta vs CUDA Makefile: nvcc -arch=sm_$(SM_VERSION) -O3 -lineinfo -> hipcc --offload-arch=gfx90a -O3 -gline-tables-only;
# link via hipcc (was gcc ... -L$(CUDA_LIB_DIR) -lcuda -lcudart); make.config/CUDA_DIR/NVCC_CCBIN dropped.
CC = gcc
CC_FLAGS = -g -O3 -Wno-implicit-int -Wno-implicit-function-declaration -Wno-incompatible-pointer-types

# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a
HIPCC_FLAGS = --offload-arch=$(OFFLOAD_ARCH)

ifeq ($(dbg),1)
	HIPCC_FLAGS += -g -O0
else
	HIPCC_FLAGS += -O3 -gline-tables-only
endif

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c

backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop_cuda_kernel.cu backprop.h
	$(HIPCC) $(HIPCC_FLAGS) -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ backprop *.qdrep *.sqlite
