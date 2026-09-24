# HIP port (hipify-perl of GPA rodinia/streamcluster[-opt]). Delta: nvcc -> hipcc --offload-arch=$(OFFLOAD_ARCH), -lineinfo -> -gline-tables-only, -lcuda dropped, .cu compiled with -x hip.
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a
FLAGS = -O3 -gline-tables-only
ifeq ($(dbg),1)
	FLAGS = -g -O0
endif
TARGET_G = sc_gpu
$(TARGET_G): streamcluster_cuda_cpu.cpp streamcluster_cuda.cu streamcluster_header.cu
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) $(FLAGS) -x hip streamcluster_cuda_cpu.cpp streamcluster_cuda.cu -o $(TARGET_G) -lstdc++ -lm
clean:
	rm -f *.o *~ output.txt $(TARGET_G) *.linkinfo *.sqlite *.qdrep
