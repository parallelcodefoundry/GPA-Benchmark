# HIP (Frontier/MI250X) build of GPA heartwall: hipify-perl of the GPA CUDA source.
# Delta vs CUDA Makefile: nvcc -ccbin ... -lineinfo -O3 -> hipcc --offload-arch=gfx90a -gline-tables-only -O3;
# link via hipcc. (heartwall-opt's CUDA Makefile used `ifdef OUTPUT`, i.e. no result.txt by default; the
# -hip copy uses the baseline's `ifndef QUIET` so result.txt is written for correctness checks.)
ifndef QUIET
override OUTPUT = -DOUTPUT
endif

# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a

heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) main.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall

main.o: main.cu kernel.cu define.c
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) -gline-tables-only $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

clean:
	rm -f *.o AVI/*.o heartwall *.qdrep *.sqlite
