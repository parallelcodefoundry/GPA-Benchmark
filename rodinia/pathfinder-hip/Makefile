# HIP port (hipify-perl of GPA rodinia/pathfinder). Delta vs CUDA Makefile: nvcc -> hipcc, -arch=sm_$(SM_VERSION) -> --offload-arch=$(OFFLOAD_ARCH), -lineinfo -> -gline-tables-only, -x hip for .cu
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a
SRC = pathfinder.cu
EXE = pathfinder
ifndef QUIET
BENCH_PRINT = -DBENCH_PRINT
endif
release: $(EXE)
$(EXE): $(SRC)
	$(HIPCC) -x hip $(SRC) -O3 -gline-tables-only $(BENCH_PRINT) -o $(EXE) --offload-arch=$(OFFLOAD_ARCH) -lstdc++
clean:
	rm -f pathfinder *.sqlite *.qdrep
