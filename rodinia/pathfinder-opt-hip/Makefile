# HIP port of GPA rodinia/pathfinder-opt. NOTE: -DBENCH_PRINT added (CUDA opt Makefile omits it) so the output can be validated.
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
