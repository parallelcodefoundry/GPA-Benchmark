# HIP port (hipify-perl of GPA rodinia/nw[-opt]). Delta: nvcc -> hipcc --offload-arch=$(OFFLOAD_ARCH), -lineinfo -> -gline-tables-only, -x hip. (-opt: -DTRACEBACK added so result.txt is written for validation.)
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a
SRC = needle.cu
EXE = needle
ifndef QUIET
TRACEBACK = -DTRACEBACK
endif
release: $(EXE)
$(EXE): $(SRC) needle_kernel.cu needle.h
	$(HIPCC) -x hip ${KERNEL_DIM} -O3 -gline-tables-only $(SRC) $(TRACEBACK) -o $(EXE) --offload-arch=$(OFFLOAD_ARCH)
clean:
	rm -f $(EXE) $(EXE).linkinfo result.txt *.sqlite *.qdrep
