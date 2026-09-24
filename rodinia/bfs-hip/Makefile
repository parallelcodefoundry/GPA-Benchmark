# HIP (Frontier/MI250X) build of GPA bfs. Source: AMD HIP-Examples rodinia_3.0/hip/bfs (native port)
# + APPEB patch (GPA EDITABLE REGION markers, PROFILING blocks removed, *.hip.cu -> *.cu, CRLF->LF).
# SM_VERSION is accepted and ignored (the CUDA Makefile never used it either).
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a

SRC = bfs.cu
EXE = bfs

release: $(EXE)

$(EXE): $(SRC) kernel.cu kernel2.cu
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) $(SRC) -O3 -gline-tables-only -o $(EXE)

clean:
	rm -f $(EXE) result.txt *.qdrep *.sqlite
