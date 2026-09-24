# HIP (Frontier/MI250X) build of GPA hotspot. Source: AMD HIP-Examples rodinia_3.0/hip/hotspot (native port)
# + APPEB patch (GPA EDITABLE REGION markers, WARP_SIZE define, PROFILING off, CRLF->LF).
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a

SRC = hotspot.cu
EXE = hotspot

release: $(EXE)

$(EXE): $(SRC)
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) -O3 -gline-tables-only $(KERNEL_DIM) $(SRC) -o $(EXE)

clean:
	rm -f $(EXE) output.out *.qdrep *.sqlite
