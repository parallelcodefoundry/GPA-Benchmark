# HIP build of GPA hotspot-opt (hipify-perl of GPA CUDA source). nvcc -O3 -lineinfo -> hipcc --offload-arch=gfx90a -O3 -gline-tables-only
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH ?= /opt/rocm-7.0.2
HIPCC ?= $(ROCM_PATH)/bin/hipcc
OFFLOAD_ARCH ?= gfx90a
hotspot: hotspot.cu
	$(HIPCC) --offload-arch=$(OFFLOAD_ARCH) -O3 -gline-tables-only $(KERNEL_DIM) hotspot.cu -o hotspot
clean:
	rm -f hotspot output.out
