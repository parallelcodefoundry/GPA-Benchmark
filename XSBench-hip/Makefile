#===============================================================================
# User Options
#===============================================================================

COMPILER    ?= amd
# Toolchain from the environment (mirror of CUDA_HOME/SM_VERSION): ROCm 7.0.2 unless ROCM_PATH/HIPCC say otherwise.
ROCM_PATH   ?= /opt/rocm-7.0.2
HIPCC       ?= $(ROCM_PATH)/bin/hipcc
OPTIMIZE    ?= yes
DEBUG       ?= no
PROFILE     ?= no
OFFLOAD_ARCH ?= gfx90a
REG_COUNT   ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.cu \
io.cu \
Simulation.cu \
GridInit.cu \
XSutils.cu \
Materials.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := 

# Linker Flags
LDFLAGS = -lm

# AMD Compiler (HIP port of GPA's CUDA XSBench; delta: nvcc -> hipcc, -arch=sm_XX -> --offload-arch, -lineinfo -> -gline-tables-only, -x hip)
# XSBench-hip is self-contained: XSbench_shared_header.h is a copy of GPA's XSBench/XSbench_shared_header.h, and
# Main.cu exits 0 whatever XSBench's own hash check says (the harness checks the checksum; see Main.cu).
ifeq ($(COMPILER),amd)
  CC = $(HIPCC)
  CFLAGS += -Wall -O3 -gline-tables-only --offload-arch=$(OFFLOAD_ARCH) -std=c++17
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS  += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Register Count Flag
ifneq ($(REG_COUNT),)
  # no hipcc equivalent of -maxrregcount; closest is __launch_bounds__ in source
  $(warning REG_COUNT ignored on AMD)
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) -x hip -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) XSbench_header.cuh

run:
	./$(program)
