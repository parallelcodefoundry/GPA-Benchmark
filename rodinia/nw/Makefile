include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = needle.cu

EXE = needle

ifndef QUIET
TRACEBACK = -DTRACEBACK
endif

release: $(EXE)

$(EXE): $(SRC)
	$(CC) ${KERNEL_DIM} -O3 -lineinfo $(SRC) $(TRACEBACK) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

enum: $(SRC)
	$(CC) ${KERNEL_DIM} -deviceemu $(SRC) $(TRACEBACK) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) ${KERNEL_DIM} -g $(SRC) $(TRACEBACK) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) ${KERNEL_DIM} -g -deviceemu $(SRC) $(TRACEBACK) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f $(EXE) $(EXE).linkinfo result.txt *.sqlite *.qdrep
