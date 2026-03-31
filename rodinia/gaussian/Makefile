include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -ccbin $(NVCC_CCBIN)

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(EXE)

$(EXE): $(SRC)
	$(CC) -lineinfo -O3 $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lm

clean:
	rm gaussian *.qdrep *.sqlite
