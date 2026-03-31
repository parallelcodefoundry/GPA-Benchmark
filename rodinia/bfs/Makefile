include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -ccbin $(NVCC_CCBIN)

INCLUDE := $(CUDA_DIR)/include

SRC = bfs.cu

EXE = bfs

release: $(EXE)

$(EXE): $(SRC)
	$(CC) $(SRC) -O3 -lineinfo -allow-unsupported-compiler -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lstdc++

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f $(EXE) $(EXE).linkinfo result.txt *.qdrep *.sqlite
