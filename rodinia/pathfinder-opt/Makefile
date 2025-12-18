include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = pathfinder

release: $(EXE)

$(EXE): $(SRC)
	$(CC) $(SRC) -O3 -lineinfo -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f pathfinder *.sqlite *.qdrep
