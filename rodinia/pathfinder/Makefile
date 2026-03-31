include ../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -ccbin $(NVCC_CCBIN)
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = pathfinder

ifndef QUIET
BENCH_PRINT = -DBENCH_PRINT
endif

release: $(EXE)

$(EXE): $(SRC)
	$(CC) $(SRC) -O3 -lineinfo $(BENCH_PRINT) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -arch=sm_$(SM_VERSION) --allow-unsupported-compiler

enum: $(SRC)
	$(CC) -deviceemu $(SRC) $(BENCH_PRINT) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) -g $(SRC) $(BENCH_PRINT) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) $(BENCH_PRINT) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm -f pathfinder *.sqlite *.qdrep
