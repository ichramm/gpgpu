##
## GPGPU
PROJECT=lab

ROOT_DIR    =.
INCLUDE_DIR =$(ROOT_DIR)/include
SRC_DIR     =$(ROOT_DIR)/src
BIN_DIR     =$(ROOT_DIR)/build/bin

OBJS_ROOT  =$(ROOT_DIR)/build/objs
OBJS_DIR   =$(OBJS_ROOT)

OUTPUT_DIR  =$(BIN_DIR)
OUTPUT_FILE =$(OUTPUT_DIR)/$(PROJECT)


ifeq ($(shell test -e /usr/local/cuda/bin/nvcc && echo -n yes),yes)
	NVCC=/usr/local/cuda/bin/nvcc
	CUDALIBDIR=-L/usr/local/cuda/lib64
	ifeq (,$(findstring /usr/local/cuda/lib64, $(LD_LIBRARY_PATH)))
		LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/cuda/lib64
	endif
else
	NVCC=/usr/bin/nvcc
endif

$(info Using cuda compiler $(NVCC))

INCLUDES =-I$(INCLUDE_DIR)
NVCCFLAGS :=$(NVCCFLAGS) -g -std=c++11 -x cu -dc --compiler-options -Wall --compiler-options -Wextra
NVLDFLAGS :=-g
LIBRARIES :=$(CUDALIBDIR) -lcurand -lcusparse
#-ccbin clang-8

ifeq ($(dbg),1)
	NVCCFLAGS :=$(NVCCFLAGS) -G
	NVLDFLAGS :=$(NVLDFLAGS) -G
else
	NVCCFLAGS :=$(NVCCFLAGS) -O3 -lineinfo
	NVLDFLAGS :=$(NVLDFLAGS) -O3 -lineinfo
endif

OBJS=$(OBJS_DIR)/main.o $(OBJS_DIR)/ejercicio1.obj $(OBJS_DIR)/ejercicio2.obj $(OBJS_DIR)/ejercicio3.obj

# default target
build: $(OUTPUT_FILE)
$(OUTPUT_FILE): $(OBJS_DIR) $(OUTPUT_DIR) $(OBJS)
	$(NVCC) $(NVLDFLAGS) $(OBJS) $(LIBRARIES) -o $(OUTPUT_FILE)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OBJS_DIR):
	mkdir -p $(OBJS_DIR)

$(OBJS_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) "$<" -o "$(OBJS_DIR)/$(*F).o"

$(OBJS_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) "$<" -o "$(OBJS_DIR)/$(*F).obj"

.PHONY: clean
clean:
	rm -vf $(OBJS_ROOT)/*.o $(OBJS_ROOT)/*.obj
	rm -vf $(OUTPUT_FILE)

.PHONY: run
run: build
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)" $(shell readlink -f $(OUTPUT_FILE)) $(ARGS)

.PHONY: prof
prof: NVCCFLAGS:=$(NVCCFLAGS) -DBENCH_TIMES=1 -DPROFILER=1
prof: build
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)" nvprof \
	-m gld_transactions,gld_efficiency,gst_transactions,gst_efficiency,shared_efficiency \
	$(shell readlink -f $(OUTPUT_FILE)) $(ARGS)

## -m gld_transactions,dram_read_transactions,
## l2_read_transactions,l2_tex_read_transactions,
## local_load_transactions,local_load_transactions
# SOURCES=$(SRC_DIR)/lab.cu $(SRC_DIR)/lab.h $(SRC_DIR)/utils.h $(SRC_DIR)/bl_matrix.hpp $(SRC_DIR)/csr_matrix.hpp \
# 	$(SRC_DIR)/ejercicio1.cuh $(SRC_DIR)/ejercicio2.cuh \
# 	$(SRC_DIR)/ejercicio3.cuh
