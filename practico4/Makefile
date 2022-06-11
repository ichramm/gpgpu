##
## GPGPU
PROJECT=practico4

ROOT_DIR   =.
SRC_DIR    =$(ROOT_DIR)/src
BIN_DIR    =$(ROOT_DIR)/build/bin

OBJS_ROOT  =$(ROOT_DIR)/build/objs
OBJS_DIR   =$(OBJS_ROOT)

OUTPUT_DIR  =$(BIN_DIR)
OUTPUT_FILE =$(OUTPUT_DIR)/$(PROJECT)

OBJS = $(OBJS_DIR)/practico4.oc

ifeq ($(shell test -e /usr/local/cuda/bin/nvcc && echo -n yes),yes)
	NVCC=/usr/local/cuda/bin/nvcc
	ifeq (,$(findstring /usr/local/cuda/lib64, $(LD_LIBRARY_PATH)))
		LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/cuda/lib64
	endif
else
	NVCC=/usr/bin/nvcc
endif

$(info Using cuda compiler $(NVCC))

NVCCFLAGS:=$(NVCCFLAGS) --compiler-options -Wall --compiler-options -Wextra -std=c++11 -g -lcurand
#-ccbin clang-8

ifeq ($(dbg),1)
	NVCCFLAGS :=$(NVCCFLAGS) -G
else
	NVCCFLAGS :=$(NVCCFLAGS) -O3 -lineinfo
endif

SOURCES=$(SRC_DIR)/practico4.cu $(SRC_DIR)/practico4.h $(SRC_DIR)/utils.h \
	$(SRC_DIR)/ejercicio1.cuh $(SRC_DIR)/ejercicio2.cuh \
	$(SRC_DIR)/ejercicio3.cuh $(SRC_DIR)/ejercicio4.cuh

# default target
build: $(OUTPUT_FILE)
$(OUTPUT_FILE): $(OUTPUT_DIR) $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/practico4.cu -o $(OUTPUT_FILE)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

.PHONY: clean
clean:
	rm -vf $(OBJS_ROOT)/**/*.o
	rm -vf $(OUTPUT_FILE)

.PHONY: run
run: build
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)" $(shell readlink -f $(OUTPUT_FILE)) $(ARGS)

.PHONY: prof
prof: NVCCFLAGS:=$(NVCCFLAGS) -DBENCH_TIMES=1
prof: build
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)" nvprof \
	-m gld_transactions,gld_efficiency,gst_transactions,gst_efficiency,shared_efficiency \
	$(shell readlink -f $(OUTPUT_FILE)) $(ARGS)

## -m gld_transactions,dram_read_transactions,
## l2_read_transactions,l2_tex_read_transactions,
## local_load_transactions,local_load_transactions
