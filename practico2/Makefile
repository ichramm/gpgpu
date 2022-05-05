##
## GPGPU
PROJECT=practico2

ROOT_DIR   =.
SRC_DIR    =$(ROOT_DIR)
BIN_DIR    =$(ROOT_DIR)/build/bin

OBJS_ROOT  =$(ROOT_DIR)/build/objs
OBJS_DIR   =$(OBJS_ROOT)

OUTPUT_DIR  =$(BIN_DIR)
OUTPUT_FILE =$(OUTPUT_DIR)/$(PROJECT)

OBJS = $(OBJS_DIR)/practico2.oc

ifeq ($(shell test -e /usr/local/cuda/bin/nvcc && echo -n yes),yes)
	NVCC=/usr/local/cuda/bin/nvcc
	ifeq (,$(findstring /usr/local/cuda/lib64, $(LD_LIBRARY_PATH)))
		LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/cuda/lib64
	endif
else
	NVCC=/usr/bin/nvcc
endif
#NVCCFLAGS:=-ccbin clang-8

$(info Using cuda compiler $(NVCC))

ifeq ($(dbg),1)
	NVCCFLAGS :=$(NVCCFLAGS) -G
else
	NVCCFLAGS :=$(NVCCFLAGS) -lineinfo
endif

# default target
build: $(OUTPUT_FILE)
$(OUTPUT_FILE): $(OUTPUT_DIR) practico2.cu
	$(NVCC) $(NVCCFLAGS) practico2.cu -o $(OUTPUT_FILE)

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
prof: build
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)" nvprof $(shell readlink -f $(OUTPUT_FILE)) $(ARGS)
