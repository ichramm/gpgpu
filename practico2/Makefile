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

NVCCFLAGS=-ccbin clang-8

ifeq ($(dbg),1)
	NVCCFLAGS :=$(NVCCFLAGS) -G
else
	NVCCFLAGS :=$(NVCCFLAGS) -lineinfo
endif

# default target
build: $(OUTPUT_FILE)
$(OUTPUT_FILE): $(OUTPUT_DIR) practico2.cu
	nvcc $(NVCCFLAGS) practico2.cu -o $(OUTPUT_FILE)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

.PHONY: clean
clean:
	rm -vf $(OBJS_ROOT)/**/*.o
	rm -vf $(OUTPUT_FILE)

.PHONY: run
run: build
	$(shell readlink -f $(OUTPUT_FILE)) $(ARGS)
