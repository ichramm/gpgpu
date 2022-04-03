##
## GPGPU
PROJECT=practico1

Target ?= release

CC      ?=clang
CXXC    ?=clang++
CFLAGS   :=$(CFLAGS) -c -Wall -Wextra -Wno-format-extra-args -std=c11
CXXFLAGS :=$(CFLAGS) -c -Wall -Wextra -std=c++11
INCLUDES =
LINKER  ?=$(CXXC)
LIBS     =-lpthread
LDFLAGS  =

ifeq (Debug, $(findstring Debug,$(Target)))
	CFLAGS   :=$(CFLAGS) -g3
	CXXFLAGS :=$(CXXFLAGS) -g3
else
	CFLAGS   :=$(CFLAGS) -O3 -march=native -mtune=native -funroll-loops
	CXXFLAGS :=$(CXXFLAGS) -O3 -march=native -mtune=native -funroll-loops
endif

ROOT_DIR   =.
SRC_DIR    =$(ROOT_DIR)
BIN_DIR    =$(ROOT_DIR)/build/bin

OBJS_ROOT  =$(ROOT_DIR)/build/objs
OBJS_DIR   =$(OBJS_ROOT)/$(Target)

OUTPUT_DIR  =$(BIN_DIR)/$(Target)
OUTPUT_FILE =$(OUTPUT_DIR)/$(PROJECT)

OBJS = $(OBJS_DIR)/practico1.o

# default target
build: $(OUTPUT_FILE)
$(OUTPUT_FILE): $(OBJS_DIR) $(OUTPUT_DIR) $(OBJS)
	$(LINKER) $(LDFLAGS) $(OBJS) $(LIBS) -o "$(OUTPUT_FILE)"

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(OBJS_DIR):
	mkdir -p $(OBJS_DIR)

# Pattern rules:
$(OBJS_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) "$<" -o "$(OBJS_DIR)/$(*F).o"

.PHONY: clean
clean:
	rm -vf $(OBJS_ROOT)/**/*.o
	rm -vf $(OUTPUT_FILE)

.PHONY: run
run: build
	$(shell readlink -f $(OUTPUT_FILE)) $(ARGS)

.PHONY: perf
perf: CFLAGS:=$(CFLAGS) -pg
perf: build
	perf record -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations $(shell readlink -f $(OUTPUT_FILE)) $(ARGS)
	perf report

callgrind: build
	valgrind --tool=callgrind $(shell readlink -f $(OUTPUT_FILE)) $(ARGS)


cachegrind: build
	valgrind --tool=cachegrind $(shell readlink -f $(OUTPUT_FILE)) $(ARGS)

#valgrind:


$(OBJS_DIR)/%-cpp.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) "$<" -o "$(OBJS_DIR)/$(*F)-cpp.o"

$(OBJS_DIR)/practico1-cpp.o: $(SRC_DIR)/practico1.cpp $(SRC_DIR)/bench.h

cpp: $(OBJS_DIR) $(OUTPUT_DIR) $(OBJS_DIR)/practico1-cpp.o $(OBJS_DIR)/dontoptimize.o
	$(LINKER) $(LDFLAGS) $(OBJS_DIR)/practico1-cpp.o $(OBJS_DIR)/dontoptimize.o $(LIBS) -o "$(OUTPUT_FILE)-cpp"

cpp-run: cpp
	$(shell readlink -f $(OUTPUT_FILE)-cpp) $(ARGS)
