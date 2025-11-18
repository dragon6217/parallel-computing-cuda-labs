CXX = g++ 
CXXFLAGS = -g -std=c++11 -Wall -Wno-sign-compare -O3 -fopenmp

NVCXX = nvcc
NVCXXFLAGS = -g -std=c++11 -O3

CUDALIB = 
CUDAINC = 

SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(wildcard $(OBJDIR)/*.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all: bin/matmul

bin:
	mkdir -p bin

bin/matmul: $(OBJS) $(CUOBJS) 
	mkdir -p bin
	@echo "OBJ: "$(OBJS)
	@echo "CUOBJ: "$(CUOBJS)
	$(CXX) $^ -o $@ $(CXXFLAGS) $(CUDALIB) -lcudart $(CUDAINC)
	@echo "Compiled "$<" successfully!"


.PHONY: test clean

$(CUOBJS): $(CUOBJDIR)/%.o : $(SRCDIR)/%.cu
	mkdir -p cuobj
	@echo $(NVCXX) $(NVCXXFLAGS) "-Iinclude -c" $< "-o" $@
	@$(NVCXX) $(NVCXXFLAGS) -Iinclude -c $< -o $@
	@echo "CUDA Compiled "$<" successfully!"

clean: 
	rm -f $(CUOBJS) $(CUOBJS:%.o=%.d) 
	rm -rf bin/*
