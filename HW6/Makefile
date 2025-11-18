CXX = g++ 
CXXFLAGS = -g -fopenmp -std=c++11 -Wall -Wno-sign-compare -O3

# [수정] nvcc 자동 탐지 및 플래그 수정
NVCXX = nvcc 
NVCXXFLAGS = -g -std=c++11 -O3

# [수정] 경로 비워두기 (시스템 기본값 사용)
CUDALIB = 
CUDAINC = 

SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(SRCS:$(SRCDIR)/%.cc=$(OBJDIR)/%.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all: bin/pr

bin/pr: $(OBJS) $(CUOBJS) 
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

$(OBJS): $(OBJDIR)/%.o : $(SRCDIR)/%.cc
	mkdir -p obj 
	@echo $(CXX)  -c $< -o $@ $(CXXFLAGS) $(CUDALIB) $(CUDAINC)
	$(CXX)  -c $< -o $@ $(CXXFLAGS) $(CUDALIB) $(CUDAINC)
	@echo "Compiled "$<" successfully!"

clean:
	rm -f $(OBJS) $(OBJS:%.o=%.d) 
	rm -f $(CUOBJS) $(CUOBJS:%.o=%.d) 
	rm -rf bin/*

#########################
# Submit
##########################
submit_facebook:
	mkdir -p result
	condor_submit pr_facebook.cmd

submit_livejournal:
	mkdir -p result
	condor_submit pr_livejournal.cmd