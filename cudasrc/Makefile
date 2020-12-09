CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -dc -std=c++11 --expt-relaxed-constexpr
GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60

SRCS = main.cu
INCS = *.h

raytracer: raytracer.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o raytracer raytracer.o

raytracer.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o raytracer.o -c main.cu

clean:
	rm -f raytracer raytracer.o