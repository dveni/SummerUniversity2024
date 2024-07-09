NVCC_FLAGS=-arch=sm_80 -std=c++11 -O3 -I../include
GCC_FLAGS=-std=c++11 -O3 -fopenmp -I../include

LINK=-lcublas -lcuda

reduction : reduction.cu
	nvcc $(NVCC_FLAGS) reduction.cu -o reduction $(LINK)

