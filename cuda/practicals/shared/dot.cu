#include <iostream>

#include <cuda.h>

#include "util.hpp"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

// TODO implement dot product kernel
// works for n < 1024
template <int THREADS>
__global__
void dot_gpu_kernel(const double *x, const double* y, double *result, int n) {
    __shared__ double buffer[THREADS];
    // __shared__ double sum[1];
    auto li = threadIdx.x;
    auto gi = li + blockDim.x*blockIdx.x;
    // printf("li %d\n", int(li));
    // printf("gi %d\n", int(gi));
    if (gi<n){
        // printf("gi %d\n", int(gi));
        auto block_width = 
        buffer[li] = x[gi]*y[gi];
        auto width = THREADS/2;
        while (width > 0){
            __syncthreads();
            if (li < width){
                // printf("%d,%d\n",li, width);
                buffer[li] += buffer[li+width];
            }
            width = width/2;
        }
        // printf("buffer[1] %f\n", buffer[1]);
        *result = buffer[0];
    }
    
}
template <int THREADS>
__global__
void dot_gpu_kernel_general(const double *x, const double* y, double *result, int n) {
    __shared__ double buffer[THREADS];
    auto li = threadIdx.x;
    auto gid = li + blockDim.x*blockIdx.x;
    // printf("li %d\n", int(li));
    // printf("gi %d\n", int(gi));
    // auto nblocks = (n+blockDim.x-1)/blockDim.x;
    if (gid<n){
        // printf("gi %d\n", int(gi));
        // auto block_width = nblocks/2;
        // printf("nblocks %d\n", nblocks);
        // while (block_width>0){
        //     if (gi<(block_width*blockDim.x)){
        //         printf("blocks %d,%d\n",gi, block_width);
        //         buffer[li] += x[gi+block_width*blockDim.x]*y[gi+block_width*blockDim.x];
        //     }
        //     block_width=block_width/2;
        // }
        buffer[li] = x[gid]*y[gid];
        // buf[i] = gid<n? x[gid]*y[gid]: 0;
        auto width = THREADS/2;
        while (width > 0){
            __syncthreads();
            if (li < width){
                // printf("threads %d,%d\n",li, width);
                buffer[li] += buffer[li+width];
            }
            width = width/2;
        }
        // printf("buffer[1] %f\n", buffer[1]);
        if (li==0){
            // *result = buffer[0];
            atomicAdd(result, *buffer);
        }
    }
    // buffer[li] = gid<n? x[gid]*y[gid]: 0;
    // auto width = THREADS/2;
    // while (width > 0){
    //     __syncthreads();
    //     if (li < width){
    //         // printf("threads %d,%d\n",li, width);
    //         buffer[li] += buffer[li+width];
    //     }
    //     width = width/2;
    // }
    // // printf("buffer[1] %f\n", buffer[1]);
    // if (li==0){
    //     // *result = buffer[0];
    //     atomicAdd(result, *buffer);
    // }
    
}

double dot_gpu(const double *x, const double* y, int n) {
    static double* result = malloc_managed<double>(1);
    // TODO call dot product kernel
    const int block_dim = 64;
    auto grid_dim = (n+block_dim-1)/block_dim;
    // dot_gpu_kernel<block_dim><<<grid_dim,block_dim>>>(x, y, result, n);
    dot_gpu_kernel_general<block_dim><<<grid_dim,block_dim>>>(x, y, result, n);
    cudaDeviceSynchronize();
    return *result;
}

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes*1e-9 << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    auto result   = dot_gpu(x_d, y_d, n);
    cudaDeviceSynchronize();
    auto expected = dot_host(x_h, y_h, n);
    printf("expected %f got %f\n", (float)expected, (float)result);

    return 0;
}

