#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "util.hpp"

// TODO : implement a kernel that reverses a string of length n in place
__global__
void reverse_string(char* str, int n){
    __shared__  char buffer[2];
    auto li = threadIdx.x;
    auto block_start = + blockIdx.x*blockDim.x;
    auto gi = li + block_start;

    // printf("i %d\n", int(i));
    if (gi<(n)/2){
        // printf("i < n %d\n", int(i));
        buffer[0] = str[gi];
        buffer[1] = str[n-1-gi];
        str[n-1-gi] = buffer[0];
        str[gi] = buffer[1];
    }
    
}

int main(int argc, char** argv) {
    // check that the user has passed a string to reverse
    if(argc<2) {
        std::cout << "useage : ./string_reverse \"string to reverse\"\n" << std::endl;
        exit(0);
    }

    // determine the length of the string, and copy in to buffer
    auto n = strlen(argv[1]);
    auto string = malloc_managed<char>(n+1);
    std::copy(argv[1], argv[1]+n, string);
    string[n] = 0; // add null terminator

    std::cout << "string to reverse:\n" << string << "\n";

    // TODO : call the string reverse function
    auto grid_dim = 1;
    auto block_dim = 128;
    std::cout << "n:\n" << n << "\n";
    reverse_string<<<grid_dim, block_dim, (block_dim)*sizeof(char)>>>(string, n);

    // print reversed string
    cudaDeviceSynchronize();
    std::cout << "reversed string:\n" << string << "\n";

    // free memory
    cudaFree(string);

    return 0;
}

