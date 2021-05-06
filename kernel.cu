#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

#define N 25

__global__ void init(unsigned int seed, curandState_t* states) 
{
    curand_init(seed, blockIdx.x, 10000, &states[blockIdx.x]); // initializing the random state for each thread
}

__global__ void randoms(curandState_t* states, unsigned int* numbers) 
{
    numbers[blockIdx.x] = curand(&states[blockIdx.x]) % 100;
}

int main() 
{
    // initializing the random states for each thread
    curandState_t* states;
    cudaMalloc((void**)&states, N * sizeof(curandState_t)); // allocate space on the GPU for the random states 
    //init << <N, 1 >> > (time(0), states); // seed using time
    init << <N, 1 >> > (13, states); // constant seed for debugging

    // send a brain to the GPU to mutate each neuron in parrelell

    unsigned int* gpu_nums;
    cudaMalloc((void**)&gpu_nums, N * sizeof(unsigned int));

    randoms << <N, 1 >> > (states, gpu_nums);

    unsigned int cpu_nums[N];
    cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(unsigned int), cudaMemcpyDeviceToHost); // copy the random numbers back 

    for (int i = 0; i < N; i++) {
        printf("%u\n", cpu_nums[i]);
    }

    cudaFree(states);
    cudaFree(gpu_nums);
    return 0;
}
