
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>

#define _USE_MATH_DEFINES   1
#include <math.h>

#if __CUDA_ARCH__
#pragma push
#pragma nv_math_opt off
#endif
static __device__ inline float ComputePrecise(float a, float b)
{
    return __powf(a, b);
}
#if __CUDA_ARCH__
#pragma pop
#endif

static __global__ void testKernel(float pSrcDst[2])
{
    const unsigned gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid == 0U)
    {
        pSrcDst[0] = ComputePrecise(pSrcDst[0], pSrcDst[1]);
        pSrcDst[1] = __powf(pSrcDst[2], pSrcDst[3]);
    }
}

int main(int argc, const char* argv[])
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed with error: %s. Do you have a CUDA-capable GPU properly installed?\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    constexpr unsigned maxElemCount = 256U;
    float hostResults[maxElemCount]{};
    for (unsigned i = 0U; i < maxElemCount; i += 2)
    {
        hostResults[i + 0] = 0.001345678f;
        hostResults[i + 1] = 11.0000033f;
    }

    float* devBuf = nullptr;

    do
    {
        // Allocate GPU buffer devBuf
        cudaStatus = cudaMalloc(&devBuf, maxElemCount * sizeof(*devBuf));
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc devBuf failed with error: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(devBuf, hostResults, sizeof(hostResults), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy from hostRsults to devBuf failed with error: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        // Launch a kernel on the GPU with one thread for each element.
        void* args[]{ &devBuf };
        cudaStatus = cudaLaunchKernel((void*)testKernel, dim3(1U, 1U, 1U), dim3(maxElemCount, 1U, 1U), args, 0U, nullptr);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "launch kernel `testKernel` failed with error: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        // Copy output the result values from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(hostResults, devBuf, sizeof(hostResults), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy from devBuf to hostResults failed with error: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        // Verify the result
        union FloatToInt { float f; unsigned i; };
        printf("IEEE-compliant value: 0x%08X, fast-math result: 0x%08X, host result: 0x%08X\n",
            FloatToInt{ hostResults[0] }.i, FloatToInt{ hostResults[1] }.i, FloatToInt{ powf(hostResults[4], hostResults[5])}.i);
    }
    while (false);

    if (devBuf != nullptr) {
        cudaFree(devBuf);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

