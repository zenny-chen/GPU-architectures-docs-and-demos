#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common.h"
#include "header.cuh"

#include <cmath>
#include <utility>
#include <algorithm>
#include <atomic>
#include <thread>


static __global__ void addKernel(int* __restrict__ c, const int* __restrict__ a, const int* __restrict__ b)
{
    const unsigned localThreadID = threadIdx.x;
    const unsigned blockSize = blockDim.x;
    const unsigned blockID = blockIdx.x;
    const unsigned globalThreadID = blockID * blockSize + localThreadID;

    c[globalThreadID] = kernelDotProduct(a[globalThreadID], b[globalThreadID], a[globalThreadID] + 1, b[globalThreadID] - 1);
}

extern "C" auto dotProductWithCUDA() -> void
{
    puts("\n======== Begin dotProductWithCUDA ========\n");

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed: %s! Do you have a CUDA-capable GPU installed?\n", cudaGetErrorString(cudaStatus));
        return;
    }

    constexpr unsigned elemCount = 4096U;
    constexpr size_t bufferSize = elemCount * sizeof(int);

    cudaFuncAttributes kernelAttrs{};
    cudaStatus = cudaFuncGetAttributes(&kernelAttrs, (const void*)addKernel);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaFuncGetAttributes for addKernel failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    int* hostBuffer = (int*)calloc(elemCount, sizeof(int));
    if (hostBuffer == nullptr) return;
    for (unsigned i = 0U; i < elemCount; ++i) {
        hostBuffer[i] = i;
    }

    int* dev_c = nullptr;
    int* dev_a = nullptr;
    int* dev_b = nullptr;

    do
    {
        cudaStatus = cudaMalloc(&dev_a, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc for dev_a failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&dev_b, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc for dev_b failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&dev_c, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc for dev_c failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(dev_a, hostBuffer, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy from hostBuffer to dev_a failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(dev_b, hostBuffer, bufferSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy from hostBuffer to dev_b failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemset(dev_c, 0, bufferSize);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemset for dev_c failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        void* args[]{ &dev_c, &dev_a, &dev_b };
        cudaStatus = cudaLaunchKernel((const void*)addKernel, dim3(elemCount / kernelAttrs.maxThreadsPerBlock, 1U, 1U), dim3(kernelAttrs.maxThreadsPerBlock, 1U, 1U), args, 0U, nullptr);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaLaunchKernel for addKernel failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        // Read back the result
        cudaStatus = cudaMemcpy(hostBuffer, dev_c, bufferSize, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy from dev_c to hostBuffer failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        // Verify the result
        for (unsigned i = 0; i < elemCount; ++i)
        {
            const int correctValue = i * (i + 1) + i * (i - 1);
            if (hostBuffer[i] != correctValue)
            {
                fprintf(stderr, "Result error @%u: %d, correct value: %d\n", i, hostBuffer[i], correctValue);
                break;
            }
        }
    }
    while (false);

    if (hostBuffer != nullptr) {
        free(hostBuffer);
    }
    if (dev_a != nullptr) {
        cudaFree(dev_a);
    }
    if (dev_b != nullptr) {
        cudaFree(dev_b);
    }
    if (dev_c != nullptr) {
        cudaFree(dev_c);
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed: %s!\n", cudaGetErrorString(cudaStatus));
        return;
    }

    puts("======== Complete dotProductWithCUDA ========\n");
}


extern "C" void* ISPCAlloc(void** handlePtr, int64_t size, int alignment)
{
#if _WIN32
    void* ptr = _aligned_malloc(size_t(size), size_t(alignment));
#else
    void* ptr = malloc(size_t(size));
#endif

    * handlePtr = ptr;
    return ptr;
}

extern "C" void ISPCLaunch(void** handlePtr, void* f, void* data, int count0, int count1, int count2)
{
    using TaskFuncPtrType = auto(*)(void* data, int threadIndex, int threadCount, int taskIndex, int taskCount, int taskIndex0, int taskIndex1, int taskIndex2, int taskCount0, int taskCount1, int taskCount2) -> void;
    // data will point to a KernelLaunchParam object:
    // KernelLaunchParam* pParam = (KernelLaunchParam*)data;
    TaskFuncPtrType const taskFuncPtr = TaskFuncPtrType(f);

    std::thread threads[16];
    for (int taskIndex = 0; taskIndex < count0; ++taskIndex)
    {
        threads[taskIndex] = std::thread([&taskFuncPtr, data, taskIndex, count0, count1, count2]() {
            taskFuncPtr(data, taskIndex, count0, taskIndex, count0, taskIndex, 0, 0, count0, count1, count2);
            });
    }

    for (int taskIndex = 0; taskIndex < count0; ++taskIndex) {
        threads[taskIndex].join();
    }
}

extern "C" void ISPCSync(void* handle)
{
#if _WIN32
    _aligned_free(handle);
#else
    free(handle);
#endif
}


template <typename ATOMIC_T>
static inline auto atomicCAS(ATOMIC_T* atom, decltype(atom->load()) expected, decltype(atom->load()) newValue) -> decltype(atom->load())
{
    std::atomic_compare_exchange_strong(atom, &expected, newValue);
    return expected;
}

extern "C" auto CPPTest() -> void
{
    puts("Hello, world!!");
    constexpr auto value = M_PI + M_E;

    std::atomic_int atomA = int(value);

    auto result = atomicCAS(&atomA, int(value), 10);
    printf("First result is: %d, atomA = %d. Is successful ? %s\n", result, atomA.load(), result == int(value) ? "YES" : "NO");

    result = atomicCAS(&atomA, int(value), 100);
    printf("The second result is: %d, atomA = %d, Is successful ? %s\n", result, atomA.load(), result == int(value) ? "YES" : "NO");

    printf("signbit(-0.0) = %d\n", std::signbit(-0.0f));
    printf("fmod(-5, -2) = %f\n", std::fmod(-5.0f, -2.0f));

    printf("logb(-4.0) = %f, logb(8.0) = %f, logb(0.0) = %f, logb(-INF) = %f, logb(NaN) = %f\n", std::logb(-4.0f), std::logb(8.0f), std::logb(0.0f), std::logb(-INFINITY), std::logb(NAN));
}

