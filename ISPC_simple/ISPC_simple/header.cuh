#ifndef ISPC_HEADER
#define ISPC_HEADER     1

#if !ISPC
#define uniform
#define noinline
typedef unsigned    uint;
#endif  // !ISPC

#if __CUDACC__
#undef noinline
#define noinline    __noinline__
#else
#define __device__
#endif

#define ISPC_MAX_INSTANCE_COUNT_PER_GANG    8

enum EnumValue
{
    EnumValue_ZERO,
    EnumValue_ONE,
    EnumValue_DUP_ONE = 1
};

//union MyTestUnion
// ISPC does not support union type
struct MyTestStruct
{
    float f;
    uint ui;
    int i;
};

struct KernelLaunchParam
{
    uniform int* uniform buffer;
    uniform int maxInstanceCountPerTask;
    uniform int totalElemCount;
};

static __device__ noinline int kernelDotProduct(int x0, int y0, int x1, int y1)
{
    const int x = x0 * x1;
    const int y = y0 * y1;
    const int sum = x + y;
    return sum;
}

#endif // !ISPC_HEADER

