#if ISPC

#include "header.cuh"

#ifndef INFINITY
#define INFINITY        floatbits(0x7f800000U)
#endif

#ifndef NAN
#define NAN             -(INFINITY * 0.0f)
#endif

typedef int8    int8_t;
typedef uint8   uint8_t;
typedef int16   int16_t;
typedef uint16  uint16_t;
typedef int32   int32_t;
typedef uint32  uint32_t;
// uint is already defined
typedef int64   int64_t;
typedef uint64  uint64_t;
typedef float16 float16_t;

typedef int8<2>     sbyte2;
typedef int8<3>     sbyte3;
typedef int8<4>     sbyte4;
typedef uint8<2>    ubyte2;
typedef uint8<3>    ubyte3;
typedef uint8<4>    ubyte4;
typedef int16<2>    short2;
typedef int16<3>    short3;
typedef int16<4>    short4;
typedef uint16<2>   ushort2;
typedef uint16<3>   ushort3;
typedef uint16<4>   ushort4;
typedef int<2>      int2;
typedef int<3>      int3;
typedef int<4>      int4;
typedef uint<2>     uint2;
typedef uint<3>     uint3;
typedef uint<4>     uint4;
typedef float<2>    float2;
typedef float<3>    float3;
typedef float<4>    float4;
typedef float16<2>  half2;
typedef float16<3>  half3;
typedef float16<4>  half4;
typedef double<2>   double2;

static inline bool isinf(float x)
{
    return x == INFINITY || x == -INFINITY;
}

static inline float log2(float x)
{
    return log(x) / log(2.0f);
}

static inline float log10(float x)
{
    return log(x) / log(10.0f);
}

static inline float logb(float x)
{
    if(x == 0.0f || x == -0.0f) {
        return -INFINITY;
    }
    if(x == INFINITY || x == -INFINITY) {
        return INFINITY;
    }
    if(isnan(x)) {
        return NAN;
    }

    return log2(abs(x));
}

static inline float remainderf(float x, float y)
{
    const float quotient = round(x / y);
    return x - quotient * y;
}

static inline float copysignf(float mag, float sgn)
{
    const uint sign = signbits(sgn);
    uint intValue = intbits(mag) & 0x7fffffffU; // Clear the sign bit
    intValue |= sign;
    return floatbits(intValue);
}

static inline float fmod(float x, float y)
{
    y = abs(y);
    float result = remainderf(abs(x), y);
    if(signbits(result) != 0U) {
        result += y;
    }
    return copysignf(result, x);
}

static inline float sinh(float num)
{
    return (exp(num) - exp(-num)) * 0.5f;
}

static void PointerTest(
    int *p,                 // uniform int32 * varying
    int q[],                // varying int32* uniform
    int &r,                 // varying int32 &
    const int *cp,          // const uniform int32 * varying
    const int cq[],         // const varying int32 * uniform
    const int &cr,          // const varying int32 &
    varying int *vp,        // varying int32 * varying
    uniform const int *ucp, // const uniform int32 * varying
    uniform int uq[]        // uniform int32 * uniform
)
{

}

export void simple(uniform float vin[], uniform float vout[], uniform int count)
{
    assume(count % programCount == 0);

    uniform int uniA = 123;

    struct MyObject
    {
        int elems[3];
    };

    _Pragma("unroll")
    foreach (index = 0 ... count)
    {
        float v = vin[index];

        int *p = &uniA;             // uniform int* varying;
        uniform int *q = &uniA;     // uniform int* varying;

        int a = index;
        int arr[3] = { a, a, a };   // varying int32[3]
        uniform int uArr[3] = { uniA, uniA, uniA };

        PointerTest(p, arr, a, p, arr, a, arr, uArr, uArr);

        uniform struct MyObject myObj = { { uArr[0], uArr[1], uArr[2] } };      // uniform struct MyObject
        uniform struct MyObject *pObj = &myObj;                                 // uniform struct MyObject * varying

        // myObj.elems: uniform int32[3]
        // pObj->elems: varying int32[3]
        // &pObj->elems[0]: uniform int32 * varying
        PointerTest(p, &arr[0], a, p, &arr[0], a, &arr[0], &pObj->elems[0], myObj.elems);

        a = kernelDotProduct(uniA, *p, a, a + 1);

        if (index == 0)
        {
            p[index] = 123;
            q[index] = 321;

            // Warning: undefined behaviour: all program instances are writing to the same location!
            // Performance Warning: Scatter required to store value.
            // p[0] = 123;

            // Warning: undefined behaviour: all program instances are writing to the same location!
            // Performance Warning: Scatter required to store value.
            // q[0] = 321;

            a += (float)(p[index] + q[index]);
            a += (float)(*p + *q);
            
            v = log2(8.0f) * log10(100.0f) + a + sinh(1.0f);
        }
        else if(index == 1) {
            v = signbits(-0.0f) == 0x80000000U ? 1.0f : 0.0f;
            const bool b0 = isinf(exp(800.0f));
            const bool b1 = isinf(NAN);
            v += (float)b0 * 100.0f + (float)b1 * 10.0f;
        }
        else if(index == 2) {
            v = fmod(-5.0f, -2.0f);
        }
        else if(v == 3.0f)
        {
            struct MyTestStruct ms = { 0.5f, 2U, -1 };
            v = ms.f + (float)((int)ms.ui + ms.i);
        }
        else if(v == 4.0f) {
            v = (float)(EnumValue_ZERO + EnumValue_ONE + EnumValue_DUP_ONE);
        }
        else if(index == 5) {
            v = (float)sizeof(v);
        }
        else if(index == 6) {
            v = (float)sizeof(vin[0]);
        }
        else if(index == 7) {
            v = (float)sizeof(struct MyTestStruct);
        }
        else if(index == 8) {
            v = (float)sizeof(uniform struct MyTestStruct);
        }
        else if(index == 9)
        {
            const float4 d = { 1, 4, 2, 6 };
            const float4 b = { 2, 3, 4, 5 };
            const float4 c = { min(d.x, b.x), min(d.y, b.y), min(d.z, b.z), min(d.w, b.w) };
            v = (float)(c.w - c.z + c.y - c.x);
        }
        else if(index == 10) {
            v = (float)sizeof(uniform float2);
        }
        else if(index == 11) {
            v = (float)sizeof(uniform float3);
        }
        else if(index == 12) {
            v = (float)sizeof(uniform float4);
        }
        else if(index == 13) {
            v = (float)sizeof(uniform int2);
        }
        else if(index == 14) {
            v = logb(NAN);
        }
        else if(index == 15)
        {
            const int x = count_leading_zeros(0x40000000U);
            const int y = count_trailing_zeros(12);
            v = (float)(x * 100 + y);
        }
        else {
            v = sqrt(v);
        }

        vout[index] = v;
    }
}

static void PointerAssign(uniform int buffer[], int p[], int index)
{
    *p = buffer[index] + 1;
}

export void pointerTest(uniform int buffer[], uniform int count)
{
    assume(count % programCount == 0);

    foreach(index = 0 ... count)
    {
        int a = 0;
        PointerAssign(buffer, &a, index);
        buffer[index] = a;
    }
}

static task void multiTaskHandler(struct KernelLaunchParam param)
{
    uniform int* uniform buffer = param.buffer;
    uniform const int maxInstanceCountPerTask = param.maxInstanceCountPerTask;
    uniform const int totalElemCount = param.totalElemCount;

    uniform const int baseIndex = taskIndex * maxInstanceCountPerTask;
    foreach(index = baseIndex ... (baseIndex + maxInstanceCountPerTask))
    {
        if(index < totalElemCount) {
            buffer[index] = index;
        }
    }
}

export void multiTaskTest(uniform int buffer[], uniform int count)
{
    uniform const int gangCountPerTask = 8;
    uniform const uint maxInstanceCountPerTask = gangCountPerTask * ISPC_MAX_INSTANCE_COUNT_PER_GANG;
    uniform const int totalInstanceCount = (count + maxInstanceCountPerTask - 1U) & ~(maxInstanceCountPerTask - 1U);
    uniform int currTaskCount = totalInstanceCount / maxInstanceCountPerTask;
    if(currTaskCount == 0) {
        currTaskCount = 1;
    }

    struct KernelLaunchParam param = { buffer, maxInstanceCountPerTask, count };

    launch[currTaskCount] multiTaskHandler(param);
}

export void atomicTest(uniform int buffer[])
{
    foreach (index = 0 ... 16)
    {
        int oldValue = buffer[0];
        int expected;
        do
        {
            expected = oldValue;
            const int newValue = expected + index;
            oldValue = atomic_compare_exchange_global(buffer, expected, newValue);
        }
        while(oldValue != expected);
    }
}

export void atomicAddInc(uniform int buffer[], uniform int atomicObject[], uniform int count)
{
    foreach(index = 0 ... count)
    {
        const int atomIndex = atomic_add_global(atomicObject, 1);
        buffer[atomIndex] = index;
    }
}

export void atomicAddCASInc(uniform int buffer[], uniform int atomicObject[], uniform int count)
{
    foreach(index = 0 ... count)
    {
        int oldValue = atomicObject[0];
        int atomIndex;
        do
        {
            atomIndex = oldValue;
            const int newValue = atomIndex + 1;
            oldValue = atomic_compare_exchange_global(atomicObject, atomIndex, newValue);
        }
        while(oldValue != atomIndex);

        buffer[index] = atomIndex;
    }
}

static inline int subgroup_atomicAddInt(uniform int* atomObj, int value)
{
    uniform const int selectedLaneIndex = count_trailing_zeros(lanemask());
    uniform const int sum = reduce_add(value);
    const int currAccValue = exclusive_scan_add(value);
    int oldValue = 0;

    if(programIndex == selectedLaneIndex) {
        oldValue = atomic_add_global(atomObj, sum);
    }

    oldValue = broadcast(oldValue, selectedLaneIndex);
    return oldValue + currAccValue;
}

export void subgroupAtomicAddIntTest(uniform int buffer[], uniform int atomicObject[], uniform int count)
{
    foreach(index = 0 ... count)
    {
        buffer[index] = subgroup_atomicAddInt(atomicObject, buffer[index]);
    }
}

static inline float subgroup_atomicAddFloat(uniform float* atomObj, float value)
{
    uniform const int selectedLaneIndex = count_trailing_zeros(lanemask());
    uniform const float sum = reduce_add(value);
    const float currAccValue = exclusive_scan_add(value);
    float oldValue = 0.0f;

    if(programIndex == selectedLaneIndex)
    {
        oldValue = atomObj[0];
        float expected;
        do
        {
            expected = oldValue;
            const float newValue = expected + sum;
            oldValue = atomic_compare_exchange_global(atomObj, expected, newValue);
        }
        while(oldValue != expected);
    }

    oldValue = broadcast(oldValue, selectedLaneIndex);
    return oldValue + currAccValue;
}

export void subgroupAtomicAddFloatTest(uniform float buffer[], uniform float atomicObject[], uniform int count)
{
    foreach(index = 0 ... count)
    {
        buffer[index] = subgroup_atomicAddFloat(atomicObject, buffer[index]);
    }
}

static inline void subgroup_atomicMax(uniform int* atomObj, int value)
{
    uniform const int selectedLaneIndex = count_trailing_zeros(lanemask());
    uniform const int maxValue = reduce_max(value);

    if(programIndex == selectedLaneIndex) {
        atomic_max_global(atomObj, maxValue);
    }
}

export void subgroupAtomicMaxTest(uniform int buffer[], uniform int atomicObject[], uniform int count)
{
    foreach(index = 0 ... count)
    {
        subgroup_atomicMax(atomicObject, buffer[index]);
    }
}

static inline int subgroup_atomicOr(uniform int* atomObj, int value)
{
    uniform const int selectedLaneIndex = 63 - count_leading_zeros(lanemask());
    const int currAccValue = exclusive_scan_or(value);
    int oldValue = 0;

    if(programIndex == selectedLaneIndex) {
        oldValue = atomic_or_global(atomObj, currAccValue | value);
    }

    oldValue = broadcast(oldValue, selectedLaneIndex);
    return oldValue | currAccValue;
}

export void subgroupAtomicOrTest(uniform int buffer[], uniform int atomicObject[], uniform int count)
{
    foreach(index = 0 ... count)
    {
        buffer[index] = subgroup_atomicOr(atomicObject, buffer[index]);
    }
}

static inline int subgroup_atomicSwap(uniform int* atomObj, int value)
{
    uniform const int leastLaneIndex = count_trailing_zeros(lanemask());
    uniform const int significantLaneIndex = 63 - count_leading_zeros(lanemask());
    const int previousValue = rotate(value, -1);
    int oldValue = 0;

    if(programIndex == significantLaneIndex) {
        oldValue = atomic_swap_global(atomObj, value);
    }

    oldValue = broadcast(oldValue, significantLaneIndex);

    return programIndex == leastLaneIndex ? oldValue : previousValue;
}

export void subgroupAtomicSwapTest(uniform int buffer[], uniform int atomicObject[], uniform int count)
{
    foreach(index = 0 ... count)
    {
        buffer[index] = subgroup_atomicSwap(atomicObject, buffer[index]);
    }
}

export void uniformTest(uniform int dst[], uniform int count, uniform int specIndex)
{
    assume((uniform uintptr_t)dst % 64U == 0U && count % 16 == 0);

    uniform int a = 100;

    foreach (index = 0 ... count)
    {
        if(index == specIndex) {
            a = 200;
        }

        dst[index] = a;
    }
}

export void indexTest(uniform int dstIndex[], uniform int dstCount[], uniform int count)
{
    assume((uniform uintptr_t)dstIndex % 64U == 0U && (uniform uintptr_t)dstCount % 64 == 0);

    foreach (index = 0 ... count)
    {
        dstIndex[index] = programIndex;
        dstCount[index] = programCount;
    }
}

export void coherentTest(uniform int dst[], uniform int width, uniform int height)
{
    assume((uniform uintptr_t)dst % 64U == 0U);

    foreach(y = 0 ... height, x = 0 ... width)
    {
        int value;
        cif(y < height - 2) {
            value = x;
        }
        else {
            value = -x;
        }

        dst[y * width + x] = value;
    }
}

export void vectorAddTest(uniform uint dst[], uniform uint src1[], uniform uint src2[], uniform int elemCount)
{
    assume((uniform uintptr_t)dst % 64U == 0U);
    assume((uniform uintptr_t)src1 % 64U == 0U);
    assume((uniform uintptr_t)src2 % 64U == 0U);
    assume(elemCount % 256U == 0U);

    foreach (index = 0 ... elemCount) {
        dst[index] = src1[index] + src2[index];
    }
}

export void allocaTest(uniform int dstShared[], uniform int64 dstPrivate[], uniform int elemCount)
{
    assume(elemCount % 8 == 0);

    foreach(index = 0 ... elemCount)
    {
        int *sharedBuffer = (int*)alloca(8U * sizeof(uniform int));
        sharedBuffer[programIndex] = index;

        int sum32 = 0;
        for(int i = 0; i < 8; ++i) {
            sum32 += sharedBuffer[i];
        }

        dstShared[index] = sum32;

        const int bufferCount = 8 + index;
        const int offset = exclusive_scan_add(bufferCount);
        uniform const int sumCount = reduce_add(bufferCount);

        int64 *privateBuffer = (int64*)alloca(sumCount * sizeof(uniform int64));    // uniform int64 * varying
        privateBuffer = &privateBuffer[offset];

        //varying int64 *privateBuffer = (varying int64*)alloca(reduce_max(sumCount) * sizeof(uniform int64));

        for(int i = 0; i < bufferCount; ++i) {
            privateBuffer[i] = i;
        }

        int64 sum64 = 0LL;
        for(int i = 0; i < bufferCount; ++i) {
            sum64 += privateBuffer[i];
        }

        dstPrivate[index] = sum64;
    }
}

#endif

