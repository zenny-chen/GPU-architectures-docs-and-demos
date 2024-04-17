// main.c : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "common.h"

extern int MyASMTest(int a);
extern void CPPTest(void);

extern void simple(float vin[], float vout[], int count);
extern void pointerTest(int buffer[], int count);
extern void atomicTest(int buffer[]);
extern void uniformTest(int dst[], int count, int specIndex);
extern void indexTest(int dstIndex[], int dstCount[], int count);
extern void multiTaskTest(int buffer[], int count);
extern void atomicAddInc(int buffer[], int atomicObject[], int count);
extern void atomicAddCASInc(int buffer[], int atomicObject[], int count);
extern void subgroupAtomicAddIntTest(int buffer[], int atomicObject[], int count);
extern void subgroupAtomicAddFloatTest(float buffer[], float atomicObject[], int count);
extern void subgroupAtomicMaxTest(int buffer[], int atomicObject[], int count);
extern void subgroupAtomicOrTest(int buffer[], int atomicObject[], int count);
extern void subgroupAtomicSwapTest(int buffer[], int atomicObject[], int count);
extern void coherentTest(int dst[], int width, int height);
extern void vectorAddTest(unsigned dst[], unsigned src1[], unsigned src2[], int elemCount);
extern void VectorAddASM(unsigned dst[], unsigned src1[], unsigned src2[], int elemCount);
extern void allocaTest(int dstShared[], long long dstPrivate[], int elemCount);
extern void dotProductWithCUDA(void);


static void VectorAddTest(uint8_t* alignedBuffer)
{
    enum {
        ELEM_COUNT = 4 * 1024 * 1024,
        BUFFER_SIZE = ELEM_COUNT * sizeof(unsigned),
        LOOP_COUNT = 20
    };

    unsigned* dst = (unsigned*)alignedBuffer;
    unsigned* src1 = (unsigned*)(alignedBuffer + BUFFER_SIZE);
    unsigned* src2 = (unsigned*)(alignedBuffer + BUFFER_SIZE * 2U);
    for (int i = 0; i < ELEM_COUNT; ++i)
    {
        dst[i] = 0U;
        src1[i] = i + 1U;
        src2[i] = i * 2U;
    }

    vectorAddTest(dst, src1, src2, ELEM_COUNT);
    // Verify the result
    bool success = true;
    for (int i = 0; i < ELEM_COUNT; ++i)
    {
        const unsigned expected = src1[i] + src2[i];
        if (dst[i] != expected)
        {
            fprintf(stderr, "VectorAddTest error! @%d, expected result: %u, current result: %u!\n", i, expected, dst[i]);
            success = false;
            break;
        }
    }
    if (!success) {
        return;
    }

    long long beginTimestamps[LOOP_COUNT];
    long long endTimestamps[LOOP_COUNT];
    unsigned long long sum = 0ULL;

    for (int iLoop = 0; iLoop < LOOP_COUNT; ++iLoop)
    {
        struct timespec timeSpec;
        timespec_get(&timeSpec, TIME_UTC);
        beginTimestamps[iLoop] = GetTimestampDuration(&timeSpec);

        vectorAddTest(dst, src1, src2, ELEM_COUNT);
        sum += dst[100];

        timespec_get(&timeSpec, TIME_UTC);
        endTimestamps[iLoop] = GetTimestampDuration(&timeSpec);
    }

    double avgDuration, maxDuration;
    double minDuration = FetchDurationsAmongTimestamps(LOOP_COUNT, beginTimestamps, endTimestamps, &avgDuration, &maxDuration);
    printf("vectorAddTest total %d elements, min duration: %.2fms, avg duration: %.2fms, max duration: %.2fms, sum = %llu\n", LOOP_COUNT, minDuration, avgDuration, maxDuration, sum);

    VectorAddASM(dst, src1, src2, ELEM_COUNT);
    // Verify the result
    success = true;
    for (int i = 0; i < ELEM_COUNT; ++i)
    {
        const unsigned expected = src1[i] + src2[i];
        if (dst[i] != expected)
        {
            fprintf(stderr, "VectorAddASM error! @%d, expected result: %u, current result: %u!\n", i, expected, dst[i]);
            success = false;
            break;
        }
    }
    if (!success) {
        return;
    }

    sum = 0ULL;
    for (int iLoop = 0; iLoop < LOOP_COUNT; ++iLoop)
    {
        struct timespec timeSpec;
        timespec_get(&timeSpec, TIME_UTC);
        beginTimestamps[iLoop] = GetTimestampDuration(&timeSpec);

        vectorAddTest(dst, src1, src2, ELEM_COUNT);
        sum += dst[100];

        timespec_get(&timeSpec, TIME_UTC);
        endTimestamps[iLoop] = GetTimestampDuration(&timeSpec);
    }
    minDuration = FetchDurationsAmongTimestamps(LOOP_COUNT, beginTimestamps, endTimestamps, &avgDuration, &maxDuration);
    printf("VectorAddASM total %d elements, min duration: %.2fms, avg duration: %.2fms, max duration: %.2fms, sum = %llu\n", LOOP_COUNT, minDuration, avgDuration, maxDuration, sum);
}

int main(void)
{
    CPPTest();
    printf("MyASMTest result: %d\n", MyASMTest(1));

    enum {
        elemCount = 16
    };

    float vin[elemCount], vout[elemCount];
    for (int i = 0; i < elemCount; ++i) {
        vin[i] = (float)i;
    }

    simple(vin, vout, elemCount);

    for (int i = 0; i < elemCount; ++i) {
        printf("[%d] elem: simple(%f) = %f\n", i, vin[i], vout[i]);
    }

    int iBuffer[elemCount];
    for (int i = 0; i < elemCount; ++i) {
        iBuffer[i] = i;
    }

    pointerTest(iBuffer, elemCount);

    for (int i = 0; i < elemCount; ++i)
    {
        if (iBuffer[i] != i + 1)
        {
            fprintf(stderr, "Error result @ %d: %d, and the correct value is: %d\n", i, iBuffer[i], i + 1);
            break;
        }
    }
    puts("Complete pointerTest!");

    int atomValue = 0;
    atomicTest(&atomValue);
    printf("atomValue = %d\n", atomValue);

    alignas(64) int buffer[256] = { 0 };

    uniformTest(buffer, 16, 2);
    printf("uniform test result specified as 2: ");
    for (int i = 0; i < 15; ++i) {
        printf("%d, ", buffer[i]);
    }
    printf("%d\n", buffer[15]);

    uniformTest(buffer, 16, 9);
    printf("uniform test result specified as 9: ");
    for (int i = 0; i < 15; ++i) {
        printf("%d, ", buffer[i]);
    }
    printf("%d\n", buffer[15]);

    indexTest(&buffer[0], &buffer[64], 20);
    printf("programIndex elements: ");
    for (int i = 0; i < 20; ++i) {
        printf("0x%02X  ", buffer[i]);
    }
    printf("\nprogramCount elements: ");
    for (int i = 0; i < 20; ++i) {
        printf("0x%02X  ", buffer[64 + i]);
    }
    puts("");

    memset(buffer, 0, sizeof(buffer));

    multiTaskTest(buffer, elemCount* elemCount - 6);

    for (int i = 0; i < elemCount * elemCount - 6; ++i)
    {
        if (buffer[i] != i)
        {
            fprintf(stderr, "buffer[%d] error! result = %d, correct value is: %d\n", i, buffer[i], i);
            break;
        }
    }
    for (int i = elemCount * elemCount - 6; i < elemCount * elemCount; ++i)
    {
        if (buffer[i] != 0)
        {
            fprintf(stderr, "buffer[%d] error! result = %d, correct value is: 0\n", i, buffer[i]);
            break;
        }
    }
    puts("\n======== Complete Multi-Task Test!========\n");

    memset(iBuffer, 0, sizeof(iBuffer));
    atomValue = 0;
    atomicAddInc(iBuffer, &atomValue, elemCount);

    memset(iBuffer, 0, sizeof(iBuffer));
    atomValue = 0;
    atomicAddCASInc(iBuffer, &atomValue, elemCount);

    memset(buffer, 0, sizeof(buffer));

    int sum = 0;
    atomValue = 0;
    for (int i = 0; i < 20; ++i)
    {
        buffer[i] = i;
        sum += i;
    }

    subgroupAtomicAddIntTest(buffer, &atomValue, 20);

    if (atomValue != sum) {
        fprintf(stderr, "subgroupAtomicAddIntTest error: %d, correct value: %d\n", atomValue, sum);
    }
    else
    {
        sum = 0;
        for (int i = 0; i < 20; ++i)
        {
            if (buffer[i] != sum)
            {
                fprintf(stderr, "subgroupAtomicAddIntTest error: %d, correct value: %d\n", buffer[i], sum);
                break;
            }
            sum += i;
        }
    }

    float floatBuffer[32] = { 0.0f };
    float atomFloat = 0.0f;
    float sumFloat = 0.0f;

    for (int i = 0; i < 20; ++i)
    {
        floatBuffer[i] = (float)i;
        sumFloat += (float)i;
    }

    subgroupAtomicAddFloatTest(floatBuffer, &atomFloat, 20);

    if (atomFloat != sumFloat) {
        fprintf(stderr, "subgroupAtomicAddFloatTest error: %f, correct value: %f\n", atomFloat, sumFloat);
    }
    else
    {
        sumFloat = 0.0f;
        for (int i = 0; i < 20; ++i)
        {
            if (floatBuffer[i] != sumFloat)
            {
                fprintf(stderr, "subgroupAtomicAddFloatTest error: %f, correct value: %f\n", floatBuffer[i], sumFloat);
                break;
            }
            sumFloat += (float)i;
        }
    }

    for (int i = 0; i < 20; ++i) {
        buffer[i] = i ^ 3;
    }
    atomValue = 15;

    subgroupAtomicMaxTest(buffer, &atomValue, 20);

    if (atomValue != 19) {
        fprintf(stderr, "subgroupAtomicMaxTest error: %d, correct value: 19\n", atomValue);
    }

    sum = 0;
    atomValue = 0;
    for (int i = 0; i < 20; ++i)
    {
        buffer[i] = 1 << i;
        sum |= buffer[i];
    }

    subgroupAtomicOrTest(buffer, &atomValue, 20);

    if (atomValue != sum) {
        fprintf(stderr, "subgroupAtomicOrTest error: 0x%08x, correct value: 0x%08x\n", atomValue, sum);
    }
    else
    {
        sum = 0;
        for (int i = 0; i < 20; ++i)
        {
            if (buffer[i] != sum)
            {
                fprintf(stderr, "subgroupAtomicOrTest buffer error: 0x%08x, correct value: 0x%08x\n", buffer[i], sum);
                break;
            }

            sum |= 1 << i;
        }
    }

    atomValue = 100;
    for (int i = 0; i < 20; ++i) {
        buffer[i] = i;
    }

    subgroupAtomicSwapTest(buffer, &atomValue, 20);

    if (atomValue != 19 || buffer[0] != 100) {
        fprintf(stderr, "subgroupAtomicSwapTest error: %d, %d, correct value: 19, 100\n", atomValue, buffer[0]);
    }
    else
    {
        for (int i = 1; i < 20; ++i)
        {
            if (buffer[i] != i - 1)
            {
                fprintf(stderr, "subgroupAtomicSwapTest buffer error: %d, correct value: %d\n", buffer[i], i - 1);
                break;
            }
        }
    }

    for (size_t i = 20U; i < sizeof(buffer) / sizeof(buffer[0]); ++i)
    {
        if (buffer[i] != 0)
        {
            fprintf(stderr, "Error occurred @%zu: %d, correct value: 0\n", i, buffer[i]);
            break;
        }
    }

    coherentTest(buffer, 8, 8);

    void* totalBuffer = calloc(1024U * 1024U, 64U);
    void* alignedBuffer = (void*)(((uintptr_t)totalBuffer + 63U) & ~63ULL);

    VectorAddTest(alignedBuffer);

    long long longBuffer[32] = { 0LL };
    memset(buffer, 0, sizeof(buffer));
    allocaTest(buffer, longBuffer, 16);

    int sumValuePair[2] = { 0 };
    for (int i = 0; i < 16; ++i) {
        sumValuePair[i / 8] += i;
    }

    long long sum64Values[16] = { 0LL };
    for (int index = 0; index < 16; ++index)
    {
        const int count = index + 8;
        for (int i = 0; i < count; ++i) {
            sum64Values[index] += i;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        if (buffer[i] != sumValuePair[i / 8])
        {
            fprintf(stderr, "allocaTest shared buffer test failed @%d: %d, correct value: %d\n", i, buffer[i], sumValuePair[i / 8]);
            break;
        }
        if (longBuffer[i] != sum64Values[i])
        {
            fprintf(stderr, "allocaTest private buffer test failed @%d: %lld, correct value: %lld\n", i, longBuffer[i], sum64Values[i]);
            break;
        }
    }

    free(totalBuffer);

    dotProductWithCUDA();
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件

