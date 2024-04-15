#pragma once

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdalign.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#define _USE_MATH_DEFINES
#include <math.h>

#ifdef _WIN32
#include <Windows.h>

static inline void GetCommandLineInputString(char dstInput[], size_t maxBufferSize)
{
    gets_s(dstInput, maxBufferSize);
}

static inline FILE* OpenFileWithRead(const char* filePath)
{
    FILE* fp = NULL;
    const errno_t ret = fopen_s(&fp, filePath, "rb");
    if (ret != 0)
    {
        if (fp != NULL)
        {
            fclose(fp);
            fp = NULL;
        }
    }

    return fp;
}

static inline FILE* OpenFileWithWrite(const char* filePath)
{
    FILE* fp = NULL;
    const errno_t ret = fopen_s(&fp, filePath, "wb");
    if (ret != 0)
    {
        if (fp != NULL)
        {
            fclose(fp);
            fp = NULL;
        }
    }

    return fp;
}

static inline size_t GetCurrentExecutablePath(char dstPath[], size_t maxPathSize)
{
    return GetModuleFileNameA(NULL, dstPath, (DWORD)maxPathSize);
}

static inline void NewDirectory(const char dirPath[])
{
    CreateDirectoryA(dirPath, NULL);
}
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>

#define sprintf_s(buffer, bufferMaxCount, format, ...)      sprintf((buffer), (format), ## __VA_ARGS__)

static inline int strcpy_s(char* dst, size_t maxSizeInDst, const char* src)
{
    strcpy(dst, src);
    return 0;
}

static inline int strcat_s(char* dst, size_t maxSizeInDst, const char* src)
{
    strcat(dst, src);
    return 0;
}

static inline void GetCommandLineInputString(char dstInput[], size_t maxBufferSize)
{
    char* contents = NULL;
    size_t initLen = 0;
    ssize_t dstLen = getline(&contents, &initLen, stdin);
    if (dstLen < 1)
    {
        dstInput[0] = '\0';
        return;
    }

    // The last character is a new-line character, so remove it!
    if (--dstLen > (ssize_t)(maxBufferSize - 1)) {
        dstLen = maxBufferSize - 1;
    }

    memcpy(dstInput, contents, dstLen);
    dstInput[dstLen] = '\0';
}

static inline FILE* OpenFileWithRead(const char* filePath)
{
    FILE* const fp = fopen(filePath, "r");
    return fp;
}

static inline FILE* OpenFileWithWrite(const char* filePath)
{
    FILE* const fp = fopen(filePath, "w");
    return fp;
}

#ifdef __APPLE__
static inline size_t GetCurrentExecutablePath(char dstPath[], size_t maxPathSize)
{
    unsigned pathSize = 0;
    _NSGetExecutablePath(NULL, &pathSize);
    if (pathSize > (unsigned)maxPathSize) {
        pathSize = (unsigned)maxPathSize;
    }

    _NSGetExecutablePath(dstPath, &pathSize);
    dstPath[pathSize] = '\0';
    return pathSize;
}

extern void NewDirectory(const char dirPath[]);
#else
// Other Unix-like platforms

static inline size_t GetCurrentExecutablePath(char dstPath[], size_t maxPathSize)
{
    size_t pathSize = readlink("/proc/self/exe", dstPath, maxPathSize);
    if (pathSize > maxPathSize) {
        pathSize = maxPathSize;
    }
    dstPath[pathSize] = '\0';
    return pathSize;
}

static inline void NewDirectory(const char dirPath[])
{
    struct stat st = { 0 };

    if (stat(dirPath, &st) == -1) {
        mkdir(dirPath, 0777);
    }
}
#endif // __APPLE__

#endif // _WIN32

#ifndef MAX_PATH
#define MAX_PATH       512
#endif // !MAX_PATH

// Get timestamp in nanoseconds
static inline long long GetTimestampDuration(const struct timespec* pTimestamp)
{
    return pTimestamp->tv_sec * 1000000000LL + pTimestamp->tv_nsec;
}

// Fetch min, avg, max durations (in milliseconds) among begin timestamps and end timestamps.
// @return min duration
static inline double FetchDurationsAmongTimestamps(int nTimestamps, const long long beginTimestamps[], const long long endTimestamps[], double* pDstAvgDuration, double* pDstMaxDuration)
{
    long long sumDurations = 0LL;
    long long minDuration = INT64_MAX;
    long long maxDuration = 0LL;

    for (int i = 0; i < nTimestamps; ++i)
    {
        const long long currDuration = endTimestamps[i] - beginTimestamps[i];
        sumDurations += currDuration;
        if (minDuration > currDuration) {
            minDuration = currDuration;
        }
        if (maxDuration < currDuration) {
            maxDuration = currDuration;
        }
    }

    sumDurations -= minDuration + maxDuration;
    nTimestamps -= 2;
    if (nTimestamps < 1) {
        nTimestamps = 1;
    }

    if (pDstAvgDuration != NULL) {
        *pDstAvgDuration = (double)sumDurations / (double)nTimestamps / 1000000.0;
    }
    if (pDstMaxDuration != NULL) {
        *pDstMaxDuration = (double)maxDuration / 1000000.0;
    }

    return (double)minDuration / 1000000.0;
}

