#£¡/bin/sh

# Build ISPC
ISPC_PATH="/usr/local/ispc-v1.22.0-linux/bin/ispc"
${ISPC_PATH}  simple.ispc  -o simple.o  -O2  --pic  --addressing=64 --arch=x86-64 --device=skylake --target=avx2-i32x8 --target-os=linux

# Build assembly
gcc test.S  -o test.o  -c

# Build main.c
gcc main.c  -o main.o  -std=gnu17  -c

# Build CUDA
CUDA_SDK_PATH="/usr/local/cuda"
CUDA_COMPILE_OPTION="-std=c++17 -m64 -maxrregcount=0  -gencode=arch=compute_75,code=sm_75  -gencode=arch=compute_86,code=sm_86 -cudart=static -cudadevrt=static -link -O2"
${CUDA_SDK_PATH}/bin/nvcc  kernel.cu  simple.o test.o main.o  -o ISPC_simple  ${CUDA_COMPILE_OPTION} -I${CUDA_SDK_PATH}/include/  -L${CUDA_SDK_PATH}/lib64/  -lm -lstdc++ -lpthread

# Remove object files
rm test.o main.o simple.o

