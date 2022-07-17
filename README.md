# About GPU architectures docs and demos
各大GPU厂商以及平台商关于3D图形渲染的demo

<br />

- [官方Vulkan编程指南](http://www.vulkanprogrammingguide.com)
- [nVidia的图形样例（含Vulkan API）](https://github.com/NVIDIAGameWorks/GraphicsSamples/tree/master/samples)
- [Google官方推荐的用于Android NDK的Vulkan API使用样例](https://github.com/LunarG/VulkanSamples)
- [Google官方的用于Android端的Vulkan API使用样例](https://github.com/googlesamples/android-vulkan-tutorials)
- [Vulkan C++ examples and demos](https://github.com/SaschaWillems/Vulkan)
- [Intel技术大牛详解Vulkan API](https://github.com/GameTechDev/IntroductionToVulkan)
- [没有任何秘密的 API：Vulkan*](https://software.intel.com/zh-cn/articles/api-without-secrets-introduction-to-vulkan-preface)
- [Vulkan Tutorial](https://vulkan-tutorial.com/Introduction)
- [Vulkan Cookbook 第一章 3 连接Vulkan Loader库](https://blog.csdn.net/qq_19473837/article/details/83056962)
- [A simple Vulkan Compute example](http://www.duskborn.com/posts/a-simple-vulkan-compute-example/)
- [\[vulkan\] compute shader](https://zhuanlan.zhihu.com/p/56106712)
- [VkDeviceCreateInfo](https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#VkDeviceCreateInfo)
- [Vulkan features（涉及如何允许逻辑设备全都允许、全都禁用或部分允许特征的方法）](https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#features)
- [Vulkan Querying and Enabling Extensions](https://github.com/KhronosGroup/Vulkan-Guide/blob/master/chapters/enabling_extensions.adoc#enabling-extensions)
- [Vulkan中的同步与缓存控制](https://zhuanlan.zhihu.com/p/161619652)
- [Vulkan Timeline Semaphores](https://www.khronos.org/blog/vulkan-timeline-semaphores)
- [VkPhysicalDeviceShaderFloat16Int8Features](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceFloat16Int8FeaturesKHR.html)
- [VK_EXT_robustness2](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_robustness2.html)
- [Vulkan Subgroup Tutorial](https://www.khronos.org/blog/vulkan-subgroup-tutorial)
- [从Vulkan API看Shader的数据绑定机制](https://zhuanlan.zhihu.com/p/111882744)
- [Vulkan推荐用法(三星设备)](https://zhuanlan.zhihu.com/p/97321638)
- [\[译\]Vulkan教程(23)暂存buffer](https://www.cnblogs.com/bitzhuwei/p/Vulkan-Tutoria-23-Staging-buffer.html)
- [一起学Vulkan图形开发](https://www.zhihu.com/column/chenyong2vulkan)
- [Translate GLSL to SPIR-V for Vulkan at Runtime](https://lxjk.github.io/2020/03/10/Translate-GLSL-to-SPIRV-for-Vulkan-at-Runtime.html)
- [How to Install LunarG Vulkan SDK for Ubuntu](https://support.amd.com/en-us/kb-articles/Pages/Install-LunarG-Vulkan-SDK.aspx)
- [Layers Overview and Configuration](https://vulkan.lunarg.com/doc/view/1.2.131.1/windows/layer_configuration.html)
- Vulkan中对`VK_INSTANCE_LAYERS`环境变量的设置：（Windows: `set VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump;VK_LAYER_KHRONOS_validation`；Linux：`export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump:VK_LAYER_KHRONOS_validation`）
- [Vulkan GFXReconstruct API Capture and Replay](https://vulkan.lunarg.com/doc/view/1.2.141.0/windows/capture_tools.html)
- [Vulkan官方SDK下载](https://www.vulkan.org/tools#download-these-essential-development-tools)
- 查看Vulkan设备信息：运行官方SDK目录下的Bin目录下的 **`vulkaninfoSDK`**
- [How to use a vulkan sampler with unnormalized texture-coordinates?](https://stackoverflow.com/questions/65790303/how-to-use-a-vulkan-sampler-with-unnormalized-texture-coordinates-without-trig)
- 将GLSL源文件编译为spv文件可以用VulkanSDK自带的 **`glslangValidator`** 工具。具体用法比如：**`%VK_SDK_PATH%/Bin/glslangValidator  --target-env vulkan1.1  -o texturingKernel.spv  texturingKernel.comp`**。这里需要注意的是， **`glslangValidator`** 工具是根据文件后缀名来判定当前所要编译的GLSL属于哪种类型的shader，所以这里不能使用通用的 **`.glsl`** 文件后缀名。
- Windows系统端要想将glsl文件编译为spv文件，还可以使用Vulkan SDK自带的 **`glslc`**。具体用法比如：**`%VK_SDK_PATH%/Bin/glslc.exe  -fshader-stage=compute  -o simpleKernel.spv  simpleKernel.glsl`**。它能指定当前要编译的GLSL源文件属于哪种shader类型，因此文件后缀名基本可以随意定义。
- 将SPIR-V可读性的汇编转为SPIR-V字节码文件（spv文件）：**`spirv-as`**。具体用法比如：**`%VK_SDK_PATH%/Bin/spirv-as  -o simpleKernel.spv  simpleKernel.spvasm`**。
- 将spv字节码反汇编为可读的SPIR-V的格式，使用 **`spirv-dis`**。具体用法比如：**`%VK_SDK_PATH%/Bin/spirv-dis simpleKernel.spv  -o simpleKernel.spvasm`**。
- [HLSL for Vulkan: Resources](https://antiagainst.github.io/post/hlsl-for-vulkan-resources/)
- [How to compile HLSL shaders with Vulkan?](https://stackoverflow.com/questions/61387495/how-to-compile-hlsl-shaders-with-vulkan)
- [Google官方OpenCL C转SPIR-V项目](https://github.com/google/clspv)
- [PowerVR Developer Documentation](https://docs.imgtec.com)
- [SIGGRAPH 2018上提供的Vulkan API使用demo](http://web.engr.oregonstate.edu/~mjb/vulkan/)
- [Microsoft基于D3D12的图形编程样例](https://github.com/Microsoft/DirectX-Graphics-Samples)
- [microsoft/Xbox-ATG-Samples](https://github.com/microsoft/Xbox-ATG-Samples)
- [nBody DirectX 12 sample (asynchronous compute version)](https://gpuopen.com/gaming-product/nbody-directx-12-async-compute-edition/)
- [Learning DirectX 12 – Lesson 4 – Textures](https://www.3dgep.com/learning-directx-12-4)
- [Direct3D 11.3 Functional Specification](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm)
- [nVidia关于最新D3D的样例](https://developer.nvidia.com/gameworks-directx-samples)
- [nVidia关于D3D11的样例](https://developer.nvidia.com/dx11-samples)
- [Introduction to 3D Game Programming With DirectX 11书中代码样例](https://github.com/jjuiddong/Introduction-to-3D-Game-Programming-With-DirectX11)
- [Introduction to 3D Game Programming with DirectX 12书中代码样例](https://github.com/d3dcoder/d3d12book/)
- [基于macOS的OpenGL的使用](https://developer.apple.com/opengl/)
- [基于iOS与tvOS的OpenGL ES的使用](https://developer.apple.com/opengl-es/)
- [Metal API介绍](https://developer.apple.com/metal/)
- [Google官方的Android上使用JNI OpenGL ES 2.0的样例](https://github.com/googlesamples/android-ndk/tree/master/hello-gl2)
- [Google官方的Android上使用JNI OpenGL ES 3.1的样例](https://github.com/googlesamples/android-ndk/tree/master/gles3jni)
- [Learn OpenGL ES 2.0 on Android in Java](http://www.learnopengles.com/android-lesson-one-getting-started/)
- [OpenGL Programming/Installation/Linux](https://en.wikibooks.org/wiki/OpenGL_Programming/Installation/Linux)
- [How to install OpenGL/GLUT libraries for Ubuntu](https://askubuntu.com/questions/96087/how-to-install-opengl-glut-libraries)
- [FreeGLUT API](http://freeglut.sourceforge.net/docs/api.php)
- [How do I get EGL and OpenGLES libraries for Ubuntu](https://askubuntu.com/questions/244133/how-do-i-get-egl-and-opengles-libraries-for-ubuntu-running-on-virtualbox)
- [Using OpenGL with GTK+](https://www.bassi.io/articles/2015/02/17/using-opengl-with-gtk/)
- Raspberry Pi comes with an OpenGL ES 2.0 example in `/opt/vc/src/hello_pi/hello_triangle2`
- [EGL guide for beginners](https://stackoverflow.com/questions/19212145/egl-guide-for-beginners)
- [WebGL官方样例](https://github.com/WebGLSamples)
- [WebGL_Compute_shader](https://github.com/9ballsyndrome/WebGL_Compute_shader)
- [OpenGL on Windows](https://docs.microsoft.com/zh-cn/windows/win32/opengl/opengl)
- [OpenGL Win32 Tutorial Sample Code](https://www.opengl.org/archives/resources/code/samples/win32_tutorial/)
- [Using OpenGL on Windows: A Simple Example](https://www.cs.rit.edu/~ncs/Courses/570/UserGuide/OpenGLonWin-11.html)
- [ROCm™ – 用于加速计算、支持高性能计算和机器学习的开放式软件生态系统](https://mp.weixin.qq.com/s?__biz=MjM5NDAyNjM0MA==&mid=2650787282&idx=8&sn=baa3373e1fa3b2564f223d5dc0dc9ca1&chksm=be856bd989f2e2cf954d48303447124992714e2b531448304d32da7b957e810203c0c46aacd9&mpshare=1&scene=23&srcid=0831IhBlly11evtjQ0cYgzs3&sharer_sharetime=1598879436131&sharer_shareid=35ac76bf9ad4a719bab0994dd606caf6#rd)
- [Introduction to Turing Mesh Shaders](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/)
- [Quick Introduction to Mesh Shaders (OpenGL and Vulkan)](https://www.geeks3d.com/20200519/introduction-to-mesh-shaders-opengl-and-vulkan/)
- [深度剖析：深度学习GPU共享技术](https://www.toutiao.com/i6906743227399881228/)
- [NVIDIA TURING GPU ARCHITECTURE](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
- [ROPs and TMUs What is it?](https://www.techpowerup.com/forums/threads/rops-and-tmus-what-is-it.227596/)
- [关于Drawcall](https://zhuanlan.zhihu.com/p/364918045)

<br/>

## GLSL源文件扩展名

文件后缀名 | 表示的着色器种类 | 着色器种类英文名
---- | ---- | ----
.vert | 顶点着色器 | vertex shader
.frag | 片段着色器 | fragment shader
.geom | 几何着色器 | geometry shader
.tesc | 细分曲面控制着色器 | tessellation control shader
.tese | 细分曲面求值着色器 | tessellation evaluation shader
.comp | 计算着色器 | compute shader
.mesh | 网格着色器 | mesh shader
.task | 任务着色器 | task shader
.rgen | 光线生成着色器 | ray generation shader
.rint | 光线求交着色器 | ray intersection shader
.rahit | 光线任一击中着色器 | ray any hit shader
.rchit | 光线最近命中着色器 | ray closest hit shader
.rmiss | 光线未命中着色器 | ray miss shader
.rcall | 光线可调用着色器 | ray callable shader
.glsl | 通用GLSL着色器文件 | OpenGL Shading Language


- 当前Xcode 10所能识别出的GLSL文件类别: `.glsl`、`.vsh`、`.fsh`、`.gsh`、`.vert`、`.frag`、`.geom`
- 当前Android Studio所能识别出的GLSL文件类别：`.glsl`、`.vsh`、`.fsh`、`.comp`、`.geom`、`.vert`、`.frag`、`.tesc`、`.tese`
- SPIR-V字节码文件扩展名：**`.spv`**；SPIR-V汇编文件扩展名：**`.spvasm`**。

<br />

## CUDA相关文档

- [CUDA Compute Capability List](https://developer.nvidia.com/zh-cn/cuda-gpus#compute)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [CUDA 11 Features Revealed](https://developer.nvidia.com/blog/cuda-11-features-revealed/)
- [NVIDIA GPUDirect Storage Overview Guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [NVIDIA OptiX™ Ray Tracing Engine](https://developer.nvidia.com/optix)
- [How to Get Started with OptiX 7](https://developer.nvidia.com/blog/how-to-get-started-with-optix-7/)
- [NVIDIA Omniverse™ Platform](https://developer.nvidia.com/nvidia-omniverse-platform)
- [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface)（nvidia-smi）
- Windows上查看CUDA程序崩溃信息使用Nsight，具体可见：[8. GPU Core Dump Files](https://docs.nvidia.com/nsight-visual-studio-edition/cuda-inspect-state/index.html#gpu-core-dump)。Linux上则使用 **cuda-gdb** 来查看core dump文件信息。要使CUDA程序崩溃时导出core dump文件，需要先开启CUDA程序调试信息（`-g`），然后设置环境变量：`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`。
- [CUDA: Common Function for both Host and Device Code](https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/)
- [CUDA common **helper** functions](https://github.com/NVIDIA/cuda-samples/tree/master/Common)
- CUDA中关于整型数据的intrinsic函数的介绍在《CUDA_Math_API》文档中。
- [cuda 函数前缀 __host__ __device__ __global__ ____noinline__ 和 __forceinline__ 简介](https://blog.csdn.net/zdlnlhmj/article/details/104896470)
- [The Aggregate Magic Algorithms](http://aggregate.org/MAGIC/)
- [How to set cache configuration in CUDA](https://codeyarns.com/2011/06/27/how-to-set-cache-configuration-in-cuda/)
- [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://devblogs.nvidia.com/how-access-global-memory-efficiently-cuda-c-kernels/)
- [CUDA \#pragma unroll](https://blog.csdn.net/nothinglefttosay/article/details/44725497)
- CUDA中获取显存总的大小及可用显存大小：`cudaError_t cudaMemGetInfo(size_t *free,  size_t *total);`。
- [CUDA编程优化（存储器访问优化，指令优化，参数优化，）](https://yq.aliyun.com/articles/513120?spm=5176.10695662.1996646101.searchclickresult.7ab377c9OTv8ug)
- [CUDA constant memory issue: invalid device symbol with cudaGetSymbolAddress](https://stackoverflow.com/questions/26735808/cuda-constant-memory-issue-invalid-device-symbol-with-cudagetsymboladdress)
- [Unified Memory for CUDA Beginners](https://devblogs.nvidia.com/unified-memory-cuda-beginners/)
- [CUDA - Unified memory (Pascal at least)](https://stackoverflow.com/questions/50679657/cuda-unified-memory-pascal-at-least)
- [为什么不能使用`cudaHostRegister（）`来标记为WriteCombined已经存在的内存区域？](https://www.it1352.com/587955.html)
- [How to Optimize Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/)
- [CUDA总结：纹理内存](https://blog.csdn.net/kelvin_yan/article/details/54019017)
- [\[CUDA\]纹理对象 Texture Object](https://blog.csdn.net/m0_38068229/article/details/89478981)
- [NVIDIA CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- [CUDA学习-计算实际线程ID](https://blog.csdn.net/weixin_51229250/article/details/121712045)
- [CUDA获取时间函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function)
- [Textures & Surfaces](https://developer.download.nvidia.cn/CUDA/training/texture_webinar_aug_2011.pdf)
- [CUDA编程入门----Thrust库简介](https://blog.csdn.net/he_wolf/article/details/23502793)（文章最后有NV官方文档介绍）
- [Thrust: sort_by_key slow due to memory allocation](https://stackoverflow.com/questions/6605498/thrust-sort-by-key-slow-due-to-memory-allocation)
- [矩阵相乘在GPU上的终极优化：深度解析Maxas汇编器工作原理](https://www.toutiao.com/a6824717488391979532/)
- [如何利用 NVIDIA 安培架构 GPU 的新一代 Tensor Core 对计算进行极致加速](https://mp.weixin.qq.com/s?srcid=0819SLpl2nBtb62AroWYJnmI&scene=23&sharer_sharetime=1597814989936&mid=2247486054&sharer_shareid=c0f8ad645f1b221a7a43ae65e09fb2ea&sn=e0507022e05c91857cce22195a504323&idx=1&__biz=MzU3MDc5MDM4MA%3D%3D&chksm=fceb52e5cb9cdbf34ab0b820f58283e4f430a140fd35d1746e1d4da25ed2205cc3f7320e909c&mpshare=1#rd)
- [nVidia omniverse](https://www.nvidia.com/en-us/omniverse/)
- [NVIDIA发布了那么多技术，哪些SDK值得开发者们关注？](https://mp.weixin.qq.com/s?__biz=MjM5NTE3Nzk4MQ==&mid=2651239720&idx=1&sn=ee3c8a6c831e9ce525994d94818d4ad4&chksm=bd0e61ba8a79e8ac42b009d693f52fe678ab34aeea4243f215c272ff32cc1b264b409180a7f3&mpshare=1&scene=23&srcid=0421vjj8MN30lK26ZUxC1zUH)
- CUDA中用于判定当前是否为设备端代码还是主机端代码使用预定义宏 **`__CUDA_ARCH__`** 。它同时表示当前设备的compute capability，比如200表示计算能力2.0。详细参考《CUDA C Programming Guide》G.4.2.1. **`__CUDA_ARCH__`** 。
- CUDA编译选项`--compiler-options`的作用是可指定当前系统编译环境的额外编译选项。比如：`--compiler-options=/EHsc,-Ob2,/wd4819`。也可以写作为：`--compiler-options="/EHsc,-Ob2,/wd4819"`，`--compiler-options /EHsc,-Ob2,/wd4819`，或是：`--compiler-options "/EHsc,-Ob2,/wd4819"`。

- CUDA编译指定多个架构：`-gencode=arch=compute_35,code="sm_35,compute_35"  -gencode=arch=compute_50,code="sm_50,compute_50"  -gencode=arch=compute_60,code="sm_60,compute_60"`

- 在CUDA编译选项中有一个 **-rdc**，意思是 *Generate Relocatable Device Code*。该选项默认是关闭的，即`-rdc=false`，在此情况下，每个cuda源文件只能包含自己的全局`__device__`和`__constant__`对象，而不能引用其他cuda源文件中所定义的全局对象，同时，即便在同一cuda源文件，一个全局对象也不能声明，因为声明了它就等于定义了它，再对它定义加初始化就会出现重复定义的错误。而在将它打开的情况下，即`-rdc=true`，那么全局对象的行为就跟普通C语言全局对象的行为一样了，在一个模块中，可以跨多个cuda源文件对同一全局对象引用，同时也能做定义前的声明。因此通常情况下，我们应该考虑将此编译选项打开。

- cudaMemcpy probably isn't actually taking that long--that will synchronize and wait for the kernel to complete. Launching a kernel is (almost) always asynchronous; when you call kernel<<<...>>>(...);, it's actually just queuing work for the GPU to perform at some point. It won't block the CPU and wait for that kernel to finish or anything like that. **However, since cudaMemcpy is a synchronous function, it implies that you want the results to be visible, so that will block the CPU until the GPU becomes idle** (indicating that all of your work has completed).

- **How to make it explicit that I am not using shared memory?** -- In Volta the L1 cache, texture cache, and shared memory are backed by a combined 128 KB data cache. As in previous architectures, such as Kepler, the portion of the cache dedicated to shared memory (known as the carveout) can be selected at runtime using cudaFuncSetAttribute() with the attribute cudaFuncAttributePreferredSharedMemoryCarveout. Volta supports shared memory capacities of 0, 8, 16, 32, 64, or 96 KB per SM. **You need to explicitly set shared memory capacity to 0.**

<br />

## 针对Vulkan API的一些常用且必要的GLSL扩展

- [GL_EXT_shader_16bit_storage](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_16bit_storage.txt)
- [EXT_shader_explicit_arithmetic_types](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt)
- [GL_EXT_shader_atomic_int64](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_atomic_int64.txt)
- [GL_ARB_gpu_shader_int64](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_gpu_shader_int64.txt)
- [GL_EXT_shader_atomic_float](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_shader_atomic_float.txt)
- [GL_ARB_shader_clock](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_clock.txt)
- [GL_EXT_shader_realtime_clock](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_realtime_clock.txt)

<br />

## CUDA样例程序（包含对 **`clock64()`** 函数的使用）

```cuda

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

static constexpr auto arraySize = 1152U;

static __global__ void addKernel(int c[], long long timeCosts[], const int a[], const int b[])
{
    auto const gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid >= arraySize) {
        return;
    }

    auto ticksBegin = clock64();

    c[gtid] = a[gtid] * a[gtid] + (b[gtid] - a[gtid]);

    timeCosts[gtid] = clock64() - ticksBegin;
}

static void AddWithCUDATest(void)
{
    puts("======== The following is Add-With-CUDA Test ========");

    int a[arraySize];
    int b[arraySize];
    int c[arraySize] = {  };
    long long timeCosts[arraySize] = { };

    cudaFuncAttributes attrs{ };
    auto cudaStatus = cudaFuncGetAttributes(&attrs, addKernel);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaFuncGetAttributes call failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    auto const maxThreadCount = attrs.maxThreadsPerBlock;

    for (unsigned i = 0U; i < arraySize; ++i)
    {
        a[i] = i + 1;
        b[i] = a[i] * 10;
    }

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;
    long long* dev_ts = nullptr;

    do
    {
        cudaStatus = cudaMalloc(&dev_c, sizeof(c));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for dev_c: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&dev_a, sizeof(a));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for dev_a: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&dev_b, sizeof(b));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for dev_b: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMalloc(&dev_ts, sizeof(timeCosts));
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for dev_ts: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(dev_a, a, sizeof(a), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_a: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(dev_b, b, sizeof(b), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_a: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        auto const blockSize = (arraySize + maxThreadCount - 1) / maxThreadCount;

        // Launch a kernel on the GPU with one thread for each element.
        addKernel <<< blockSize, maxThreadCount >>> (dev_c, dev_ts, dev_a, dev_b);

        cudaStatus = cudaMemcpy(c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_c: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(c, dev_c, sizeof(c), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_c: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        cudaStatus = cudaMemcpy(timeCosts, dev_ts, sizeof(timeCosts), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMemcpy failed for dev_c: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);
        printf("ts[0, 1, -1, -2] = {%lld, %lld, %lld, %lld}\n", timeCosts[0], timeCosts[1], timeCosts[arraySize - 1], timeCosts[arraySize - 2]);
    }
    while (false);

    if (dev_a != nullptr) {
        cudaFree(dev_a);
    }
    if (dev_b != nullptr) {
        cudaFree(dev_b);
    }
    if (dev_c != nullptr) {
        cudaFree(dev_c);
    }
    if (dev_ts != nullptr) {
        cudaFree(dev_ts);
    }
}

int main(int argc, const char* argv[])
{
    auto cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaSetDevice call failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    AddWithCUDATest();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return 0;
}

```

<br />

