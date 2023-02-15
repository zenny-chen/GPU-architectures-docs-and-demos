# About GPU architectures docs and demos
各大GPU厂商以及平台商关于3D图形渲染的demo

<br />

- [官方Vulkan编程指南](http://www.vulkanprogrammingguide.com)
- [nVidia的图形样例（含Vulkan API）](https://github.com/NVIDIAGameWorks/GraphicsSamples/tree/master/samples)
- [Android Vulkan 图形 API](https://developer.android.google.cn/ndk/guides/graphics)
- [Android Vulkan core topics](https://source.android.google.cn/devices/graphics/arch-vulkan?hl=en)
- [Google官方推荐的用于Android NDK的Vulkan API使用样例](https://github.com/LunarG/VulkanSamples)
- [Google官方的用于Android端的Vulkan API使用样例](https://github.com/googlesamples/android-vulkan-tutorials)
- [Vulkan C++ examples and demos](https://github.com/SaschaWillems/Vulkan)
- [Intel技术大牛详解Vulkan API](https://github.com/GameTechDev/IntroductionToVulkan)
- [API without Secrets: Introduction to Vulkan](https://github.com/GameTechDev/IntroductionToVulkan)
- [Vulkan Tutorial](https://vulkan-tutorial.com/Introduction)
- [Vulkan Cookbook 第一章 3 连接Vulkan Loader库](https://blog.csdn.net/qq_19473837/article/details/83056962)
- [VkDeviceCreateInfo](https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#VkDeviceCreateInfo)
- [Vulkan features（涉及如何允许逻辑设备全都允许、全都禁用或部分允许特征的方法）](https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#features)
- [Vulkan Querying and Enabling Extensions](https://github.com/KhronosGroup/Vulkan-Guide/blob/master/chapters/enabling_extensions.adoc#enabling-extensions)
- [Vulkan Timeline Semaphores](https://www.khronos.org/blog/vulkan-timeline-semaphores)
- [Vulkan Subgroup Tutorial](https://www.khronos.org/blog/vulkan-subgroup-tutorial)
- [VkPhysicalDeviceShaderFloat16Int8Features](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceFloat16Int8FeaturesKHR.html)
- [VK_EXT_robustness2](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_robustness2.html)
- [VK_EXT_conservative_rasterization](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_conservative_rasterization.html)
- [Translate GLSL to SPIR-V for Vulkan at Runtime](https://lxjk.github.io/2020/03/10/Translate-GLSL-to-SPIRV-for-Vulkan-at-Runtime.html)
- [How to Install LunarG Vulkan SDK for Ubuntu](https://support.amd.com/en-us/kb-articles/Pages/Install-LunarG-Vulkan-SDK.aspx)
- [Layers Overview and Configuration](https://vulkan.lunarg.com/doc/view/1.2.131.1/windows/layer_configuration.html)
- Vulkan中对`VK_INSTANCE_LAYERS`环境变量的设置：（Windows: `set VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump;VK_LAYER_KHRONOS_validation`；Linux：`export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump:VK_LAYER_KHRONOS_validation`）
- [Vulkan GFXReconstruct API Capture and Replay](https://vulkan.lunarg.com/doc/view/1.2.141.0/windows/capture_tools.html)
- [Vulkan官方SDK下载](https://www.vulkan.org/tools#download-these-essential-development-tools)
- 查看Vulkan设备信息：运行官方SDK目录下的Bin目录下的 **`vulkaninfoSDK`**
- 将GLSL源文件编译为spv文件可以用VulkanSDK自带的 **`glslangValidator`** 工具。具体用法比如：**`%VK_SDK_PATH%/Bin/glslangValidator  --target-env vulkan1.1  -o texturingKernel.spv  texturingKernel.comp`**。这里需要注意的是， **`glslangValidator`** 工具是根据文件后缀名来判定当前所要编译的GLSL属于哪种类型的shader，所以这里不能使用通用的 **`.glsl`** 文件后缀名。但是，我们可以对shader文件使用shader类型名后面再跟 **`.glsl`** 的方式让 **`glslangValidator`** 工具做shader类型识别。比如我们要编译一个vertex shader文件，可以将文件名命名为：**`flatten.vert.glsl`**；然后使用命令行 **`%VK_SDK_PATH%/Bin/glslangValidator  --target-env vulkan1.1  -o flatten.vert.spv  flatten.vert.glsl`** 进行编译。
- Windows系统端要想将glsl文件编译为spv文件，还可以使用Vulkan SDK自带的 **`glslc`**。具体用法比如：**`%VK_SDK_PATH%/Bin/glslc.exe  -fshader-stage=compute  -o simpleKernel.spv  simpleKernel.glsl`**。它能指定当前要编译的GLSL源文件属于哪种shader类型，因此文件后缀名基本可以随意定义。
- 将SPIR-V可读性的汇编转为SPIR-V字节码文件（spv文件）：**`spirv-as`**。具体用法比如：**`%VK_SDK_PATH%/Bin/spirv-as  -o simpleKernel.spv  simpleKernel.spvasm`**。
- 将spv字节码反汇编为可读的SPIR-V的格式，使用 **`spirv-dis`**。具体用法比如：**`%VK_SDK_PATH%/Bin/spirv-dis simpleKernel.spv  -o simpleKernel.spvasm`**。
- 将spv字节码反编译为GLSL：使用 **`spirv-cross`**。具体用法比如：**`%VK_SDK_PATH%/Bin/spirv-cross  --vulkan-semantics  --output dst.glsl  src.spv`**
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
- [Metal API介绍](https://developer.apple.com/metal/)
- [基于macOS的OpenGL的使用](https://developer.apple.com/opengl/)
- [基于iOS与tvOS的OpenGL ES的使用](https://developer.apple.com/opengl-es/)
- [Tutorial 12 : OpenGL Extensions](http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-12-opengl-extensions/)
- [Google官方的Android上使用JNI OpenGL ES 2.0的样例](https://github.com/googlesamples/android-ndk/tree/master/hello-gl2)
- [Google官方的Android上使用JNI OpenGL ES 3.1的样例](https://github.com/googlesamples/android-ndk/tree/master/gles3jni)
- [Learn OpenGL ES 2.0 on Android in Java](http://www.learnopengles.com/android-lesson-one-getting-started/)
- [OpenGL Programming/Installation/Linux](https://en.wikibooks.org/wiki/OpenGL_Programming/Installation/Linux)
- [How to install OpenGL/GLUT libraries for Ubuntu](https://askubuntu.com/questions/96087/how-to-install-opengl-glut-libraries)
- [FreeGLUT API](http://freeglut.sourceforge.net/docs/api.php)
- [OpenGL on Windows](https://docs.microsoft.com/zh-cn/windows/win32/opengl/opengl)
- [OpenGL Win32 Tutorial Sample Code](https://www.opengl.org/archives/resources/code/samples/win32_tutorial/)
- [Using OpenGL on Windows: A Simple Example](https://www.cs.rit.edu/~ncs/Courses/570/UserGuide/OpenGLonWin-11.html)
- [Using OpenGL with GTK+](https://www.bassi.io/articles/2015/02/17/using-opengl-with-gtk/)
- [How do I get EGL and OpenGLES libraries for Ubuntu](https://askubuntu.com/questions/244133/how-do-i-get-egl-and-opengles-libraries-for-ubuntu-running-on-virtualbox)
- Raspberry Pi comes with an OpenGL ES 2.0 example in `/opt/vc/src/hello_pi/hello_triangle2`
- [EGL guide for beginners](https://stackoverflow.com/questions/19212145/egl-guide-for-beginners)
- **Khronos OpenGL官方文档中关于描述的 `gbufferImage` 类型，实际所对应的类型为**：**`imageBuffer`**、**`iimageBuffer`** 和 **`uimageBuffer`**。
- [WebGL官方样例](https://github.com/WebGLSamples)
- [WebGL_Compute_shader](https://github.com/9ballsyndrome/WebGL_Compute_shader)
- [ROCm™ – 用于加速计算、支持高性能计算和机器学习的开放式软件生态系统](https://mp.weixin.qq.com/s?__biz=MjM5NDAyNjM0MA==&mid=2650787282&idx=8&sn=baa3373e1fa3b2564f223d5dc0dc9ca1&chksm=be856bd989f2e2cf954d48303447124992714e2b531448304d32da7b957e810203c0c46aacd9&mpshare=1&scene=23&srcid=0831IhBlly11evtjQ0cYgzs3&sharer_sharetime=1598879436131&sharer_shareid=35ac76bf9ad4a719bab0994dd606caf6#rd)
- [Introduction to Turing Mesh Shaders](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/)
- [Quick Introduction to Mesh Shaders (OpenGL and Vulkan)](https://www.geeks3d.com/20200519/introduction-to-mesh-shaders-opengl-and-vulkan/)
- [关于Drawcall](https://zhuanlan.zhihu.com/p/364918045)
- [GPU渲染架构-IMR \& TBR \& TBDR](https://zhuanlan.zhihu.com/p/531900597)
- [深度剖析：深度学习GPU共享技术](https://www.toutiao.com/i6906743227399881228/)
- [NVIDIA TURING GPU ARCHITECTURE](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [GeForce RTX 4090天梯榜首发评测“践踏”摩尔定律，开启未来游戏画卷](https://mp.weixin.qq.com/s?__biz=MjM5NDMxNjkyNA==&mid=2651537889&idx=1&sn=92c94d45f93dae6f49bf5f0d82da6217&chksm=bd7676f28a01ffe46456f0fa7490b24083a63a8621b75d9c7bfc3c304a311ec7ecca457a9d5f&mpshare=1&scene=23&srcid=1011y2IeQqo4hL49pBpni0DM)
- [ROPs and TMUs What is it?](https://www.techpowerup.com/forums/threads/rops-and-tmus-what-is-it.227596/)
- [Nvidia RT Cores vs. AMD Ray Accelerators – Explained](https://appuals.com/nvidia-rt-cores-vs-amd-ray-accelerators-explained/)
- [OpenSL ES播放PCM音频](https://www.toutiao.com/article/7153610070319071758/)

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

## 针对Vulkan API的一些常用且必要的GLSL扩展

- [GL_KHR_vulkan_glsl](https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_vulkan_glsl.txt)
- [GL_ARB_gpu_shader5](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_gpu_shader5.txt)（包含 **precise** 限定符）
- [GL_EXT_scalar_block_layout](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_scalar_block_layout.txt)（此扩展需要支持 [VK_EXT_scalar_block_layout](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_scalar_block_layout.html) 这一Vulkan扩展）
- [GL_EXT_shader_16bit_storage](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_16bit_storage.txt)（此扩展需要支持 [VK_KHR_16bit_storage](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_16bit_storage.html)  这一Vulkan扩展）
- [GL_EXT_shader_explicit_arithmetic_types](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt)（此扩展需要支持 [VK_KHR_shader_float16_int8](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_float16_int8.html) 这一Vulkan扩展）
- [GL_EXT_shader_atomic_int64](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_atomic_int64.txt)
- [GL_ARB_gpu_shader_int64](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_gpu_shader_int64.txt)
- [GL_EXT_shader_atomic_float](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_shader_atomic_float.txt)
- [GL_ARB_shader_clock](https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_clock.txt)
- [GL_EXT_shader_realtime_clock](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_realtime_clock.txt)
- [GL_EXT_demote_to_helper_invocation](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_demote_to_helper_invocation.txt)

<br />

## GLSL中的一些内建函数用法

- 将浮点数转为IEEE整数：[floatBitsToInt](https://registry.khronos.org/OpenGL-Refpages/gl4/html/floatBitsToInt.xhtml)
- 将IEEE规格化浮点的整数转为浮点数：[intBitsToFloat](https://registry.khronos.org/OpenGL-Refpages/gl4/html/intBitsToFloat.xhtml)
- 从一个整数中获取指定位置与长度的比特值：[bitfieldExtract](https://registry.khronos.org/OpenGL-Refpages/gl4/html/bitfieldExtract.xhtml)
- 对一个整数插入指定位置与长度的比特：[bitfieldInsert](https://registry.khronos.org/OpenGL-Refpages/gl4/html/bitfieldInsert.xhtml)
- 对一个整数指定位置与长度的比特进行取反：[bitfieldReverse](https://registry.khronos.org/OpenGL-Refpages/gl4/html/bitfieldReverse.xhtml)

<br />

## CUDA相关文档

- [CUDA Compute Capability List](https://developer.nvidia.com/cuda-gpus)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [CUDA 11 Features Revealed](https://developer.nvidia.com/blog/cuda-11-features-revealed/)
- [nVidia omniverse](https://www.nvidia.com/en-us/omniverse/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#abstract)
- [NVIDIA GPUDirect Storage Overview Guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [NVIDIA OptiX™ Ray Tracing Engine](https://developer.nvidia.com/optix)
- [How to Get Started with OptiX 7](https://developer.nvidia.com/blog/how-to-get-started-with-optix-7/)
- [NVIDIA Omniverse™ Platform](https://developer.nvidia.com/nvidia-omniverse-platform)
- [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface)（nvidia-smi）
- [NVIDIA CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- Windows上查看CUDA程序崩溃信息使用Nsight，具体可见：[8. GPU Core Dump Files](https://docs.nvidia.com/nsight-visual-studio-edition/cuda-inspect-state/index.html#gpu-core-dump)。Linux上则使用 **cuda-gdb** 来查看core dump文件信息。要使CUDA程序崩溃时导出core dump文件，需要先开启CUDA程序调试信息（`-g`），然后设置环境变量：`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`。
- [CUDA: Common Function for both Host and Device Code](https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/)
- [CUDA common **helper** functions](https://github.com/NVIDIA/cuda-samples/tree/master/Common)
- CUDA中关于整型数据的intrinsic函数的介绍在《CUDA_Math_API》文档中。
- [cuda 函数前缀 __host__ __device__ __global__ ____noinline__ 和 __forceinline__ 简介](https://blog.csdn.net/zdlnlhmj/article/details/104896470)
- [The Aggregate Magic Algorithms](http://aggregate.org/MAGIC/)
- [How to set cache configuration in CUDA](https://codeyarns.com/2011/06/27/how-to-set-cache-configuration-in-cuda/)
- [Preview support for alloca](https://developer.nvidia.com/blog/programming-efficiently-with-the-cuda-11-3-compiler-toolchain/)
- [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://devblogs.nvidia.com/how-access-global-memory-efficiently-cuda-c-kernels/)
- [CUDA \#pragma unroll](https://blog.csdn.net/nothinglefttosay/article/details/44725497)
- CUDA中获取显存总的大小及可用显存大小：`cudaError_t cudaMemGetInfo(size_t *free,  size_t *total);`。
- [CUDA编程优化（存储器访问优化，指令优化，参数优化，）](https://yq.aliyun.com/articles/513120?spm=5176.10695662.1996646101.searchclickresult.7ab377c9OTv8ug)
- [CUDA constant memory issue: invalid device symbol with cudaGetSymbolAddress](https://stackoverflow.com/questions/26735808/cuda-constant-memory-issue-invalid-device-symbol-with-cudagetsymboladdress)
- [Unified Memory for CUDA Beginners](https://devblogs.nvidia.com/unified-memory-cuda-beginners/)
- [CUDA - Unified memory (Pascal at least)](https://stackoverflow.com/questions/50679657/cuda-unified-memory-pascal-at-least)
- [为什么不能使用`cudaHostRegister（）`来标记为WriteCombined已经存在的内存区域？](https://www.it1352.com/587955.html)
- [How to Optimize Data Transfers in CUDA C/C++](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/)
- [“Pitch” in cudaMallocPitch()?](https://forums.developer.nvidia.com/t/pitch-in-cudamallocpitch/8065)（**Pitch** is the padded size of each “row” in the array. If you have an array that has 12 float rows, CUDA runs faster if you pad the data to 16 floats: The data is 12 floats wide, the padding is 4 floats, and the **pitch** is 16 floats. (Or 64 bytes, as **`cudaMallocPitch`** sees it.)）
- [CUDA学习-计算实际线程ID](https://blog.csdn.net/weixin_51229250/article/details/121712045)
- [CUDA总结：纹理内存](https://blog.csdn.net/kelvin_yan/article/details/54019017)
- [\[CUDA\]纹理对象 Texture Object](https://blog.csdn.net/m0_38068229/article/details/89478981)
- [Textures & Surfaces](https://developer.download.nvidia.cn/CUDA/training/texture_webinar_aug_2011.pdf)
- [CUDA学习笔记：Texture与Surface](https://zhuanlan.zhihu.com/p/414956511)（**Surface的性质**：Surface与Texture不同在于，Surface是可读且可写的。）
- [CUDA获取时间函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function)
- [CUDA编程入门----Thrust库简介](https://blog.csdn.net/he_wolf/article/details/23502793)（文章最后有NV官方文档介绍）
- [Thrust: sort_by_key slow due to memory allocation](https://stackoverflow.com/questions/6605498/thrust-sort-by-key-slow-due-to-memory-allocation)
- [矩阵相乘在GPU上的终极优化：深度解析Maxas汇编器工作原理](https://www.toutiao.com/a6824717488391979532/)
- [如何利用 NVIDIA 安培架构 GPU 的新一代 Tensor Core 对计算进行极致加速](https://mp.weixin.qq.com/s?srcid=0819SLpl2nBtb62AroWYJnmI&scene=23&sharer_sharetime=1597814989936&mid=2247486054&sharer_shareid=c0f8ad645f1b221a7a43ae65e09fb2ea&sn=e0507022e05c91857cce22195a504323&idx=1&__biz=MzU3MDc5MDM4MA%3D%3D&chksm=fceb52e5cb9cdbf34ab0b820f58283e4f430a140fd35d1746e1d4da25ed2205cc3f7320e909c&mpshare=1#rd)
- [NVIDIA发布了那么多技术，哪些SDK值得开发者们关注？](https://mp.weixin.qq.com/s?__biz=MjM5NTE3Nzk4MQ==&mid=2651239720&idx=1&sn=ee3c8a6c831e9ce525994d94818d4ad4&chksm=bd0e61ba8a79e8ac42b009d693f52fe678ab34aeea4243f215c272ff32cc1b264b409180a7f3&mpshare=1&scene=23&srcid=0421vjj8MN30lK26ZUxC1zUH)
- CUDA中用于判定当前是否为设备端代码还是主机端代码使用预定义宏 **`__CUDA_ARCH__`** 。它同时表示当前设备的compute capability，比如200表示计算能力2.0。详细参考《CUDA C Programming Guide》G.4.2.1. **`__CUDA_ARCH__`** 。
- 由于CUDA NVCC编译器是寄生于MSVC或GCC等主机端编译工具链的。因此，如果我们在一个CUDA源文件（.cu）中要判定当前仅适用于CUDA源文件的主机端与设备端代码，那么我们要写预处理器条件判断时需要包含或排除掉CUDA环境。而 **`__CUDACC__`** 这个宏就是NVCC编译器内置的宏，用于判定当前的编译器用的是NVCC。而此时，如果我们用 **`_MSC_VER`** 或 **`__GNUC__`** 宏来判定的话，条件也成立。因此，如果我们对某些处理需要针对仅使用某种主机端的编译器（比如GCC）而不适用NVCC的话可以这么判定：
```c
#if defined(__GNUC__) && !defined(__CUDACC__)
// Only work for GCC, but not for NVCC
#endif
```
- CUDA编译选项`--compiler-options`的作用是可指定当前系统编译环境的额外编译选项。比如：`-Xcompiler=/EHsc,-Ob2,/wd4819`。也可以写作为：`-Xcompiler="/EHsc,-Ob2,/wd4819"`，`-Xcompiler /EHsc,-Ob2,/wd4819`，或是：`-Xcompiler "/EHsc,-Ob2,/wd4819"`。

- CUDA编译时指定单个架构：比如就指定使用真实架构SM7.5：`-arch=sm_75`。
- CUDA编译时指定多个架构：将虚拟架构（比如：Compute 7.5）与真实架构（比如：SM 7.5）进行结合，然后声明多个架构：`-gencode=arch=compute_52,code=sm_52  -gencode=arch=compute_60,code=sm_60`。

- 在CUDA编译选项中有一个 **-rdc**，意思是 *Generate Relocatable Device Code*。该选项默认是关闭的，即`-rdc=false`，在此情况下，每个cuda源文件只能包含自己的全局`__device__`和`__constant__`对象，而不能引用其他cuda源文件中所定义的全局对象，同时，即便在同一cuda源文件，一个全局对象也不能声明，因为声明了它就等于定义了它，再对它定义加初始化就会出现重复定义的错误。而在将它打开的情况下，即`-rdc=true`，那么全局对象的行为就跟普通C语言全局对象的行为一样了，在一个模块中，可以跨多个cuda源文件对同一全局对象引用，同时也能做定义前的声明。因此通常情况下，我们应该考虑将此编译选项打开。

- [How to use the static option with g ++ used by nvcc?](https://forums.developer.nvidia.com/t/how-to-use-the-static-option-with-g-used-by-nvcc/55787)  这里重要的是最底下的评论。

- cudaMemcpy probably isn't actually taking that long--that will synchronize and wait for the kernel to complete. Launching a kernel is (almost) always asynchronous; when you call kernel<<<...>>>(...);, it's actually just queuing work for the GPU to perform at some point. It won't block the CPU and wait for that kernel to finish or anything like that. **However, since cudaMemcpy is a synchronous function, it implies that you want the results to be visible, so that will block the CPU until the GPU becomes idle** (indicating that all of your work has completed).

- **How to make it explicit that I am not using shared memory?** -- In Volta the L1 cache, texture cache, and shared memory are backed by a combined 128 KB data cache. As in previous architectures, such as Kepler, the portion of the cache dedicated to shared memory (known as the carveout) can be selected at runtime using cudaFuncSetAttribute() with the attribute cudaFuncAttributePreferredSharedMemoryCarveout. Volta supports shared memory capacities of 0, 8, 16, 32, 64, or 96 KB per SM. **You need to explicitly set shared memory capacity to 0.**

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

