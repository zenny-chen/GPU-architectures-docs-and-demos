# About GPU architectures docs and demos
各大GPU厂商以及平台商关于3D图形渲染的demo

<br />

- [官方Vulkan编程指南](http://www.vulkanprogrammingguide.com)
- [nVidia的图形样例（含Vulkan API）](https://github.com/NVIDIAGameWorks/GraphicsSamples/tree/master/samples)
- [Google官方推荐的用于Android NDK的Vulkan API使用样例](https://github.com/LunarG/VulkanSamples)
- [Google官方的用于Android端的Vulkan API使用样例](https://github.com/googlesamples/android-vulkan-tutorials)
- [Intel技术大牛详解Vulkan API](https://github.com/GameTechDev/IntroductionToVulkan)
- [没有任何秘密的 API：Vulkan*](https://software.intel.com/zh-cn/articles/api-without-secrets-introduction-to-vulkan-preface)
- [A simple Vulkan Compute example](http://www.duskborn.com/posts/a-simple-vulkan-compute-example/)
- [How to Install LunarG Vulkan SDK for Ubuntu](https://support.amd.com/en-us/kb-articles/Pages/Install-LunarG-Vulkan-SDK.aspx)
- [PowerVR Developer Documentation](https://docs.imgtec.com)
- [SIGGRAPH 2018上提供的Vulkan API使用demo](http://web.engr.oregonstate.edu/~mjb/vulkan/)
- [Microsoft基于D3D12的图形编程样例](https://github.com/Microsoft/DirectX-Graphics-Samples)
- [microsoft/Xbox-ATG-Samples](https://github.com/microsoft/Xbox-ATG-Samples)
- [nBody DirectX 12 sample (asynchronous compute version)](https://gpuopen.com/gaming-product/nbody-directx-12-async-compute-edition/)
- [Learning DirectX 12 – Lesson 4 – Textures](https://www.3dgep.com/learning-directx-12-4)
- [Direct3D 11.3 Functional Specification](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm)
- [nVidia关于最新D3D的样例](https://developer.nvidia.com/gameworks-directx-samples)
- [nVidia关于D3D11的样例](https://developer.nvidia.com/dx11-samples)
- [Introduction to 3D Game Programming With DirectX11 书中代码样例](https://github.com/jjuiddong/Introduction-to-3D-Game-Programming-With-DirectX11)
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

<br/>

## GLSL源文件扩展名

- 当前Xcode 10所能识别出的GLSL文件类别: `.glsl`、`.vsh`、`.fsh`、`.gsh`、`.vert`、`.frag`、`.vert`、`.frag`、`.geom`
- [What is the correct file extension for GLSL shaders?](https://stackoverflow.com/questions/6432838/what-is-the-correct-file-extension-for-glsl-shaders)

<br />

## CUDA相关文档

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [CUDA: Common Function for both Host and Device Code](https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/)
- [CUDA constant memory issue: invalid device symbol with cudaGetSymbolAddress](https://stackoverflow.com/questions/26735808/cuda-constant-memory-issue-invalid-device-symbol-with-cudagetsymboladdress)
- [Unified Memory for CUDA Beginners](https://devblogs.nvidia.com/unified-memory-cuda-beginners/)
- [CUDA - Unified memory (Pascal at least)](https://stackoverflow.com/questions/50679657/cuda-unified-memory-pascal-at-least)
- [CUDA总结：纹理内存](https://blog.csdn.net/kelvin_yan/article/details/54019017)
- [\[CUDA\]纹理对象 Texture Object](https://blog.csdn.net/m0_38068229/article/details/89478981)
- [NVIDIA CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- [CUDA编程入门----Thrust库简介](https://blog.csdn.net/he_wolf/article/details/23502793)（文章最后有NV官方文档介绍）

- 在CUDA编译选项中有一个 **-rdc**，意思是 *Generate Relocatable Device Code*。该选项默认是关闭的，即`-rdc=false`，在此情况下，每个cuda源文件只能包含自己的全局`__device__`和`__constant__`对象，而不能引用其他cuda源文件中所定义的全局对象，同时，即便在同一cuda源文件，一个全局对象也不能声明，因为声明了它就等于定义了它，再对它定义加初始化就会出现重复定义的错误。而在将它打开的情况下，即`-rdc=true`，那么全局对象的行为就跟普通C语言全局对象的行为一样了，在一个模块中，可以跨多个cuda源文件对同一全局对象引用，同时也能做定义前的声明。因此通常情况下，我们应该考虑将此编译选项打开。

- cudaMemcpy probably isn't actually taking that long--that will synchronize and wait for the kernel to complete. Launching a kernel is (almost) always asynchronous; when you call kernel<<<...>>>(...);, it's actually just queuing work for the GPU to perform at some point. It won't block the CPU and wait for that kernel to finish or anything like that. **However, since cudaMemcpy is a synchronous function, it implies that you want the results to be visible, so that will block the CPU until the GPU becomes idle** (indicating that all of your work has completed).

- **How to make it explicit that I am not using shared memory?** -- In Volta the L1 cache, texture cache, and shared memory are backed by a combined 128 KB data cache. As in previous architectures, such as Kepler, the portion of the cache dedicated to shared memory (known as the carveout) can be selected at runtime using cudaFuncSetAttribute() with the attribute cudaFuncAttributePreferredSharedMemoryCarveout. Volta supports shared memory capacities of 0, 8, 16, 32, 64, or 96 KB per SM. **You need to explicitly set shared memory capacity to 0.**



