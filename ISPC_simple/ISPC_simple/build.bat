set cmd="C:\ispc-v1.22.0-windows\bin\ispc.exe"
%cmd%  simple.ispc  -o simple.obj  -O0  -g  --addressing=64 --arch=x86-64 --device=skylake --target=avx2-i32x8 --target-os=windows

