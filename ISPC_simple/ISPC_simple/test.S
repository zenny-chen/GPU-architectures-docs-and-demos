.text
.align 4
.intel_syntax noprefix

.globl MyASMTest
.globl VectorAddASM

// int MyASMTest(int a)

MyASMTest:

    inc     edi
    mov     eax, edi
    ret


// void VectorAddASM(unsigned dst[], unsigned src1[], unsigned src2[], int elemCount)

VectorAddASM:

    // dst: RDI, src1: RSI, src2: RDX, elemCount: ECX
    
    shr     ecx, 6      // elemCount /= 64 (8 elements/YMM * 8 YMM registers)

VectorAddASM_LOOP:

    vmovdqa     ymm0, ymmword ptr [rsi]
    vmovdqa     ymm1, ymmword ptr [rsi + 32]
    vmovdqa     ymm2, ymmword ptr [rsi + 64]
    vmovdqa     ymm3, ymmword ptr [rsi + 96]
    vmovdqa     ymm4, ymmword ptr [rsi + 128]
    vmovdqa     ymm5, ymmword ptr [rsi + 160]
    vmovdqa     ymm6, ymmword ptr [rsi + 192]
    vmovdqa     ymm7, ymmword ptr [rsi + 224]

    vmovdqa     ymm8, ymmword ptr [rdx]
    vmovdqa     ymm9, ymmword ptr [rdx + 32]
    vmovdqa     ymm10, ymmword ptr [rdx + 64]
    vmovdqa     ymm11, ymmword ptr [rdx + 96]
    vmovdqa     ymm12, ymmword ptr [rdx + 128]
    vmovdqa     ymm13, ymmword ptr [rdx + 160]
    vmovdqa     ymm14, ymmword ptr [rdx + 192]
    vmovdqa     ymm15, ymmword ptr [rdx + 224]

    vpaddd      ymm0, ymm0, ymm8
    vpaddd      ymm1, ymm1, ymm9
    vpaddd      ymm2, ymm2, ymm10
    vpaddd      ymm3, ymm3, ymm11
    vpaddd      ymm4, ymm4, ymm12
    vpaddd      ymm5, ymm5, ymm13
    vpaddd      ymm6, ymm6, ymm14
    vpaddd      ymm7, ymm7, ymm15

    vmovdqa     ymmword ptr [rdi], ymm0
    vmovdqa     ymmword ptr [rdi + 32], ymm1
    vmovdqa     ymmword ptr [rdi + 64], ymm2
    vmovdqa     ymmword ptr [rdi + 96], ymm3
    vmovdqa     ymmword ptr [rdi + 128], ymm4
    vmovdqa     ymmword ptr [rdi + 160], ymm5
    vmovdqa     ymmword ptr [rdi + 192], ymm6
    vmovdqa     ymmword ptr [rdi + 224], ymm7

    add     rsi, 256
    add     rdx, 256
    add     rdi, 256
    dec     ecx
    jnz     VectorAddASM_LOOP

    ret

