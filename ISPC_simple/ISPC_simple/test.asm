.code

; int MyASMTest(int a)

MyASMTest    proc public

    inc     ecx
    mov     eax, ecx
    ret

MyASMTest    endp

; void VectorAddASM(unsigned dst[], unsigned src1[], unsigned src2[], int elemCount)

VectorAddASM    proc public

    mov     r10, rsp
    mov     eax, 15
    sub     r10, (12 * 16)     ; PUSH (12 * 16) bytes
    mov     r11, rsp    ; Use R11 to preserve original RSP
    ; get 16-byte aligned RSP
    add     r10, rax
    not     rax
    and     r10, rax
    mov     rsp, r10

    vmovdqa     xmmword ptr [rsp], xmm6
    vmovdqa     xmmword ptr [rsp + 16], xmm7
    vmovdqa     xmmword ptr [rsp + 32], xmm8
    vmovdqa     xmmword ptr [rsp + 48], xmm9
    vmovdqa     xmmword ptr [rsp + 64], xmm10
    vmovdqa     xmmword ptr [rsp + 80], xmm11
    vmovdqa     xmmword ptr [rsp + 96], xmm12
    vmovdqa     xmmword ptr [rsp + 112], xmm13
    vmovdqa     xmmword ptr [rsp + 128], xmm14
    vmovdqa     xmmword ptr [rsp + 144], xmm15

    ; dst: RCX, src1: RDX, src2: R8, elemCount: R9D
    shr         r9d, 6       ; elemCount /= 64 (8 elements/YMM * 8 YMM registers)

VectorAddASM_LOOP:

    vmovdqa     ymm0, ymmword ptr [rdx]
    vmovdqa     ymm1, ymmword ptr [rdx + 32]
    vmovdqa     ymm2, ymmword ptr [rdx + 64]
    vmovdqa     ymm3, ymmword ptr [rdx + 96]
    vmovdqa     ymm4, ymmword ptr [rdx + 128]
    vmovdqa     ymm5, ymmword ptr [rdx + 160]
    vmovdqa     ymm6, ymmword ptr [rdx + 192]
    vmovdqa     ymm7, ymmword ptr [rdx + 224]

    vmovdqa     ymm8, ymmword ptr [r8]
    vmovdqa     ymm9, ymmword ptr [r8 + 32]
    vmovdqa     ymm10, ymmword ptr [r8 + 64]
    vmovdqa     ymm11, ymmword ptr [r8 + 96]
    vmovdqa     ymm12, ymmword ptr [r8 + 128]
    vmovdqa     ymm13, ymmword ptr [r8 + 160]
    vmovdqa     ymm14, ymmword ptr [r8 + 192]
    vmovdqa     ymm15, ymmword ptr [r8 + 224]

    vpaddd      ymm0, ymm0, ymm8
    vpaddd      ymm1, ymm1, ymm9
    vpaddd      ymm2, ymm2, ymm10
    vpaddd      ymm3, ymm3, ymm11
    vpaddd      ymm4, ymm4, ymm12
    vpaddd      ymm5, ymm5, ymm13
    vpaddd      ymm6, ymm6, ymm14
    vpaddd      ymm7, ymm7, ymm15

    vmovdqa     ymmword ptr [rcx], ymm0
    vmovdqa     ymmword ptr [rcx + 32], ymm1
    vmovdqa     ymmword ptr [rcx + 64], ymm2
    vmovdqa     ymmword ptr [rcx + 96], ymm3
    vmovdqa     ymmword ptr [rcx + 128], ymm4
    vmovdqa     ymmword ptr [rcx + 160], ymm5
    vmovdqa     ymmword ptr [rcx + 192], ymm6
    vmovdqa     ymmword ptr [rcx + 224], ymm7

    add     rdx, 256
    add     r8, 256
    add     rcx, 256
    dec     r9d
    jnz     VectorAddASM_LOOP

    mov     rsp, r11    ; Restore orignial RSP
    ret

VectorAddASM    endp


end

