#pragma once
#include <cuda_runtime.h>

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

inline dim3 getOptimalBlockDim(int width, int height) {
    if (width < 16 || height < 16) {
        return dim3(8, 8);
    }

    if (width >= 1024 && height >= 1024) {
        return dim3(32, 32);
    }

    return dim3(16, 16); 
}