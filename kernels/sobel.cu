// assumes a grayscale image
// calculate the Gx and Gy filters by convolving for horizontal and vertical edges.

//       _               _                   _                _
//      |                 |                 |                  |
//      | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
// Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
//      | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
//      |_               _|                 |_                _|

// calculate the gradient magnitude.

#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "utils.cuh"

__global__ void sobelKernel(unsigned char* Pin, unsigned char* Pout, int width, int height) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= height || c >= width) {
        return;
    }

    // sobel kernels
    const int Kx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
    const int Ky[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

    int gx = 0;
    int gy = 0;

    for (int i = -1; i <= 1; i++) { 
        for (int j = -1; j <= 1; j++) { 
            
            int nr = r + i; 
            int nc = c + j; 

            unsigned char I_in = 0;
            if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                I_in = Pin[nr * width + nc];
            }

            gx += I_in * Kx[i + 1][j + 1];
            gy += I_in * Ky[i + 1][j + 1];
        }
    }
    
    // fast approximation |Gx| + |Gy|
    int magnitude = abs(gx) + abs(gy); 

    
    // clamp the 32-bit integer result to 8-bit unsigned char [0, 255]
    unsigned char output_value = (magnitude > 255) ? 255 : (unsigned char)magnitude;

    Pout[r * width + c] = output_value;
}


torch::Tensor sobel(torch::Tensor img){
    TORCH_CHECK(img.device().type() == torch::kCUDA);
    TORCH_CHECK(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width, 1}, 
                               torch::TensorOptions()
                                   .dtype(torch::kByte)
                                   .device(img.device()));

    sobelKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(), 
        result.data_ptr<unsigned char>(), 
        width, 
        height);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}