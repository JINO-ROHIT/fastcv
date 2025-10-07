#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


#include "utils.cuh" 

__global__ void dilationKernel(unsigned char *in,
               unsigned char *out,
               int w,
               int h,
               int FILTER_SIZE) {
  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < w && row < h) {
        
        unsigned char pixMax = 0;


        for (int f_row = -FILTER_SIZE; f_row <= FILTER_SIZE; ++f_row) {
            for (int f_col = -FILTER_SIZE; f_col <= FILTER_SIZE; ++f_col) {
            
                int curRow = f_row + row;
                int curCol = f_col + col;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixMax = max(pixMax, in[curRow * w + curCol]);
                }
            }
        }

        out[row * w + col] = pixMax;
    }
}

torch::Tensor dilation(torch::Tensor img, int filterSize) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    assert(img.dim() == 2); 

    const auto height = img.size(0);
    const auto width = img.size(1);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));


    auto result = torch::empty({height, width}, 
                torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    dilationKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        width,
        height,
        filterSize
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
