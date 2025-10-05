#include <torch/extension.h>

// forward declarations
torch::Tensor rgb_to_gray(torch::Tensor img);
torch::Tensor gaussian_blur(torch::Tensor img, int blurSize);
torch::Tensor sobel(torch::Tensor img);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rgb2gray", &rgb_to_gray, "rgb to grayscale kernel");
    m.def("blur", &gaussian_blur, "gaussian blur kernel");
    m.def("sobel", &sobel, "sobel filter kernel");
}