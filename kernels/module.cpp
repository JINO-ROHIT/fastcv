#include <torch/extension.h>

// forward declarations
torch::Tensor rgb_to_gray(torch::Tensor img)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rgb2gray", &rgb_to_gray, "rgb to grayscale kernel");
}