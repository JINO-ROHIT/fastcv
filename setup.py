from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                "kernels/grayscale.cu",
                "kernels/gaussian_blur.cu",
                "kernels/sobel.cu",
                "kernels/module.cpp",
            ],
            extra_compile_args={"cxx": ["-O2"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
