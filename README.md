## fastcv

`fastcv` is a C++ CUDA rewrite with Pytorch bindings of the image filters in the OpenCV library.

### Filters To implement

- Grayscale [x]
- Brightness []
- Thresholding []
- Mean Blur []
- Gaussian Blur []
- TBD

### How to run

1. Clone the repo

```bash
https://github.com/JINO-ROHIT/fastcv
```

2. Move to fastcv

```bash
cd fastcv
```

3. Build the package

```bash
pip install -e . --no-build-isolation
```

You’ll need:
1. CUDA toolkit installed
2. PyTorch with CUDA support
3. A working C++ compiler (MSVC on Windows, gcc/clang on Linux/Mac)


### Current Benchmarks

Note: This was tested on an rtx 4060 ti.

1. RGB2GRAY KERNEL

| Image Size | OpenCV (CPU) | fastcv (CUDA) |
|-----------:|-------------:|--------------:|
| 1024×1024  | 0.3785 ms    | 0.0811 ms     |
| 2048×2048  | 0.7343 ms    | 0.2547 ms     |
| 4096×4096  | 3.1841 ms    | 0.9857 ms     |