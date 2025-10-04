## fastcv

`fastcv` is a C++ CUDA rewrite with Pytorch bindings of the image filters in the OpenCV library.

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
| 1024×1024  | 0.3785 ms    | 0.0288 ms     |
| 2048×2048  | 0.7343 ms    | 0.0398 ms     |
| 4096×4096  | 3.1841 ms    | 0.2737 ms     |

1. BLUR KERNEL

| Image Size | OpenCV (CPU) | fastcv (CUDA) |
|-----------:|-------------:|--------------:|
| 1024×1024  | 3.9455 ms    | 0.6731 ms     |
| 2048×2048  | 11.019 ms    | 2.4990 ms     |
| 4096×4096  | 44.459 ms    | 10.012 ms     |


### TO-DO

1. read sobel filters and implement.
2. morph ops.
3. feature detection.
4. TBD.