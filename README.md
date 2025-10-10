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


## Benchmarks

Tested on: **RTX 4060 Ti**

| Kernel      | Image Size | OpenCV (CPU) | fastcv (CUDA) | OpenCV CUDA  (soon)         | Speedup (×) |
|------------ |-----------:|-------------:|--------------:|----------------------------:|------------:|
| RGB2GRAY    | 4096×4096  | 3.1841 ms    | 0.2737 ms     | -                           | 11.63       |
| BOX BLUR    | 4096×4096  | 44.459 ms    | 10.012 ms     | -                           | 4.44        |
| SOBEL       | 4096×4096  | 384.83 ms    | 0.3223 ms     | -                           | 1194.0      |
| EROSION     | 4096×4096  | 2.9758 ms    | 0.5827 ms     | -                           | 5.11        |
| DILATION    | 4096×4096  | 2.7539 ms    | 0.5856 ms     | -                           | 4.70        |