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
| RGB2GRAY    | 1024×1024  | 0.3785 ms    | 0.0288 ms     | -                           | 13.13       |
| RGB2GRAY    | 2048×2048  | 0.7343 ms    | 0.0398 ms     | -                           | 18.44       |
| RGB2GRAY    | 4096×4096  | 3.1841 ms    | 0.2737 ms     | -                           | 11.63       |
| BOX BLUR    | 1024×1024  | 3.9455 ms    | 0.6731 ms     | -                           | 5.86        |
| BOX BLUR    | 2048×2048  | 11.019 ms    | 2.4990 ms     | -                           | 4.41        |
| BOX BLUR    | 4096×4096  | 44.459 ms    | 10.012 ms     | -                           | 4.44        |
| SOBEL       | 1024×1024  | 23.287 ms    | 0.0373 ms     | -                           | 624.4       |
| SOBEL       | 2048×2048  | 94.812 ms    | 0.0808 ms     | -                           | 1173.9      |
| SOBEL       | 4096×4096  | 384.83 ms    | 0.3223 ms     | -                           | 1194.0      |
| EROSION     | 1024×1024  | 0.0981 ms    | 0.0443 ms     | -                           | 2.21        |
| EROSION     | 2048×2048  | 0.3486 ms    | 0.1470 ms     | -                           | 2.37        |
| EROSION     | 4096×4096  | 2.9758 ms    | 0.5827 ms     | -                           | 5.11        |
| DILATION    | 1024×1024  | 0.0930 ms    | 0.0433 ms     | -                           | 2.15        |
| DILATION    | 2048×2048  | 0.4037 ms    | 0.1480 ms     | -                           | 2.73        |
| DILATION    | 4096×4096  | 2.7539 ms    | 0.5856 ms     | -                           | 4.70        |