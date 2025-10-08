import time

import cv2
import torch
import fastcv
import numpy as np

def benchmark_dilate(sizes=[1024, 2048, 4096], runs=50):
    results = []
    
    for size in sizes:
        print(f"\n=== Benchmarking {size}x{size} image ===")
        
        img_np = np.random.randint(0, 2, (size, size), dtype=np.uint8) * 255
        img_torch = torch.from_numpy(img_np).cuda()

        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.dilate(img_np, (3, 3), iterations=1)
            
        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000  # ms per run

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.dilate(img_torch, 1) # doing 1 is half size of the filter ie (3 x 3)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000  # ms per run

        results.append((size, cv_time, fc_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUDA): {fc_time:.4f} ms")
    
    return results


if __name__ == "__main__":
    results = benchmark_dilate()
    print("\n=== Final Results ===")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)")
    for size, cv_time, fc_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms")
