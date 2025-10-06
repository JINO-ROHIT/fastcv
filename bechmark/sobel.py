import time

import cv2
import torch
import fastcv
import numpy as np

def benchmark_sobel(sizes=[1024, 2048, 4096], runs=50):
    results = []
    
    for size in sizes:
        print(f"\n=== Benchmarking {size}x{size} image ===")
        
        img_np = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        img_torch = torch.from_numpy(img_np).cuda()

        start = time.perf_counter()
        for _ in range(runs):
            sobelx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize = 3) 
            sobely = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize = 3)
            gradient_magnitude = cv2.magnitude(sobelx, sobely)
            _ = cv2.convertScaleAbs(gradient_magnitude)
            
        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000  # ms per run

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.sobel(img_torch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000  # ms per run

        results.append((size, cv_time, fc_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUDA): {fc_time:.4f} ms")
    
    return results


if __name__ == "__main__":
    results = benchmark_sobel()
    print("\n=== Final Results ===")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)")
    for size, cv_time, fc_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms")
