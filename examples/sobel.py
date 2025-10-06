import cv2
import torch
import fastcv

img = cv2.imread("artifacts/grayscale.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
gray_tensor = fastcv.sobel(img_tensor)
gray_np = gray_tensor.cpu().numpy()
cv2.imwrite("output_sobel.jpg", gray_np)

print("saved sobel image.")
