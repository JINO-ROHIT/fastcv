import cv2
import torch
import fastcv

img = cv2.imread("artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
gray_tensor = fastcv.rgb2gray(img_tensor)
gray_np = gray_tensor.squeeze(-1).cpu().numpy()
cv2.imwrite("output_gray.jpg", gray_np)

print("saved grayscale image.")
