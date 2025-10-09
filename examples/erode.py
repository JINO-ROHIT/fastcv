import cv2
import torch
import fastcv

img = cv2.imread("artifacts/binary.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
gray_tensor = fastcv.erode(img_tensor, 1)
gray_np = gray_tensor.squeeze(-1).cpu().numpy()
cv2.imwrite("output_eroded.jpg", gray_np)

print("saved eroded image.")