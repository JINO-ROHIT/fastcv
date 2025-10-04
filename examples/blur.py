import cv2
import torch
import fastcv

img = cv2.imread("artifacts/test.jpg")
img_tensor = torch.from_numpy(img).cuda()
blurred_tensor = fastcv.blur(img_tensor, 10)

blurred_image = blurred_tensor.cpu().numpy()
cv2.imwrite("output_blur.jpg", blurred_image)

print("saved blurred image.")