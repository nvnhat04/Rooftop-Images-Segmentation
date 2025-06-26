import torch
from unet_model import UNet
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def dice_score(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

model = UNet(n_channels=3, n_classes=1, bilinear=True)
model.load_state_dict(torch.load('model_weights/best_model.pth', map_location=device))
model.to(device)
model.eval()

# # Đường dẫn ảnh
image_path = 'data/images/Caugiay_1.1_patch_0064.png'  # sửa tên ảnh thật
mask_path = 'data/masks/Caugiay_1.1_patch_0064.png'
# Tiền xử lý
input_tensor = preprocess_image(image_path)
# mask = Image.open(mask_path)
mask = Image.open(mask_path).convert('L')
mask = transforms.Resize((256, 256))(mask)
mask = transforms.ToTensor()(mask)
mask = mask.unsqueeze(0).to(device)  # shape: (1, 1, H, W)

# Dự đoán
with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output) 
    pred_mask = (output > 0.45).float()  # Ngưỡng hóa
#dice
dice = dice_score(pred_mask, mask)
iou = iou_score(pred_mask, mask)
print(f'Dice Score: {dice.item()}')
print(f'IoU Score: {iou.item()}')    
# Hiển thị kết quả
plt.figure(figsize=(10, 4))

# Hiển thị ảnh gốc
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(input_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
plt.axis('off')

# Hiển thị mask dự đoán
plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(pred_mask.squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Mask")
# plt.imshow(mask)
plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
plt.axis('off')
plt.show()

