import os
from PIL import Image

def create_patches(image_path, mask_path, output_img_dir, output_mask_dir, patch_size=256):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # mask là ảnh nhị phân

    width, height = img.size
    print(f"Kích thước ảnh: {width}x{height}")
    basename = os.path.splitext(os.path.basename(image_path))[0]

    count = 0
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            box = (left, top, left + patch_size, top + patch_size)
            img_patch = img.crop(box)
            mask_patch = mask.crop(box)

            if img_patch.size[0] != patch_size or img_patch.size[1] != patch_size:
                continue  # bỏ qua nếu patch không đủ kích thước

            img_patch.save(os.path.join(output_img_dir, f"{basename}_patch_{count:04d}.png"))
            mask_patch.save(os.path.join(output_mask_dir, f"{basename}_patch_{count:04d}.png"))
            count += 1

    print(f"Đã tạo {count} patch cho {basename}")

# Áp dụng cho 3 cặp ảnh
# create_patches("image/Caugiay_1.1.png", "gt-image/Caugiay_1.1.png", "data/images", "data/masks")
create_patches("image/Caugiay_1.2.png", "gt-image/Caugiay_1.2.png", "data/images", "data/masks")
create_patches("image/Caugiay_1.3.png", "gt-image/Caugiay_1.3.png", "data/images", "data/masks")
