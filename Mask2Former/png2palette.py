from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 定义调色板
palette = [
    [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], 
    [90, 120, 150], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], 
    [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110], 
    [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60], 
    [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128], [255, 255, 255], 
    [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152], 
    [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180], 
    [220, 220, 220], [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40], 
    [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150], [210, 170, 100], 
    [153, 153, 153], [128, 128, 128], [0, 0, 80], [250, 170, 30], [192, 192, 192], 
    [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255], [0, 60, 100], 
    [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], 
    [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]
]
# 将调色板转换为 NumPy 数组
bin_colormap = np.array(palette, dtype=np.uint8)
# 创建颜色到索引的映射
color_to_index = {tuple(color): index for index, color in enumerate(palette)}

root_dir = '/root/GenLane/output_2'
new_root_dir = '/root/GenLane/output_2_new'
img_path_list = [os.path.join(root_dir, img_path) for img_path in os.listdir(root_dir)]

def process_image(image_path):
    image = Image.open(image_path)
    # 将图像转换为 RGB 模式
    rgb_image = image.convert('RGB')
    image_data = np.array(rgb_image)

    # 构建一个映射表，将颜色映射到调色板索引
    flat_image_data = image_data.reshape(-1, image_data.shape[-1])
    indexed_image_data = np.zeros(flat_image_data.shape[0], dtype=np.uint8)

    # 仅需一次遍历即可映射所有颜色
    for color, index in color_to_index.items():
        matches = np.all(flat_image_data == color, axis=1)
        indexed_image_data[matches] = index

    # 重塑回原始图像形状
    index_image_data = indexed_image_data.reshape(image_data.shape[:2])

    # 创建新的 P 模式图像
    p_image = Image.fromarray(index_image_data, mode='P')
    p_image.putpalette(bin_colormap.flatten())

    # 保存图像
    output_path = os.path.join(new_root_dir, Path(image_path).name)
    p_image.save(output_path)

if __name__ == '__main__':
    os.makedirs(new_root_dir, exist_ok=True)
    num_workers = min(cpu_count(), len(img_path_list))
    with Pool(num_workers) as p:
        list(tqdm(p.imap(process_image, img_path_list), total=len(img_path_list)))
