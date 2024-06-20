import sys
sys.path.append("/root/img_blur/src")
from img_blur_processor import IMGBlurProcessor
import json
from tqdm import tqdm
import pandas as pd

output_file = "/root/img_blur/bc_pd.pkl"
processor = IMGBlurProcessor()
processor.input_reader()

# input_file = "/data_path/2w_lane_day.txt"

# with open(input_file,'r') as f:
#     jsons = f.readlines()
# input_imgs = []
# for j in tqdm(jsons):
#     with open(j.strip(),'r') as f:
#         json_info = json.load(f)
#     img_info = json_info['camera']
#     for view in img_info:
#         if view['name'] == 'front_middle_camera':
#             input_imgs.append("/" + view['oss_path'])
# input_imgs = input_imgs

input_file = "/data_path/badcase_img.txt"
with open (input_file, "r") as f:
    input_imgs = ["/" + _.strip() for _ in f]

n = 100
ambs = pd.Series()
for i in range((len(input_imgs) // (n + 1)) + 1):
    img_paths = input_imgs[n * i : n * (i + 1)]
    ori_imgs = processor.origin_img_reader(img_paths)
    # Mask Process
    solid_masks, morph_masks = processor.mask_process(img_paths, ori_imgs, save=False)
    ambiguitys = processor.ambiguity_calculator(img_paths, solid_masks, morph_masks)
    print(ambiguitys)
    ambs = ambs.append(pd.Series(ambiguitys), ignore_index=True)
    print(ambs)

processor.save_to_pickle(ambs, output_file)
print(output_file)
print(ambs.describe())

result = processor.load_from_pickle(output_file)
print(result)
    
