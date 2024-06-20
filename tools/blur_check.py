import sys
sys.path.append("/root/img_blur/src")
from img_blur_processor import IMGBlurProcessor
import json
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import cv2
from PIL import Image


total = 200
ori = "/root/img_blur/origin.txt"
gen = "/root/img_blur/gen.txt"

with open(ori, "r") as ori_file:
    ori_imgs = np.array(["/" + _.strip() for _ in ori_file])
    
with open(gen, "r") as gen_file:
    gen_imgs = np.array(["/" + _.strip() for _ in gen_file])

amount = len(ori_imgs)

selected = [random.randint(0, amount) for _ in range(total)]

assert len(ori_imgs) == len(gen_imgs)

output_add = "/mnt/ve_share/songyuhao/generation/mask_compare"
processor = IMGBlurProcessor(output = output_add)
processor.input_reader()


def hconcattt(imgs, masks) -> list:
    res = []
    for i in tqdm(range(len(imgs))):
        cv2_img = cv2.cvtColor(np.array(imgs[i]), cv2.COLOR_RGB2BGR)
        cv2_mask = cv2.cvtColor(np.array(masks[i]), cv2.COLOR_RGB2BGR)
        hc = cv2.hconcat([cv2_img, cv2_mask])
        res.append(hc)
        
    return res

def save_conbined(oris, gens, paths):
    final_imgs = []
    for i in tqdm(range(len(paths))):
        combined = cv2.vconcat([oris[i], gens[i]])
        pil_img = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        final_img = Image.fromarray(np.uint8(pil_img))
        final_imgs.append(final_img)
    processor.img_saver(paths, final_imgs, "_compare")

n = 20
for i in range((len(selected) // (n + 1)) + 1):
    
    iter_selected = selected[n * i : n * (i + 1)]
    
    ori_paths = ori_imgs[iter_selected]
    oris = processor.origin_img_reader(ori_paths)
    ori_masks = processor.lane_mask_reader(ori_paths)
    ori_mask_imgs = processor.mask2img(oris, ori_masks)
    
    ori_concat = hconcattt(oris, ori_mask_imgs)
    
    gen_paths = gen_imgs[iter_selected]
    gens = processor.origin_img_reader(gen_paths)
    gen_masks = processor.lane_mask_reader(gen_paths)
    gen_mask_imgs = processor.mask2img(gens, gen_masks)
    
    gen_concat = hconcattt(gens, gen_mask_imgs)
    
    save_conbined(ori_concat, gen_concat, ori_paths)

print(output_add)
