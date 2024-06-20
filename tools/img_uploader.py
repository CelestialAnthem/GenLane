import sys
sys.path.append("/share/GANs/tools/haomo_ai_framework")
from haomoai.cards import CardOperation
import glob
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import json
from pathlib import Path


def sort_by_name(lst: list) -> list:
    sorted_lst = sorted(lst, key=lambda _: _.split("/")[-1])
    return sorted_lst


def create_json(json_txt: list, ori_imgs: list, gen_imgs: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Create New JSON
    with open(json_txt, "r") as input_file:
        json_paths = [_ .strip() for _ in input_file]
    
    for json_file in tqdm(json_paths, desc="CREATING JSON"):
        with open(json_file) as f:
            json_info = json.load(f)
        img_info = json_info['camera']
        for view in img_info:
            if view['name'] == 'front_middle_camera':
                ori_img = view['oss_path']
                ind = ori_imgs.index(ori_img)
                gen_img = gen_imgs[ind]
                view["oss_path_blur_lane"] = gen_img
                json_info["gen_blur_lane"] = True
                # print(json_info)
                # print(json_file)
                json_name = json_file.split("/")[-1].split(".")[0]
                # print(json_name)
                with open("%s/%s.json" % (output_dir, json_name), "w") as output_file:
                    json.dump(json_info, output_file)
    print(output_dir)


def upload_img(img_dir: str, project: str = "icu30", media_name: str = "LANE"):
    card_inst = CardOperation() # 实例化一个对象
    
    # Upload imgs
    img_path_list = [str(path) for path in Path(img_dir).rglob("*.jpg")] + [str(path) for path in Path(img_dir).rglob("*.png")]
    img_names = [_.split('/')[-1] for _ in img_path_list]
    card_id = card_inst.create_card_oss(project=project, media_name=media_name, file_paths=img_path_list, names=img_names)
    print("Card id: %s, Project: %s, Media_Name: %s" % (card_id, project, media_name))
    new_img_osses = card_inst.get_oss_paths(card_id, project, media_name)
    
    # Save new img paths
    output_name = "%s/new_img_paths.txt" % img_dir
    with open (output_name, "w") as output_file:
        for img_oss in tqdm(new_img_osses, desc="UPLOADING IMGs"):
            output_file.writelines(img_oss + "\n")
    print(output_name)
    
    return new_img_osses
    
    
if __name__ == "__main__":
    for i in [5]:
        img_dir = "/mnt/ve_share/songyuhao/generation/data/new_lane_output_%d" % i
        json_txt = "/data_path/0301_lane%d.txt" % i
        output_dir = "/mnt/ve_share/songyuhao/generation/data/new_lane_json_%d" % i
        
        gen_path = "%s/new_img_paths.txt" % img_dir
        
        if Path(gen_path).is_file():
            # If already upload imgs
            with open (gen_path, "r") as gen_file:
                gen_imgs = [_.strip() for _ in gen_file]
        else:
            # If upload imgs now
            gen_imgs = upload_img(img_dir)
        
        ori_path = "%s/img_paths.txt" % img_dir
        with open (ori_path, "r") as ori_file:
            ori_imgs = [_.strip()[1:] if _.startswith("/") else _.strip() for _ in ori_file]
            
        ori_imgs = sort_by_name(ori_imgs)
        gen_imgs = sort_by_name(gen_imgs)
            
        create_json(json_txt, ori_imgs, gen_imgs, output_dir)
    