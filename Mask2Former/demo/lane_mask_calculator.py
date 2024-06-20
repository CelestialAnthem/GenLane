# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from multiprocessing.pool import ThreadPool

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

sys.path.append("./detectron2")


from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

# constants
WINDOW_NAME = "mask2former demo"

class LaneMaskCalculator():
    def __init__(self, target_amb: int, amount: int, online: bool) -> None:
        self.target_amb = target_amb
        self.amount = amount
        self.online = online
        pass


    def setup_cfg(self,):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        config_file = "./Mask2Former/configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
        cfg.merge_from_file(config_file)
        model_path = "/share/generation/models/pretrained/mapillary-vistas_semantic-segmentation_swin.pkl" if self.online else "/mnt/ve_share/songyuhao/generation/models/pretrained/mapillary-vistas_semantic-segmentation_swin.pkl"
        cfg.merge_from_list(['MODEL.WEIGHTS', model_path])
        cfg.freeze()
        return cfg


    def lane_mask(self, img_paths: list, ambiguity: list = [], masks: list = []) -> list:
        
        ambiguity = [0] * self.amount if ambiguity == [] else ambiguity
        
        mp.set_start_method("spawn", force=True)
        setup_logger(name="fvcore")
        logger = setup_logger()
        cfg = self.setup_cfg()
        demo = VisualizationDemo(cfg)
        
        lane_masks = []
        for i in tqdm.tqdm(range(len(img_paths)), desc="SEMANTIC SEGMENTATION PROCESSING..."):
            # use PIL, to be consistent with evaluation
            path = img_paths[i]
            amb = ambiguity[i]
            
            if amb <= self.target_amb:
                img = read_image(path, format="BGR")
                predictions, visualized_output = demo.run_lane_binary_mask_on_image(img)
                
                lane_mask = visualized_output == 1
            else:
                mask = masks[i]
                lane_mask = mask
                
            lane_masks.append(lane_mask)
        
        # print(img_paths)
        # print(ambiguity)
        # print(masks)
        # def worker(_):
        #     path = _[0]
        #     amb = _[1]
        #     mask = _[2]
            
        #     if amb <= self.target_amb:
        #         img = read_image(path, format="BGR")
        #         start_time = time.time()
        #         predictions, visualized_output = demo.run_lane_binary_mask_on_image(img)
        #         lane_mask = visualized_output == 1
                
        #         return lane_mask

        #     else:
        #         return mask
        
        # combine_lst = list(map(lambda a, b, c: [a, b, c], img_paths, ambiguity, masks))
            
        # with ThreadPool(processes = 1) as pool:
        #     lane_masks = list(tqdm.tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='SEMANTIC SEGMENTATION PROCESSING...'))
        #     pool.terminate()
            
            
        return lane_masks
    
    def lane_mask_2(self, img_paths: list, masks: list = []) -> list:
        
        mp.set_start_method("spawn", force=True)
        setup_logger(name="fvcore")
        logger = setup_logger()
        cfg = self.setup_cfg()
        demo = VisualizationDemo(cfg)
        
        lane_masks = []
        for i in tqdm.tqdm(range(len(img_paths)), desc="SEMANTIC SEGMENTATION PROCESSING..."):
            # use PIL, to be consistent with evaluation
            path = img_paths[i]
            
            img = read_image(path, format="BGR")
            predictions, visualized_output = demo.run_lane_binary_mask_on_image(img)
            
            lane_mask = visualized_output == 1
            lane_masks.append(lane_mask)
        
        # print(img_paths)
        # print(ambiguity)
        # print(masks)
        # def worker(_):
        #     path = _[0]
        #     amb = _[1]
        #     mask = _[2]
            
        #     if amb <= self.target_amb:
        #         img = read_image(path, format="BGR")
        #         start_time = time.time()
        #         predictions, visualized_output = demo.run_lane_binary_mask_on_image(img)
        #         lane_mask = visualized_output == 1
                
        #         return lane_mask

        #     else:
        #         return mask
        
        # combine_lst = list(map(lambda a, b, c: [a, b, c], img_paths, ambiguity, masks))
            
        # with ThreadPool(processes = 1) as pool:
        #     lane_masks = list(tqdm.tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='SEMANTIC SEGMENTATION PROCESSING...'))
        #     pool.terminate()
            
            
        return lane_masks


if __name__ == "__main__":
    lane_mask_calculator = LaneMaskCalculator()
    masks = lane_mask_calculator.lane_mask(["/mnt/ve_share/songyuhao/generation/data/dev_test/10086355203788093.jpg"])
    for mask in masks:
        rows = len(mask)
        cols = len(mask[0])
        print(rows, cols)

        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if mask[i][j] == 1:
                    count += 1
                    
        print(count)

