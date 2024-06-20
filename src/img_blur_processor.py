import argparse
import math
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, "./Mask2Former/demo")
import glob
import json
import os
import pickle
import random
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
from lane_mask_calculator import LaneMaskCalculator
from tqdm import tqdm


class IMGBlurProcessor():
    def __init__(self, input: list or str = "", strength : int = 0.3, max_iter: int = 5, output: str ="", blur_type: str = "Gaussian", online: bool = False) -> None:
        self.input = input
        self.output = output
        self.already_save = [str(path) for path in Path(output).rglob("*.jpg")] + [str(path) for path in Path(output).rglob("*.png")]
        self.already_save = [_.split("/")[-1].split("_")[0] for _ in self.already_save]
        self.strength = strength
        self.max_iter = max_iter
        self.input_imgs = []
        self.blur_type = blur_type
        self.online = online
        if self.output == "" and type(self.input) == str:
            self.output = "%s_%s_v1.5" % (self.input, self.blur_type)
        os.makedirs(self.output, exist_ok=True)
        # self.mask_output = "%s_mask" % self.output
        # os.makedirs(self.mask_output, exist_ok=True)

        
    def input_reader(self) -> None:
        print(self.input)
        if type(self.input) == list:
            self.input_imgs = self.input
        elif type(self.input) == str:
            if '.txt' in self.input:  # jsons in txt
                with open(self.input,'r') as f:
                    jsons = f.readlines()
                input_imgs = []
                for j in jsons:
                    with open(j.strip(),'r') as f:
                        json_info = json.load(f)
                    img_info = json_info['camera']
                    for view in img_info:
                        if view['name'] == 'front_middle_camera':
                            input_imgs.append("/" + view['oss_path'])
                self.input_imgs = input_imgs
            elif '.txt' not in self.input:
                if self.input == "":
                    self.input_imgs = []
                else:
                    p_list = os.listdir(self.input)
                    input_imgs = [os.path.join(self.input, p) for p in p_list]
                    self.input_imgs = input_imgs
            else:
                self.input_imgs = glob.glob(os.path.expanduser(self.input))
                assert self.input_imgs, "The input path(s) was not found"
        
        self.save_img_paths()
        print("TOTAL IAMGE AMOUNT: %d" % len(self.input_imgs))
        self.input_imgs = [_ for _ in self.input_imgs if _.split("/")[-1].split(".")[0] not in self.already_save]
        print("TOTAL EXIST IMAGES: %d" % len(self.already_save))
        self.amount = len(self.input_imgs)
        self.lane_mask_calculator = LaneMaskCalculator(self.strength, self.amount, self.online)
        print("REAL TOTAL IAMGE AMOUNT: %d" % self.amount)
        if self.amount == 0:
            sys.exit(0)
    
    
    def save_img_paths(self, ):
        txt_path = "%s/img_paths.txt" % self.output
        with open (txt_path, "w") as output_file:
            for img_path in self.input_imgs:
                output_file.writelines(img_path + "\n")
        
    
    def origin_img_reader(self, img_paths: list) -> list:

        def worker(_):
            img_path = _
            img = Image.open(img_path)
            return img
        
        with ThreadPool(processes = 20) as pool:
            img_maps = list(tqdm(pool.imap(worker, img_paths), total=len(img_paths), desc='ORIGIN READING...'))
            pool.terminate()

        return img_maps
    
        # img_maps = []
        # for img_path in img_paths:
        #     img = Image.open(img_path)
        #     img_maps.append(img)
            
        # return img_maps

    
    def blur_img_processor(self, imgs: list, blur_type: str, p1: list = [], p2: list = [], p3: list = []) -> list:
        
        p1 = [0] * len(imgs) if p1 == [] else p1
        p2 = [0] * len(imgs) if p2 == [] else p2
        
        def worker(_):
            img = _[0]
            if blur_type == "Gaussian":
                p1 = _[1]
                blur_img = img.filter(ImageFilter.GaussianBlur(radius=p1))
                
            elif blur_type == "Box":
                blur_img = img.filter(ImageFilter.BoxBlur(radius=10))
                
            elif blur_type == "Bilateral":
                p1 = _[1]
                p2 = _[2]
                # Filter size: Large filters (d > 5) are very slow, 
                # so it is recommended to use d=5 for real-time applications, 
                # and perhaps d=9 for offline applications that need heavy noise filtering.
                
                # Sigma values: For simplicity, you can set the 2 sigma values to be the same. 
                # If they are small (< 10), the filter will not have much effect, whereas if they are large (> 150), 
                # they will have a very strong effect, making the image look "cartoonish".
                
                # d: Diameter of each pixel neighborhood.
                # sigmaColor: Value of \sigma  in the color space. The greater the value, 
                # the colors farther to each other will start to get mixed.
                # sigmaSpace: Value of \sigma  in the coordinate space. 
                # The greater its value, the more further pixels will mix together, 
                # given that their colors lie within the sigmaColor range.
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                temp_img = cv2.bilateralFilter(img, d=p1, sigmaColor=p2, sigmaSpace=p2)
                pil_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                blur_img = Image.fromarray(np.uint8(pil_img))
                
            elif blur_type == "Median":
                # ksize	aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                temp_img = cv2.medianBlur(img, ksize=5)
                pil_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                blur_img = Image.fromarray(np.uint8(pil_img))
                
            elif blur_type == "Gaussian-CV2":
                # ksize	Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. 
                # Or, they can be zero's and then they are computed from sigma.
                
                # sigmaX	Gaussian kernel standard deviation in X direction.
                
                # sigmaY	Gaussian kernel standard deviation in Y direction; 
                # if sigmaY is zero, it is set to be equal to sigmaX, 
                # if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively (see getGaussianKernel for details); 
                # to fully control the result regardless of possible future modifications of all this semantics, 
                # it is recommended to specify all of ksize, sigmaX, and sigmaY.
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                temp_img = cv2.GaussianBlur(img, (5, 5), 0)
                pil_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                blur_img = Image.fromarray(np.uint8(pil_img))
                
            elif blur_type == "CV2":
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                temp_img = cv2.blur(img, (5, 5))
                pil_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                blur_img = Image.fromarray(np.uint8(pil_img))
            
            return blur_img
            
        combine_lst = list(map(lambda a, b, c: [a, b, c], imgs, p1, p2))
            
        with ThreadPool(processes = 20) as pool:
            blur_maps = list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='PIXEL BLUR PROCESSING...'))
            pool.terminate()
            
        return blur_maps
            
       
    def lane_mask_reader(self, img_paths: list) -> list:
        masks = self.lane_mask_calculator.lane_mask_2(img_paths)
        return masks
        
    
    def pixel_replacer(self, ori_imgs: list, masks: list, blur_imgs: list, dropout: float):
        
        def worker(_):
            mask = _[0]
            ori_img = _[1]
            blur_img = _[2]
            random_mask = np.random.rand(ori_img.size[1], ori_img.size[0])
            random_bool = random_mask > dropout
            mask_bool = mask == 1 
            
            # Replace Pixel
            ori_np = np.asarray(ori_img)
            blur_np = np.asarray(blur_img)
            temp_np = ori_np.copy()
            temp_np[mask_bool & random_bool] = blur_np[mask_bool & random_bool]
            
            temp_img = Image.fromarray(np.uint8(temp_np))
            final_mask = mask_bool & random_bool
            
            return temp_img, final_mask
            
        combine_lst = list(map(lambda a, b, c: [a, b, c], masks, ori_imgs, blur_imgs))
        
        with ThreadPool(processes = 20) as pool:
            result = list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='PIXEL REPLACING...'))
            pool.terminate()
        
        replaced_imgs, final_masks = list(map(list, zip(*result)))
        
        return replaced_imgs, final_masks
    
    
    def mask_pickle_saver(self, img_paths: list, masks: list) -> None:
        def worker(_):
            mask_name = _[0].split("/")[-1].split(".")[0]
            output_name = "%s/%s.pkl" % (self.mask_output, mask_name)
            self.save_to_pickle(_[1], output_name)
                        
        combine_lst = list(map(lambda a, b: [a, b], img_paths, masks))
        with ThreadPool(processes = 20) as pool:
            list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='MASK SAVING...'))
            pool.terminate()
        
        
    def mask_loader(self, img_paths: list) -> None:
        def worker(_):
            mask_name = _.split("/")[-1].split(".")[0]
            output_name = "%s/%s.pkl" % (self.mask_output, mask_name)
            return self.load_from_pickle(output_name)
                        
        with ThreadPool(processes = 20) as pool:
            masks = list(tqdm(pool.imap(worker, img_paths), total=len(img_paths), desc='MASK LOADING...'))
            pool.terminate()
        return masks


    def save_to_pickle(self, pickle_obj: dict, save_path: str) -> None:
        with open(save_path, "wb") as pickle_file: 
            pickle.dump(pickle_obj, pickle_file)
            
            
    def load_from_pickle(self, load_path: str):
        with open(load_path, "rb") as pickle_file:
            pickle_obj = pickle.load(pickle_file)
        return pickle_obj

            
    def img_saver(self, img_paths: list, imgs: list, suffix: str = "", save: bool = True) -> None:
        def worker(_):
            img_name = _[0].split("/")[-1].split(".")[0]
            output_name = "%s/%s%s.png" % (self.output, img_name, suffix)
            if save:
                _[1].save(output_name)
            return output_name
            
        combine_lst = list(map(lambda a, b: [a, b], img_paths, imgs))
        with ThreadPool(processes = 20) as pool:
            img_paths = list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='IMG SAVING / PATH CALCULATING...'))
            pool.terminate()
            
        return img_paths
    
    
    def mask2img(self, imgs: list, masks: list) -> None:
        def worker(_):
            img = _[0]
            mask = _[1]
            img_mask = np.zeros_like(img)
            img_mask[mask] = (255, 255, 255)
            img_mask[~mask] = (0, 0, 0)
            return Image.fromarray(img_mask)
            
        combine_lst = list(map(lambda a, b: [a, b], imgs, masks))
        
        with ThreadPool(processes = 20) as pool:
            img_masks = list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='MASK TO IMG...'))
            pool.terminate()
            
        return img_masks
            
            
    def ambiguity_calculator(self, img_paths: list, masks1: list, masks2: list) -> list:
        ambiguity_lst = []
        for i in tqdm(range(len(masks1)), desc="AMBIGUITY CALCULATING..."):
            mask1 = masks1[i]
            mask2 = masks2[i]
            # print("#############################################")
            # print("M1", np.count_nonzero(mask1))
            # print("M2", np.count_nonzero(mask2))
            mask1_n = np.count_nonzero(mask1)
            mask2_n = np.count_nonzero(mask2)
            ambiguity = round(1 - mask1_n / mask2_n, 4) if mask2_n != 0 else 0
            # print(ambiguity)
            # print("#############################################")
            ambiguity_lst.append(ambiguity)
            # print(img_paths[i], ambiguity)
        return ambiguity_lst
    
    
    def solid_decider(self, imgs: list, masks: list) -> list:
        
        def worker(_):
            img = np.asarray(_[0])
            mask = _[1]
            
            lane_img = np.zeros_like(img)
            lane_img[mask] = img[mask]
            mean_all = np.sum(lane_img) / (np.sum(mask) * 3 + 0.0000001)
            solid_mask = np.mean(lane_img, 2) > mean_all
            
            return solid_mask
            
        combine_lst = list(map(lambda a, b: [a, b], imgs, masks))
        
        with ThreadPool(processes = 20) as pool:
            solid_masks = list(tqdm(pool.imap(worker, combine_lst), total=len(combine_lst), desc='SOLID LANE DEFINING...'))
            pool.terminate()
        
        return solid_masks
        
        
    def morph_mask_calculator(self, masks: list) -> list:
        
        def worker(_):
            mask = _
            
            kernel = np.ones((1, 5), np.uint8)
            morph_mask = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_CLOSE, kernel, anchor=(2,0), iterations=5)
            morph_mask = morph_mask == 1
            
            return morph_mask
            
        with ThreadPool(processes = 20) as pool:
            morph_masks = list(tqdm(pool.imap(worker, masks), total=len(masks), desc='MORPH MASK CALCULATING...'))
            pool.terminate()
        
        return morph_masks
        
        
    def smooth_post_process(self, img_paths: list, masks: list, imgs: list, strengths: list) -> list:
        bila_1, bila_2 = self.bila_calculator(strengths)
        # print("bila_1", bila_1)
        # print("bila_2", bila_2)
        blur_imgs = self.blur_img_processor(imgs, "Bilateral", bila_1, bila_2)
        replaced_imgs, final_masks = self.pixel_replacer(imgs, masks, blur_imgs, dropout=0.0)
        res_img_paths = self.img_saver(img_paths, replaced_imgs, "_bila")
        return replaced_imgs
        
        
    def smooth_post_process_2(self, img_paths: list, masks: list, imgs: list) -> list:
        blur_imgs = self.blur_img_processor(imgs, "CV2")
        replaced_imgs, final_masks = self.pixel_replacer(imgs, masks, blur_imgs, dropout=0.0)
        res_img_paths = self.img_saver(img_paths, replaced_imgs, "_CV2")
    
    
    def mask_process(self, img_paths: list, ori_imgs: list, save: bool = False):
        # Inf Semantic Segmentation
        ori_masks = self.lane_mask_reader(img_paths)
        # print("MO", np.count_nonzero(ori_masks[0]))
        
        if save:
            ori_mask_imgs = self.mask2img(ori_imgs, ori_masks)
            self.img_saver(img_paths, ori_mask_imgs, "_ori_mask")
        
        
        # Decide Solid
        solid_masks = self.solid_decider(ori_imgs, ori_masks)
        # print("MS", np.count_nonzero(solid_masks[0]))
        
        if save:
            solid_mask_imgs = self.mask2img(ori_imgs, solid_masks)
            self.img_saver(img_paths, solid_mask_imgs, "_solid_mask")
        
        
        # Morph Masks
        morph_masks = self.morph_mask_calculator(solid_masks)
        # print("MM", np.count_nonzero(morph_masks[0]))
        
        if save:
            morph_mask_imgs = self.mask2img(ori_imgs, morph_masks)
            self.img_saver(img_paths, morph_mask_imgs, "_morph_mask")
        
        return solid_masks, morph_masks
    
    
    def cycle_post_process(self, solid_masks: list, imgs: list, para_list: list, img_paths: list = []):
        for i in range(self.max_iter):
            blur_imgs = self.blur_img_processor(imgs, self.blur_type, para_list[i])
            replaced_imgs, final_masks = self.pixel_replacer(imgs, solid_masks, blur_imgs, dropout=0.9)
            imgs = replaced_imgs  
            # img_paths = self.img_saver(img_paths, imgs, "+")
            
        return replaced_imgs
    
    
    def resize(self, imgs: list, ratio: int):
        def worker(_):
            img_size = _.size
            img_resized = _.resize((img_size[0] // ratio, img_size[1] // ratio), Image.ANTIALIAS)
            img_reresized = img_resized.resize(img_size, Image.ANTIALIAS)
            return img_reresized
            
        with ThreadPool(processes = 20) as pool:
            img_lst = list(tqdm(pool.imap(worker, imgs), total=len(imgs), desc='RESIZING...'))
            pool.terminate()
            
        return img_lst
        
    
    def para_calculator(self, ori_paras: list, cycle_paras: list) -> list:
        para_lst = []
        for i in range(self.max_iter):
            iter_lst = [ori_paras[_] + (i + 1) * cycle_paras[_] for _ in range(len(ori_paras))]
            para_lst.append(iter_lst)
        return para_lst
    
    
    def bila_calculator(self, strengths: list):
        bila_lst_1 = [5 + int((strength * 10) ** 2 / 10 * 4) if strength > 0 else 5 for strength in strengths]
        bila_lst_2 = [75 + int((strength * 10) ** 2 / 10 * 75) if strength > 0 else 75 for strength in strengths]
        return bila_lst_1, bila_lst_2
        
        
    def run(self,) -> None:
        start = datetime.now()
        self.input_reader()
        n = 50
        for i in range((len(self.input_imgs) // (n + 1)) + 1):
            img_paths = self.input_imgs[n * i : n * (i + 1)]
            ori_imgs = self.origin_img_reader(img_paths)
            
            # Mask Process
            solid_masks, morph_masks = self.mask_process(img_paths, ori_imgs, save=False)
            
            
            ambiguitys = self.ambiguity_calculator(img_paths, solid_masks, morph_masks)
            # print("amb", ambiguitys)
            
            strengths = [self.strength - _ for _ in ambiguitys]
            # print("strength", strengths)
            ori_paras = [int(_ * 100) if _ > 0 else 10 for _ in strengths]
            cycle_paras = [int(_ * 30) if _ > 0 else 2 for _ in strengths]
            # print("ori_paras", ori_paras)
            # print("cycle_paras", cycle_paras)
            
            # BLUR
            blur_imgs = self.blur_img_processor(ori_imgs, self.blur_type, ori_paras)
            # self.img_saver(img_paths, blur_imgs, "_blur")
            
            # Replace Pixel
            replaced_imgs, final_masks = self.pixel_replacer(ori_imgs, solid_masks, blur_imgs, dropout=0.0)
            # Save Result IMG
            # res_img_paths = self.img_saver(img_paths, replaced_imgs)
            
            # Mask to Image
            # final_mask_imgs = self.mask2img(ori_imgs, final_masks)
            # self.img_saver(img_paths, final_mask_imgs, "_final_mask")
            
            # Post-process
            cycle_paras = self.para_calculator(ori_paras, cycle_paras)
            # cycle_imgs = self.cycle_post_process(solid_masks, res_img_paths, replaced_imgs, cycle_paras)
            cycle_imgs = self.cycle_post_process(solid_masks, replaced_imgs, cycle_paras)
            smooth_imgs = self.smooth_post_process(img_paths, solid_masks, cycle_imgs, strengths)
            
        print(datetime.now() - start)
        print("IMAGES HAVE BEEN SAVED IN: %s" % self.output)
        # print("PKLS HAVE BEEN SAVED IN: %s" % self.mask_output)

            
if __name__ == "__main__":
    blur_types = ["Gaussian", "Bilateral", "Box", "Median", "Gaussian-CV2", "CV2"]
    blur_type = blur_types[0]
    input = "/mnt/ve_share/songyuhao/generation/data/test/v0.0"
    strength = 0.3
    max_iter = 5
    output = "/mnt/ve_share/songyuhao/generation/data/lane_output_-1" 
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input: txt of json | images in folder",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ouput Folder",
    )
    
    parser.add_argument(
        "--strength",
        default=0.3,
        type=float,
        help="Blurred Strength",
    )
    
    parser.add_argument(
        "--max_iter",
        default=5,
        type=int,
        help="Blurred Strength",
    )
    
    parser.add_argument(
        "--blur_type",
        default=blur_type,
        type=str,
        help="Max Iter",
    )
    
    parser.add_argument(
        "--online",
        default=False,
        type=bool,
        help="online",
    )
    
    args = parser.parse_args()
    img_blur_processor = IMGBlurProcessor(args.input, args.strength, args.max_iter, args.output, args.blur_type, args.online)
    img_blur_processor.run()