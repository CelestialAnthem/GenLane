# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings
import json

import cv2
import numpy as np
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

# Constants
WINDOW_NAME = "mask2former demo"

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--dict",
        help="json dict name",
        default=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def categorize_by_resolution(res_dict):
    categorized_dict = {}
    for item in res_dict:
        img_path = item["img_path"]
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        resolution_key = f"{width}_{height}"
        if resolution_key not in categorized_dict:
            categorized_dict[resolution_key] = []
        categorized_dict[resolution_key].append(item)
    return categorized_dict

def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].extend(value)
        else:
            dict1[key] = value
    return dict1

if __name__ == "__main__":
    PURE = True
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            if '.txt' in args.input[0]:
                with open(args.input[0],'r') as f:
                    lines = f.readlines()
                args.input = [line.strip() for line in lines]
            elif '.' not in args.input[0]:
                p_list = os.listdir(args.input[0])
                input_imgs = [os.path.join(args.input[0], p) for p in p_list]
                args.input = input_imgs
            else:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
                
        res_dict = []
        for path in tqdm.tqdm(args.input, disable=not args.output):
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output, pred, res = demo.run_on_image(img, PURE)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, "seg", args.dict, path.split('/')[-2] + '_' + os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                out_filename = out_filename.replace(".jpg", ".png")
                visualized_output.save(out_filename)
                res_dict.append({"img_path": path, "seg_path": out_filename})

                pred_out_filename = os.path.join(args.output, "pred", args.dict, path.split('/')[-2] + '_' + os.path.basename(path).replace(".png", "_pred.npy").replace(".jpg", "_pred.npy"))
                np.save(pred_out_filename, pred.numpy())
                res_out_filename = os.path.join(args.output, "res", args.dict, path.split('/')[-2] + '_' + os.path.basename(path).replace(".png", "_res.npy").replace(".jpg", "_res.npy"))
                np.save(res_out_filename, res.numpy())
                
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

        categorized_res_dict = categorize_by_resolution(res_dict)
        
        json_file_path = "%s/clean_res/%s/%s.json" % (args.output, args.dict, args.dict)
        
        # Read existing JSON file if it exists
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                existing_categorized_res_dict = json.load(json_file)
            # Merge the dictionaries
            categorized_res_dict = merge_dicts(existing_categorized_res_dict, categorized_res_dict)

        # Save the merged dictionary
        with open(json_file_path, 'w') as json_file:
            json.dump(categorized_res_dict, json_file, indent=4)
    
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()