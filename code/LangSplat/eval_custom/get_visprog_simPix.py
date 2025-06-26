#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
from scipy import stats
import torch
import time
from tqdm import tqdm
import shutil
import yaml

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def activate_stream(sem_map, 
                    clip_model, 
                    image_name: Path = None,
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    heatmap_path_list = []
    for k in range(n_prompt):
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            heatmap_path_list.append(str(output_path_relev) + ".png")
            break
            
    return heatmap_path_list


def evaluate(feat_dir, output_path, ae_ckpt_path, mask_thresh, encoder_hidden_dims, decoder_hidden_dims, eval_index_list, prompt, image_shape, logger, adjust_gs_axis):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )


    # eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    # import pdb;pdb.pdb.set_trace()
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        for j, idx in enumerate(eval_index_list):
            if adjust_gs_axis:
                npy_loaded = np.load(feat_paths_lvl[idx]).transpose(1,0,2)
                npy_loaded = npy_loaded[::-1, :, :]
                print("npy_loaded.shape:", npy_loaded.shape)
                print("image_shape:", image_shape)
                if (npy_loaded.shape[0], npy_loaded.shape[1]) != image_shape:
                    npy_loaded = cv2.resize(npy_loaded, (image_shape[1],image_shape[0]))
                    print("resized npy_loaded.shape:", npy_loaded.shape)
                compressed_sem_feats[i][j] = npy_loaded
            else:
                compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    all_heatmap_path_list = []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512
        
        clip_model.set_positives([prompt])
        
        heatmap_path_list = activate_stream(restored_feat, clip_model, image_name,
                                            thresh=mask_thresh, colormap_options=colormap_options)
        all_heatmap_path_list.extend(heatmap_path_list)
    return all_heatmap_path_list

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    parser = ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="geoprog3d/config/config.yml")
    parser.add_argument("--ins_fname_json", type=str, default=None)
    args = parser.parse_args()
    ins_fname_json = args.ins_fname_json
    # =================================================================
    # load geoprog3d config
    CONFIG_FILE = args.config

    with open(CONFIG_FILE, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)        
    project_root = config["project_root"]
    scene_name = config["scene_info"]["scene_name"]
    CONFIG_PATH = Path(config["scene_info"]["config_path"])
    CHECKPOINT_PATH = Path(config["scene_info"]["ckpt_path"])
    COLORS_CHECKPOINT_PATH = Path(config["scene_info"]["ckpt_colors_path"])
    topdown_param = config["scene_info"]["camera_param"]["topdown"]
    arbitrary_param_list = config["scene_info"]["camera_param"]["arbitrary"]

    mask_thresh = config["get_sim_param"]["mask_thresh"]
    heatmap_thresh = config["get_sim_param"]["heatmap_thresh"]
    adjust_gs_axis = config["get_sim_param"]["adjust_gs_axis"]
    eval_index_list = config["scene_info"]["camera_param"]["eval_index_list"]
    height = config["scene_info"]["camera_param"]["arbitrary_height"]
    width = config["scene_info"]["camera_param"]["arbitrary_width"]
    encoder_dims = config["get_sim_param"]["ae_encoder_dims"]
    decoder_dims = config["get_sim_param"]["ae_decoder_dims"]
    ae_ckpt_path = config["get_sim_param"]["ae_ckpt_path"]

    request_similar_area_json = config['request_similar_area_json']
    with open(request_similar_area_json, 'r') as f:
        loaded_data = json.load(f)
    query = loaded_data["query"]
    area_filter_flag = loaded_data["area_filter_flag"]
    area = loaded_data["area"]
    print(query, area_filter_flag)
    # =================================================================

    dataset_name = scene_name
    output_path = "{}LoG/renders4visprog/{}/similarity".format(project_root, dataset_name)
    root_path="{}LangSplat/".format(project_root)
    # ae_ckpt_path = os.path.join(f"{root_path}ckpts", dataset_name, "best_ckpt.pth")

    prompt = query.replace(" ", "_")
    if ins_fname_json != None:
        with open(ins_fname_json, 'r') as f:
            loaded_data = json.load(f)
        height = loaded_data["height"]
        width = loaded_data["width"]
        eval_index_list = loaded_data["eval_index_list"]

    image_shape = [width, height]
    feat_dir = "{}LoG/renders4visprog/{}/renders_npy".format(project_root, dataset_name)



    feat_dir = [feat_dir, feat_dir]

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    all_heatmap_path_list = evaluate(feat_dir, output_path, ae_ckpt_path, mask_thresh, encoder_dims, decoder_dims, eval_index_list, prompt, image_shape, logger, adjust_gs_axis)

    THRESHOLD = heatmap_thresh

    result_dict = {}

    heatmap_path_dict = {}
    heatmap_path_dict["heatmap_path_list"] = all_heatmap_path_list
    heatmap_info_json_path = os.path.join(output_path, 'heatmap_list.json')
    with open(heatmap_info_json_path, 'w') as f:
        json.dump(heatmap_path_dict, f)

    mask_dir = os.path.join(output_path, 'mask_images')
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir, exist_ok=True)

    for image_path in all_heatmap_path_list:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = img >= THRESHOLD
        mask = mask.astype(np.uint8) * 255

        # Get the outer edge pixels of an image
        edge_pixels = np.concatenate([
            mask[0, :],  # Top edges
            mask[-1, :],  # bottom edges
            mask[:, 0],  # Left edges
            mask[:, -1]  # right edges
        ])

        # Calculate the most frequent value of the outer edge pixels
        mode_color = stats.mode(edge_pixels)[0][0]

        # If the most frequent value is 255 (white), the image is inverted.
        if mode_color == 255:
            mask = cv2.bitwise_not(mask)

        # For confirmation, output background color
        print(f"Background color: {'white' if mode_color == 255 else 'black'}")

        # Save the mask image
        mask_filename = os.path.join(mask_dir, f"mask_{os.path.basename(image_path)}")
        cv2.imwrite(mask_filename, mask)
        
        # Obtain coordinates
        coords = np.argwhere(mask)
        
        # Convert to a list of x, y coordinates
        coords_list = coords[:, [1, 0]].tolist()
        
        # Add the results to the dictionary
        result_dict[image_path] = coords_list

    json_path = os.path.join(output_path, 'heatmap_coords.json')

    with open(json_path, 'w') as f:
        json.dump(result_dict, f)


    with open(json_path, 'r') as f:
        loaded_data = json.load(f)


    print("Number of images processed:", len(loaded_data))
    for image_path, coords in list(loaded_data.items())[:3]:
        print(f"Image: {image_path}")
        print(f"Number of coordinates: {len(coords)}")
        print(f"First 5 coordinates: {coords[:5]}")
        print(f"Mask image saved as: mask_{os.path.basename(image_path)}")
        print()
    print("--- complete ---")