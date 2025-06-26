from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser
import yaml
import re
import time
import json
import sys
import os
import shutil
import cv2
import numpy as np
import pygame
import pygame.locals
import torch
import torch.utils
import torch.utils.data
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from LoG.dataset.base import prepare_camera
from LoG.model.level_of_gaussian import LoG
from LoG.render.renderer import NaiveRendererAndLoss
from LoG.utils.trainer import prepare_batch
from os.path import join
from tqdm import tqdm
from LoG.utils.config import load_object, Config
from LoG.utils.command import update_global_variable, load_statedict, copy_git_tracked_files

def load_list_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def demo(cfg, model, device, batch_file_basename, RGB_SAVE_DIR, NPY_SAVE_DIR):
    if cfg.split == "train":
        # When cfg.split = “train”, the scale of the rendering image size depends on the maximum value in the list of dataset.yml>dataset>args>scales. To avoid unintended downscaling, rewrite as follows.
        cfg[cfg.split].dataset.args.scales = [1]

    dataset = load_object(cfg[cfg.split].dataset.module, cfg[cfg.split].dataset.args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # prepare the renderer
    if 'render' in cfg[cfg.split]:
        renderer = load_object(cfg[cfg.split].render.module, cfg[cfg.split].render.args)
    else:
        renderer = load_object(cfg.train.render.module, cfg.train.render.args)
        renderer.split = 'demo'
    renderer.to(device)
    model.to(device)
    model.eval()
    if 'model_state' in cfg[cfg.split]:
        model.set_state(**cfg[cfg.split]['model_state'])
    if 'render_state' in cfg[cfg.split]:
        renderer.set_state(**cfg[cfg.split]['render_state'])
    from LoG.utils.trainer import prepare_batch
    from tqdm import tqdm
    total_time = 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    render_type = cfg.get('render_type', 'rgb')
    if render_type == 'depth':
        renderer.render_depth = True
        depth_min = cfg.get('depth_min', 0.01)
        depth_max = cfg.get('depth_max', 10.)
    elif render_type == 'height':
        renderer.render_depth = True
        height_min = cfg.get('height_min', 0.01)
        height_max = cfg.get('height_max', 10.)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch = prepare_batch(batch, device)
        with torch.no_grad():
            output = renderer.vis(batch, model)
        if batch_idx > 10:
            break

    print("batch_file_basename:", batch_file_basename)
    multi_cam_file = False
    if "/" in batch_file_basename:
        multi_cam_file = True

    for batch in tqdm(dataloader):
        batch = prepare_batch(batch, device)
        
        if multi_cam_file:
            file_name_eles = batch["imgname"][0].split("/")
            file_basename = "/".join(file_name_eles[-2:])
        else:
            file_basename = os.path.basename(batch["imgname"][0])

        if file_basename != batch_file_basename:
            continue
        else:
            print(file_basename)

        if 'model_state' in batch:
            model.set_state(**batch['model_state'])
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = renderer.vis(batch, model)
            end.record()
            end.synchronize()
            total_time += start.elapsed_time(end)
        render = output['render'][0]
        if render_type == 'depth':
            depth = output['depth'][0]
            depth = (depth - depth_min)/(depth_max - depth_min)
            vis = renderer.marigold_depth_vis(depth)
        elif render_type == 'height':
            depth = output['height'][0]
            print(depth.min(), depth.max(), depth.mean())
            depth = (depth - height_min)/(height_max - height_min)
            vis = renderer.marigold_depth_vis(depth)        
        else:
            vis = renderer.tensor_to_bgr(render)

        outname = os.path.join(RGB_SAVE_DIR, f'{batch["index"].item():06d}.jpg')
        cv2.imwrite(outname, vis)
        print("outname", outname)
        if cfg.split == "train":
            render_as_ae_feature = render.detach().cpu().numpy().transpose(1, 2, 0)
            npy_outname = os.path.join(NPY_SAVE_DIR, f'{batch["index"].item():05d}.npy')
            np.save(npy_outname, render_as_ae_feature)

        if 'mask' in output:
            mask = output['mask'][0].detach().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            vis = np.dstack([vis, mask[:, :, None]])
            rgbaname = os.path.join(cfg.exp, cfg.split, 'rgba', f'{batch["index"].item():06d}.png')
            os.makedirs(os.path.dirname(rgbaname), exist_ok=True)
            cv2.imwrite(rgbaname, vis)
    print("vis.shape:", vis.shape)
    print("--- complete ---")

def main(CONFIG_FILE, log_model_type):
    with open(CONFIG_FILE, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)        
    project_root = config["project_root"]
    scene_name = config["scene_info"]["scene_name"]
    CONFIG_PATH = Path(config["scene_info"]["config_path"])
    CHECKPOINT_PATH = Path(config["scene_info"]["ckpt_path"])
    COLORS_CHECKPOINT_PATH = Path(config["scene_info"]["ckpt_colors_path"])
    topdown_param = config["scene_info"]["camera_param"]["topdown"]
    arbitrary_param_list = config["scene_info"]["camera_param"]["arbitrary"]
    topdown_view_height = config["topdown_view_parameter"]["height"]
    one_side_meter = config["topdown_view_parameter"]["one_side_meter"]
    arbitrary_height = config["scene_info"]["camera_param"]["arbitrary_height"]
    arbitrary_width = config["scene_info"]["camera_param"]["arbitrary_width"]

    request_similar_area_json = config['request_similar_area_json']
    with open(request_similar_area_json, 'r') as f:
        loaded_data = json.load(f)
    query = loaded_data["query"]
    area_filter_flag = loaded_data["area_filter_flag"]
    area = loaded_data["area"]
    print(query, area_filter_flag)
    # =================================================================

    FINAL_SAVE_DIR = Path('renders4visprog/{}/'.format(scene_name))
    FINAL_SAVE_DIR.mkdir(exist_ok=True)
    RGB_SAVE_DIR = Path('renders4visprog/{}/renders_rgb/'.format(scene_name))
    if RGB_SAVE_DIR.exists():
        shutil.rmtree(RGB_SAVE_DIR)
    RGB_SAVE_DIR.mkdir(exist_ok=True)
    NPY_SAVE_DIR = Path('renders4visprog/{}/renders_npy/'.format(scene_name))
    if NPY_SAVE_DIR.exists():
        shutil.rmtree(NPY_SAVE_DIR)
    NPY_SAVE_DIR.mkdir(exist_ok=True)


    parent_path = "{}LoG/renders4visprog/".format(project_root)
    scene_path = "{}{}/".format(parent_path, scene_name)
    save_gaussians_indices_json = "{}ins_gaussians_indices.json".format(parent_path)

    ins_fname_json = "/workspace/t2024/geoprog3d/ins_fname.json"
    with open(ins_fname_json, 'r') as f:
        loaded_data = json.load(f)
    batch_file_basename = loaded_data["fname"]

    # load config
    cfg = Config.load(CONFIG_PATH)
    cfg = update_global_variable(cfg, cfg)
    device = torch.device('cuda')
    usage = 'run'
    cfg.split = 'trainvis'

    exp = cfg.exp
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    print(f'Using GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print('Write to {}'.format(exp))
    # write the parameter to the exp
    os.makedirs(exp, exist_ok=True)
    if cfg.split == 'train':
        print(cfg, file=open(os.path.join(exp, 'config.yaml'), 'w'))
    from LoG.utils.trainer import Trainer, seed_everything
    seed_everything(666)

    device = torch.device('cuda')
    model = load_object(cfg.model.module, cfg.model.args)# look level_of_gaussian.yml LoG.model.level_of_gaussian.LoG

    if cfg.split == 'trainvis':
        cfg.split = 'train'

    if log_model_type == "tree_colors":
        model.load_state_dict(load_statedict(COLORS_CHECKPOINT_PATH))
    elif log_model_type == "tree":
        model.load_state_dict(load_statedict(CHECKPOINT_PATH))
    else:
        print("--- unknown log_model_type ---")
        sys.exit()
    
    # If there is information to narrow down the indices, execute the coordinate reset for the corresponding Gaussian.
    if os.path.exists(save_gaussians_indices_json):
        xyz_array = model.gaussian.xyz.detach().cpu().numpy()# ([1206494, 3])
        in_segment_gaussians_indices = load_list_from_json(save_gaussians_indices_json)
        mask = np.ones_like(xyz_array, dtype=bool)
        mask[in_segment_gaussians_indices, :] = False
        xyz_array[mask] = 0
        model.gaussian.xyz = torch.from_numpy(xyz_array).to(device)# add

    demo(cfg, model, device, batch_file_basename, RGB_SAVE_DIR, NPY_SAVE_DIR)



if __name__ == '__main__':
    parser = ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="geoprog3d/config/config.yml")
    parser.add_argument("--log_model_type", type=str, default="tree_colors")
    args = parser.parse_args()
    # =================================================================
    # load geoprog3d config
    CONFIG_FILE = args.config
    log_model_type = args.log_model_type

    main(CONFIG_FILE, log_model_type)
