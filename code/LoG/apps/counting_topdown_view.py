from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re
import time
import json
import glob
import sys
import os
import shutil
import pickle
from argparse import ArgumentParser
import cv2
import numpy as np
import yaml
import pygame
import pygame.locals
import torch
import torch.utils
import torch.utils.data
import torch
import torchvision.transforms as T
from torchvision.ops import box_convert

from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
from groundingdino.util.inference import load_model, load_image, predict, annotate

from LoG.dataset.base import prepare_camera
from LoG.model.level_of_gaussian import LoG
from LoG.render.renderer import NaiveRendererAndLoss
from LoG.utils.command import update_global_variable
from LoG.utils.config import Config
from LoG.utils.trainer import prepare_batch

OUTPUT_DIR = Path('captured_images/')
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
SPLIT = 'demo_interpolate'
DEVICE = 'cuda'
IMAGE_SIZE = 2 ** 11# 2 ** 10
# IMAGE_SIZE = 2 ** 9# 2 ** 10
IMAGE_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.display.init()


def extract_parameters(input_string):
    pattern = r'x=([-\d.]+),\s*y=([-\d.]+),\s*z=([-\d.]+),\s*yaw=([-\d.]+),\s*pitch=([-\d.]+),\s*roll=([-\d.]+),\s*image_size=\(([\d]+),\s*([\d]+)\),\s*focal_length=([-\d.]+),\s*scale=([-\d.]+),\s*znear=([-\d.]+),\s*zfar=([-\d.]+)'    
    match = re.search(pattern, input_string)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        z = float(match.group(3))
        yaw = float(match.group(4))
        pitch = float(match.group(5))
        roll = float(match.group(6))
        image_size = (int(match.group(7)), int(match.group(8)))  # tuple
        focal_length = float(match.group(9))
        scale = float(match.group(10))
        znear = float(match.group(11))
        zfar = float(match.group(12))
        return x, y, z, yaw, pitch, roll, image_size, focal_length, scale, znear, zfar
    else:
        return None



@dataclass
class Camera:
    x: float = 0.
    y: float = 0.
    z: float = 0.
    yaw: float = 0.
    pitch: float = 0.
    roll: float = 0.
    image_size:tuple = IMAGE_SIZE
    focal_length: Optional[float] = None
    scale: float = 1.
    znear: float = 0.001
    zfar: float = 100.

    def __post_init__(self):
        if self.focal_length is None:
            self.focal_length = np.mean(self.image_size)
    
    @property
    def center(self):
        return (self.x, self.y, self.z)

    @property
    def center_colvec(self):
        return np.array(self.center).reshape((3, 1))
    
    @property
    def translation(self):
        return - self.rotation_matrix @ self.center_colvec

    @property
    def rotation_matrix(self):
        return (R.from_euler('XYZ', (-self.roll, -self.pitch, -self.yaw))).as_matrix()

    @property
    def intrinsic_matrix(self):
        f = self.focal_length
        h, w = self.image_size
        return np.array([
            [f, 0, w//2], 
            [0, f, h//2], 
            [0, 0, 1]
        ])
    
    @property
    def extrinsic_matrix(self):
        extrinsic = np.concatenate([self.rotation_matrix, self.translation], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
        return extrinsic
    
    def to_image_rowcol(self, world_xyz: np.ndarray) -> np.ndarray:
        n_pts, dims = world_xyz.shape
        assert world_xyz.ndim == 2 and dims == 3

        world_homogeneous_xyz = np.concatenate([world_xyz, np.ones((n_pts, 1))], axis=1)
        cam_x, cam_y, cam_z, _ = self.extrinsic_matrix @ world_homogeneous_xyz.T
        image_rowcols = (self.intrinsic_matrix @ np.stack([cam_x/cam_z, cam_y/cam_z, np.ones_like(cam_z)]))[:2][::-1].T
        return image_rowcols
    
    def as_dict(self):
        camera_params =  {
            'R': self.rotation_matrix,
            'T': self.translation,
            'K': self.intrinsic_matrix,
            'W': self.image_size[1],
            'H': self.image_size[0],
            'center': self.center_colvec,
        }
        
        return prepare_camera(camera_params, self.scale, self.znear, self.zfar)

    def update(self, pressed_keys: pygame.key.ScancodeWrapper, step=0.1):
        if pressed_keys[pygame.locals.K_UP]:
            self.x += np.cos(self.yaw) * step
            self.y += np.sin(self.yaw) * step
        if pressed_keys[pygame.locals.K_DOWN]:
            self.x -= np.cos(self.yaw) * step
            self.y -= np.sin(self.yaw) * step
        if pressed_keys[pygame.locals.K_LEFT]:
            self.x += np.sin(self.yaw) * step
            self.y -= np.cos(self.yaw) * step
        if pressed_keys[pygame.locals.K_RIGHT]:
            self.x -= np.sin(self.yaw) * step
            self.y += np.cos(self.yaw) * step
        if pressed_keys[pygame.locals.K_a]:
            self.yaw = (self.yaw - np.pi/16) % (2 * np.pi)
        if pressed_keys[pygame.locals.K_d]:
            self.yaw = (self.yaw + np.pi/16) % (2 * np.pi)
        if pressed_keys[pygame.locals.K_w]:
            self.pitch = (self.pitch + np.pi/32) % (2 * np.pi)
        if pressed_keys[pygame.locals.K_s]:
            self.pitch = (self.pitch - np.pi/32) % (2 * np.pi)
        if pressed_keys[pygame.locals.K_r]:
            self.z -= step
        if pressed_keys[pygame.locals.K_f]:
            self.z += step


def batchify(cameras, device=DEVICE):
    cameras = [
        {
            'index': i,
            'camera': cam.as_dict(),
        }
        for i, cam in enumerate(cameras)
    ]
    batch = torch.utils.data.default_collate(cameras)
    batch = prepare_batch(batch, device)
    return batch


def render_LoG(
    renderer: NaiveRendererAndLoss,
    model: LoG,
    camera: Camera,
    background,# : tuple[float, float, float]
):
    batch = batchify([camera])

    print("====")
    print(batch['camera']['camera_center'].shape)

    with torch.no_grad():
        output = renderer.vis(batch, model, background)
    render = output['render'][0]
    rgb = renderer.tensor_to_bgr(render)[:, ::-1, ::-1]

    return rgb


def render_and_display(
    renderer: NaiveRendererAndLoss,
    model: LoG,
    camera: Camera,
    background=(0.5, 0.65, 0.85),
):
    rgb = render_LoG(renderer, model, camera, background)
    return rgb

def load_list_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_list_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_grounding_dino_model(pyfile_path, weight_path):
    model = load_model(pyfile_path, weight_path)
    return model

def resize_image(image, target_size=1024):
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image


def detect_and_count_objects(image_path, query_text, model, box_threshold=0.5):

    image_source, image = load_image(image_path)
    ch, height, width = image.shape
    image_area = height * width

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=query_text,
        box_threshold=box_threshold,
        # box_threshold=0.35,
        text_threshold=0.25
    )
    boxes, logits, phrases = remove_contained_boxes(boxes, logits, phrases)

    # Drawing bounding boxes
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = np.array(annotated_frame)
    # Count the number of objects
    object_count = len(boxes)

    return annotated_frame, object_count


if __name__ == "__main__":

    parser = ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="geoprog3d/config/config.yml")
    args = parser.parse_args()

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
    topdown_small_param = config["scene_info"]["camera_param"]["topdown_small"]
    arbitrary_param_list = config["scene_info"]["camera_param"]["arbitrary"]
    topdown_view_height = config["topdown_view_parameter"]["height"]
    arbitrary_height = config["scene_info"]["camera_param"]["arbitrary_height"]
    arbitrary_width = config["scene_info"]["camera_param"]["arbitrary_width"]
    dino_pyfile_path = config["grounding_dino_param"]["pyfile"]
    dino_weight_path = config["grounding_dino_param"]["weight_path"]
    box_threshold = config["grounding_dino_param"]["box_threshold"]

    request_similar_area_json = config['request_similar_area_json']
    with open(request_similar_area_json, 'r') as f:
        loaded_data = json.load(f)
    query = loaded_data["query"]
    area_filter_flag = loaded_data["area_filter_flag"]
    area = loaded_data["area"]
    print(query, area_filter_flag)
    # =================================================================
    ins_counting_res = "/workspace/t2024/geoprog3d/ins_counting_res.jpg"
    gaussian_abs_image_size = topdown_view_height
    my_camera_image_size = arbitrary_height
    pixel_value_scaleup_rate = int(gaussian_abs_image_size / my_camera_image_size)

    parent_path = "{}LoG/renders4visprog/".format(project_root)
    scene_path = "{}{}/".format(parent_path, scene_name)
    # ins_segment_list_json = "{}similarity/heatmap_coords.json".format(scene_path)
    # save_gaussians_indices_json = "{}ins_gaussians_indices.json".format(parent_path)

    # Preparations for saving arbitrary viewpoint images for visualization
    arbitrary_save_dir = "{}arbitrary_dir/".format(scene_path)
    if os.path.exists(arbitrary_save_dir):
        shutil.rmtree(arbitrary_save_dir)
    os.makedirs(arbitrary_save_dir, exist_ok=True)
    arbitrary_img_path_json = "{}arbitrary_img_path.json".format(arbitrary_save_dir)
    arbitrary_img_dir = "{}images/".format(arbitrary_save_dir)
    os.makedirs(arbitrary_img_dir, exist_ok=True)

    # Preparations for saving the results of counting arbitrary viewpoint images
    counting_save_dir = "{}counting_dir/".format(scene_path)
    if os.path.exists(counting_save_dir):
        shutil.rmtree(counting_save_dir)
    os.makedirs(counting_save_dir, exist_ok=True)
    counting_img_path_json = "{}counting_img_path.json".format(counting_save_dir)
    counting_img_dir = "{}images/".format(counting_save_dir)
    os.makedirs(counting_img_dir, exist_ok=True)

    # load config
    configs = Config.load(CONFIG_PATH)
    configs = update_global_variable(configs, configs)

    # load model
    model = LoG(**configs.model.args)# from LoG.model.level_of_gaussian import LoG
    model.load_state_dict(torch.load(CHECKPOINT_PATH)['state_dict'])
    model.to(DEVICE)
    model.set_state(**configs[SPLIT].model_state)

    # load renderer
    renderer = NaiveRendererAndLoss(**configs.train.render.args, render_depth=True)
    renderer.to(DEVICE).eval()

    ########################## If there is an area_filter, narrow down the Gaussians
    if area_filter_flag:
        segment_list = area
        segment_list = np.array(segment_list)

        # Obtain the coordinates of each Gaussian image pixelization
        result = extract_parameters(topdown_param)
        x, y, z, yaw, pitch, roll, image_size, focal_length, scale, znear, zfar = result
        camera = Camera(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, image_size=image_size, focal_length=focal_length, scale=scale, znear=znear, zfar=zfar)

        xyz_array = model.gaussian.xyz.detach().cpu().numpy()# ([1206494, 3])
        image_rowcols = camera.to_image_rowcol(xyz_array)

        image_rowcols_tuples = [tuple(list(map(int, row))) for row in image_rowcols]    # Convert each row of image_rowcols into a tuple
        one_thing_segment_set = set(tuple([row[1], 2048 - row[0]]) for row in segment_list)
        in_topdown_segment_gaussians_indices = [i for i, row_tuple in enumerate(image_rowcols_tuples) if row_tuple in one_thing_segment_set]

        mask = np.ones_like(xyz_array, dtype=bool)
        mask[in_topdown_segment_gaussians_indices, :] = False
        xyz_array[mask] = 999999
        model.gaussian.xyz = torch.from_numpy(xyz_array).to(DEVICE)# add
    ##########################

    # Gaussian coordinates within the model
    xyz_array = model.gaussian.xyz.detach().cpu().numpy()# ([1206494, 3])

    # Camera parameters for arbitrary viewpoints
    result = extract_parameters(topdown_small_param)
    x, y, z, yaw, pitch, roll, image_size, focal_length, scale, znear, zfar = result
    camera = Camera(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, image_size=image_size, focal_length=focal_length, scale=scale, znear=znear, zfar=zfar)

    # Save an image from an arbitrary viewpoint for visualization
    rgb = render_and_display(renderer, model, camera)
    captured_image = np.flip(rgb, -1).transpose(1, 0, 2)
    captured_image_path = os.path.join(arbitrary_img_dir, "00000"[:-len(str("0"))] + str("0") + ".jpg")
    cv2.imwrite(captured_image_path, captured_image)
    arbitrary_view_image_f = captured_image_path

    # Load each arbitrary viewpoint and count.
    count_list = []
    count_res_img_list = []
    dino_model = load_grounding_dino_model(dino_pyfile_path, dino_weight_path)

    print("render_shape:", cv2.imread(arbitrary_view_image_f).shape)

    arbitrary_view_image_basename = os.path.basename(arbitrary_view_image_f)
    result_image, count = detect_and_count_objects(arbitrary_view_image_f, query, dino_model,box_threshold=box_threshold)
    res_image_path = os.path.join(counting_img_dir, arbitrary_view_image_basename)

    plt.figure(figsize=(12,12))
    plt.imshow(result_image[:,:,::-1])
    plt.axis('off')
    plt.savefig(res_image_path)
    plt.close()
    print(f"Number of {query}: {count}")

    count_list.append(count)
    count_res_img_list.append(res_image_path)

    counting_res_dict = {
        "count_list":count_list,
        "arbitrary_view_image_list":[arbitrary_view_image_f],
        "counting_result_image_list":count_res_img_list,
    }

    # Save in JSON format
    with open(counting_img_path_json, 'w') as f:
        json.dump(counting_res_dict, f)

    shutil.copy(res_image_path, ins_counting_res)
    print("--- complete ---")