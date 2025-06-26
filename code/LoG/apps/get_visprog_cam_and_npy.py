
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re
import time
import os
import glob
import json
import yaml
import shutil
import cv2
from argparse import ArgumentParser
import numpy as np
import pygame
import pygame.locals
import torch
import torch.utils
import torch.utils.data
from scipy.spatial.transform import Rotation as R

from LoG.dataset.base import prepare_camera
from LoG.model.level_of_gaussian import LoG
from LoG.render.renderer import NaiveRendererAndLoss
from LoG.utils.command import update_global_variable
from LoG.utils.config import Config
from LoG.utils.trainer import prepare_batch


SPLIT = 'demo_interpolate'
DEVICE = 'cuda'
IMAGE_SIZE = 2 ** 11# 2 ** 10
IMAGE_SIZE = 2 ** 9# 2 ** 10
IMAGE_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

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

    with torch.no_grad():
        output = renderer.vis(batch, model, background)
    render = output['render'][0]
    rgb = renderer.tensor_to_bgr(render)[:, ::-1, ::-1]
    render_as_ae_feature = render.detach().cpu().numpy().transpose(1, 2, 0)
    return rgb, render_as_ae_feature


def display_rgb(rgb: np.ndarray):
    surface = pygame.surfarray.make_surface(rgb)
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def render_and_display(
    renderer: NaiveRendererAndLoss,
    model: LoG,
    camera: Camera,
    background=(0.5, 0.65, 0.85),
):
    rgb, render_as_ae_feature = render_LoG(renderer, model, camera, background)
    display_rgb(rgb)
    return rgb, render_as_ae_feature



if __name__ == "__main__":

    parser = ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="geoprog3d/config/config.yml")
    args = parser.parse_args()

    # =================================================================
    # load geoprog3d config
    CONFIG_FILE = args.config

    with open(CONFIG_FILE, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)        
    scene_name = Path(config["scene_info"]["scene_name"])
    CONFIG_PATH = Path(config["scene_info"]["config_path"])
    CHECKPOINT_PATH = Path(config["scene_info"]["ckpt_path"])
    COLORS_CHECKPOINT_PATH = Path(config["scene_info"]["ckpt_colors_path"])
    topdown_param = config["scene_info"]["camera_param"]["topdown"]
    arbitrary_param_list = config["scene_info"]["camera_param"]["arbitrary"]
    filtering_gaussian = config["filtering_gaussian"]

    request_similar_area_json = config['request_similar_area_json']
    with open(request_similar_area_json, 'r') as f:
        loaded_data = json.load(f)
    query = loaded_data["query"]
    area_filter_flag = loaded_data["area_filter_flag"]
    area = loaded_data["area"]
    print(query, area_filter_flag)
    # =================================================================
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()

    OUTPUT_DIR = Path('renders4visprog/')
    OUTPUT_DIR.mkdir(exist_ok=True)

    OUTPUT_DIR = Path('captured_images/')
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
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





    # load config
    configs = Config.load(CONFIG_PATH)
    configs = update_global_variable(configs, configs)

    colors_model = LoG(**configs.model.args)
    colors_model.load_state_dict(torch.load(COLORS_CHECKPOINT_PATH)['state_dict'])
    colors_model.to(DEVICE)
    colors_model.set_state(**configs[SPLIT].model_state)

    ########################## If there is an area_filter, narrow down the Gaussians
    if area_filter_flag and filtering_gaussian:
        segment_list = area
        segment_list = np.array(segment_list)

        # Obtain the coordinates of each Gaussian image pixelization
        result = extract_parameters(topdown_param)
        x, y, z, yaw, pitch, roll, image_size, focal_length, scale, znear, zfar = result
        camera = Camera(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, image_size=image_size, focal_length=focal_length, scale=scale, znear=znear, zfar=zfar)

        xyz_array = colors_model.gaussian.xyz.detach().cpu().numpy()# ([1206494, 3])
        image_rowcols = camera.to_image_rowcol(xyz_array)

        image_rowcols_tuples = [tuple(list(map(int, row))) for row in image_rowcols]    # Convert each row of image_rowcols into a tuple
        one_thing_segment_set = set(tuple([row[1], 2048 - row[0]]) for row in segment_list)
        in_segment_gaussians_indices = [i for i, row_tuple in enumerate(image_rowcols_tuples) if row_tuple in one_thing_segment_set]

        mask = np.ones_like(xyz_array, dtype=bool)
        mask[in_segment_gaussians_indices, :] = False
        xyz_array[mask] = 0
        colors_model.gaussian.xyz = torch.from_numpy(xyz_array).to(DEVICE)
    ##########################

    # load renderer
    renderer = NaiveRendererAndLoss(**configs.train.render.args, render_depth=True)
    renderer.to(DEVICE).eval()

    # init pygame
    pygame.init()
    screen = pygame.display.set_mode(IMAGE_SIZE)
    clock = pygame.time.Clock()
    running = True

    for p in arbitrary_param_list:
        result = extract_parameters(p)
        x, y, z, yaw, pitch, roll, image_size, focal_length, scale, znear, zfar = result
        camera = Camera(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, image_size=image_size, focal_length=focal_length, scale=scale, znear=znear, zfar=zfar)

        colors_rgb, render_as_ae_feature = render_and_display(renderer, colors_model, camera)
        save_image = np.flip(colors_rgb, -1).transpose(1, 0, 2)

        exist_f = list(glob.glob(str(RGB_SAVE_DIR) + "/*"))
        exist_f_count = len(exist_f)

        save_number = "00000"[:-len(str(exist_f_count))] + str(exist_f_count)
        save_rgb_path = str(RGB_SAVE_DIR/f"{save_number}_{camera}.png")
        save_npy_path = str(NPY_SAVE_DIR/f"{save_number}.npy")
        cv2.imwrite(save_rgb_path, save_image)
        np.save(save_npy_path, render_as_ae_feature)

    pygame.quit()

    print("--- complete ---")

