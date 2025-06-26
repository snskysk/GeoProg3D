from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image
from argparse import ArgumentParser
import yaml
import time
import json
import sys
import re
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
from sklearn.decomposition import PCA

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


def calculate_rotation_params(points):
    # Move the center of the point cloud to the origin
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Execute PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # Obtain the main component
    components = pca.components_

    # The last principal component (the direction with the smallest variance) is taken as the upward direction.
    up_vector = components[2]

    # Create a rotation matrix so that the positive z-axis points upwards.
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(up_vector, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    cos_theta = np.dot(up_vector, z_axis)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Calculate the rotation matrix using the Rodrigues rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)

    return centroid, rotation_matrix


def align_point_cloud(points, centroid, rotation_matrix):
    # Move the center of the point cloud to the origin
    centered_points = points - centroid

    # Rotate point cloud
    aligned_points = np.dot(centered_points, rotation_matrix.T)

    # Restore center
    aligned_points += centroid

    return aligned_points


def calculate_horizontal_plane(aligned_points):
    # Equation of a horizontal plane: z = ax + by + d
    # Here, the average value of the z-coordinate is taken as the horizontal plane.
    z_mean = np.mean(aligned_points[:, 2])
    return z_mean

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

    parent_path = "{}LoG/renders4visprog/".format(project_root)
    scene_path = "{}{}/".format(parent_path, scene_name)
    height_save_dir = "{}height_dir/".format(scene_path)
    if os.path.exists(height_save_dir):
        shutil.rmtree(height_save_dir)
    os.makedirs(height_save_dir, exist_ok=True)
    height_json = "{}height.json".format(height_save_dir)
    height_fig_path = "{}height_3d_fig.jpg".format(height_save_dir)

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

    ########################## If there is an area_filter, get the element number of the Gaussian in the area
    in_topdown_segment_gaussians_indices = []
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
    ##########################

    # topdown
    result = extract_parameters(topdown_param)
    x, y, z, yaw, pitch, roll, image_size, focal_length, scale, znear, zfar = result
    camera = Camera(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, image_size=image_size, focal_length=focal_length, scale=scale, znear=znear, zfar=zfar)

    # Obtain the coordinates of each Gaussian image pixelization
    xyz_array = model.gaussian.xyz.detach().cpu().numpy()# ([1206494, 3])
    image_rowcols = camera.to_image_rowcol(xyz_array)
    print("image_rowcols.shape:", image_rowcols.shape)

    y_max_idx = np.argmax(image_rowcols[:,0])
    y_min_idx = np.argmin(image_rowcols[:,0])
    image_y_diff = abs(image_rowcols[y_max_idx, 0] - image_rowcols[y_min_idx, 0])
    weight_y_diff = abs(xyz_array[y_max_idx, 0] - xyz_array[y_min_idx, 0])
    image_scale_with_weight = image_y_diff / weight_y_diff
    print("Difference (absolute value) between the pixels furthest apart in the y direction:", image_y_diff)
    print("Difference (absolute value) between the weights farthest apart in the y direction:", weight_y_diff)
    print("Weight difference*weight_y_diff = Pixel difference:", image_scale_with_weight)

    # Retrieving indices information
    parent_path = "/workspace/t2024/LoG/renders4visprog/"
    ins_gaussians_indices_json = "{}ins_gaussians_indices.json".format(parent_path)
    save_topdown_seg_list_json = "{}querys_topdown_seg_list.json".format(parent_path)

    x_array = xyz_array[:, 0]
    y_array = xyz_array[:, 1]
    z_array = xyz_array[:, 2]


    # xyz_array is a numpy array with shape=(number of points, 3)
    centroid, rotation_matrix = calculate_rotation_params(xyz_array)
    # Rotate xyz_array
    xyz_array_rotated = align_point_cloud(xyz_array, centroid, rotation_matrix)
    # Calculate the z-coordinate of the horizontal plane after rotation
    # horizontal_plane_z = calculate_horizontal_plane(xyz_array_rotated)
    # print("z-coordinate of the horizontal plane after rotation:", horizontal_plane_z)

    # search_batch. Ground search. Divide the point cloud into<groups_n> sets, and set the minimum value in the maximum value list as the ground.
    groups_n = int(len(image_rowcols) / int(one_side_meter/25))
    rotated_z_array = xyz_array_rotated[:, 2]
    search_batch_size = len(rotated_z_array) // groups_n
    max_list = []
    for l in range(groups_n):
        max_list.append(np.max(rotated_z_array[search_batch_size*l:search_batch_size*(l+1) - 1]))
    ground_z = min(max_list)
    print("z-coordinate of the horizontal plane after rotation (search):", ground_z)


    # Rotate the other point cloud, other_points, using the same parameters.
    filtered_xyz_array = xyz_array[in_topdown_segment_gaussians_indices]
    filtered_xyz_array_rotated = align_point_cloud(filtered_xyz_array, centroid, rotation_matrix)

    # points is a numpy array of shape=(number of points, 3)
    x_array = filtered_xyz_array_rotated[:, 0]
    y_array = filtered_xyz_array_rotated[:, 1]
    z_array = filtered_xyz_array_rotated[:, 2]

    z_max_idx = np.argmax(z_array)
    z_min_idx = np.argmin(z_array)
    z_diff = abs(z_array[z_max_idx] - ground_z)

    obj_height = round((z_diff*image_scale_with_weight)*(one_side_meter/topdown_view_height), 2)
    print("obj_height: ", obj_height)


    highest_point = x_array[z_max_idx], y_array[z_max_idx], z_array[z_max_idx]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the entire point cloud
    ax.scatter(x_array, y_array, z_array, s=0.01, alpha=0.1, c="black")
    # Plot the vertex
    ax.scatter(*highest_point, s=80, c="red", label="Highest Point")
    # Plot a horizontal line segment
    horizontal_point = (highest_point[0], highest_point[1], ground_z)
    ax.plot([highest_point[0], horizontal_point[0]],
            [highest_point[1], horizontal_point[1]],
            [highest_point[2], horizontal_point[2]],
            color='green', linewidth=4, label='Height Difference')

    # Plot points on a horizontal plane
    ax.scatter(*horizontal_point, s=80, c="blue", label="Horizontal Plane Point")

    # Graph settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.set_xlim3d(-2,2)
    # ax.set_ylim3d(0, 3.5)
    # ax.set_zlim3d(2.2, 3.75)
    # plt.axis('off')

    ax.view_init(elev=10, azim=60)
    plt.tight_layout()
    plt.savefig(height_fig_path)
    plt.close()


    height_res_dict = {
        "obj_height":obj_height,
        "height_fig_path":height_fig_path
    }

    # Save in JSON format
    with open(height_json, 'w') as f:
        json.dump(height_res_dict, f)

    print("--- complete ---")