import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.abspath(BASE_DIR))
import shutil
import pickle
import PIL.Image as Image
import argparse
import torch
import numpy as np
from utils import *
import pycolmap

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str)
parser.add_argument("--gt_path", type=str)
parser.add_argument("--split_path", type=str)
parser.add_argument("--img_base_path", type=str)
args = parser.parse_args()

gt_path = args.gt_path
split_path = args.split_path
img_base_path = args.img_base_path
sfm_dir = args.source_path + '/sparse/0/'
output_colmap_path = sfm_dir
output_gt_colmap_path = args.source_path + '/gt/sparse/0/'
img_folder_path = args.source_path + '/images'

os.makedirs(img_folder_path, exist_ok=True)
os.makedirs(output_colmap_path, exist_ok=True)
os.makedirs(output_gt_colmap_path, exist_ok=True)


# build images/ folder
train_img_list, test_img_list = read_train_test_split(split_path, img_base_path)
for img_name in train_img_list:
    src_path = os.path.join(img_base_path, img_name)
    tgt_path = os.path.join(img_folder_path, img_name)
    shutil.copy(src_path, tgt_path)

# save train set to source_path/
intr_info, extr_info = load_colmap(os.path.join(gt_path, 'sparse/0/'), img_base_path)
_, tmp_intr = next(iter(intr_info.items()))
img_size = (tmp_intr['width'], tmp_intr['height'])
intrinsics = [v["K"] for (k, v) in intr_info.items()]
poses = [extr_info[name]["pose"] for name in train_img_list]
cam_ids = [extr_info[name]['cam_id'] for name in train_img_list]
img_ids = [extr_info[name]['img_id'] for name in train_img_list]
save_colmap_cameras(intrinsics, os.path.join(output_colmap_path, 'cameras.txt'), image_size=img_size, cam_ids=list(intr_info.keys()))
save_colmap_images(np.array(poses), os.path.join(output_colmap_path, 'images.txt'), img_names=train_img_list, cam_ids=cam_ids, img_ids=img_ids)

# save gt to source_path/gt/
intr_info, extr_info = load_colmap(os.path.join(gt_path, 'sparse/0/'), img_base_path)
_, tmp_intr = next(iter(intr_info.items()))
img_size = (tmp_intr['width'], tmp_intr['height'])
gt_intrinsics = [v["K"] for (k, v) in intr_info.items()]
gt_poses = [v["pose"] for (k, v) in extr_info.items()]
gt_img_list = list(extr_info.keys())
gt_cam_ids = [extr_info[name]['cam_id'] for name in gt_img_list]
gt_img_ids = [extr_info[name]['img_id'] for name in gt_img_list]
save_colmap_cameras(gt_intrinsics, os.path.join(output_gt_colmap_path, 'cameras.txt'), image_size=img_size, cam_ids=list(intr_info.keys()))
save_colmap_images(np.array(gt_poses), os.path.join(output_gt_colmap_path, 'images.txt'), img_names=gt_img_list, cam_ids=gt_cam_ids, img_ids=gt_img_ids)
print('Created Dataset.')