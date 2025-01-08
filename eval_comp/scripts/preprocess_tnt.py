import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.abspath(BASE_DIR))
import glob
from PIL import Image
import shutil
import argparse
import numpy as np
from eval_utils.io_utils import save_colmap_images, save_colmap_cameras

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/local/home/zhangtia/data/tnt")
parser.add_argument("--root_gt_path", type=str, default="/local/home/zhangtia/data/tnt_eval")
args = parser.parse_args()

def read_tnt_poses(path):
    pose_dict = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                if len(elems) == 3:
                    pose = np.empty((4, 4))
                    id = map(int, elems[0])
                    for i in range(4):
                        pose[i] = np.array(tuple(map(float, fid.readline().split())))
                    pose_dict[id] = pose
    return pose_dict

root_path = args.root_path
root_gt_path = args.root_gt_path
datasets = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
datasets.remove('Church') # remove church because the pose and image numbers are different
scales = [4, 8]
for dataset in datasets:
    dataset_path = root_path + f'/{dataset}'
    os.makedirs(dataset_path + f'/images', exist_ok=True)
    #### resize ####
    for scale in scales:
        source_folder = dataset_path + '/images'  # Change this to your source folder path
        target_folder = dataset_path + f'/images_{scale}'  # Change this to your target folder path
        # Ensure the target folder exists
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        os.makedirs(target_folder, exist_ok=True)
        image_files = glob.glob(f"{source_folder}/*.*")  
        for image_file in image_files:
            try:
                with Image.open(image_file) as img:
                    target_resolution = (int(img.width / scale), int(img.height / scale))  # (W, H)
                    resized_img = img.resize(target_resolution)
                    
                    base_name = os.path.basename(image_file)
                    save_path = os.path.join(target_folder, base_name)
                    resized_img.save(save_path)

                    print(f"Resized and saved: {save_path}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    # #### get ground truth ####
    # colmap_folder = dataset_path + '/colmap'
    # if os.path.exists(colmap_folder):
    #     shutil.rmtree(colmap_folder)
    # os.makedirs(colmap_folder, exist_ok=True)
    # print(dataset_path + '/images')
    # image_path = dataset_path + '/images'
    # database_path = colmap_folder + '/database.db'
    # output_path = colmap_folder + '/sparse/'
    # command = f"colmap feature_extractor --image_path {image_path} --database_path {database_path} \
    #     --ImageReader.camera_model PINHOLE --camera_mode 1 "
    # os.system(command)
    # command = f"colmap exhaustive_matcher --database_path {database_path}"
    # os.system(command)
    # os.makedirs(colmap_folder + '/sparse/0/', exist_ok=True)
    # command = f"colmap mapper --image_path {image_path} --database_path {database_path} --output_path {output_path}"
    # os.system(command)
    # command = f"colmap model_converter --input_path {output_path + '/0'} --output_path {output_path + '/0'} --output_type TXT "
    # os.system(command)
    
    # convert ground truth from tnt_eval
    colmap_folder = dataset_path + '/colmap'
    if os.path.exists(colmap_folder):
        shutil.rmtree(colmap_folder)
    os.makedirs(colmap_folder, exist_ok=True)
    image_path = dataset_path + '/images'
    gt_dataset_path = root_gt_path + f'/{dataset}'
    # convert the poses
    gt_poses = read_tnt_poses(gt_dataset_path + f'/{dataset}_COLMAP_SfM.log') # c2w
    gt_poses = [np.linalg.inv(v) for k, v in gt_poses.items()] # w2c
    # gt intrinsics
    image_files = glob.glob(f"{image_path}/*.*")
    img_list = sorted([os.path.basename(f) for f in image_files])  
    _img = Image.open(image_files[0])
    W, H = _img.width, _img.height
    focal = 0.7 * _img.width
    print('focal:', focal)
    intr = np.eye(3)
    intr[0, 0] = focal
    intr[1, 1] = focal
    intr[0, 2] = _img.width / 2
    intr[1, 2] = _img.height / 2
    intrinsics = [intr]
    output_path = colmap_folder + '/sparse/0/'
    os.makedirs(output_path, exist_ok=True)
    cam_ids = [1]
    save_colmap_cameras(intrinsics, os.path.join(output_path, 'cameras.txt'), image_size=(W, H), cam_ids=cam_ids)
    img_ids = list(range(1, len(img_list)+1))
    cam_ids = [1 for img in img_list]
    save_colmap_images(gt_poses, os.path.join(output_path, 'images.txt'), img_names=img_list, cam_ids=cam_ids, img_ids=img_ids)

