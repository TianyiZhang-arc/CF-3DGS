import glob
from PIL import Image
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/local/home/zhangtia/data/Tanks")
args = parser.parse_args()

root_path = args.root_path
datasets = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
for dataset in datasets:
    dataset_path = root_path + f'/{dataset}'
    #### get ground truth ####
    colmap_folder = dataset_path + '/colmap'
    # if os.path.exists(colmap_folder):
    #     shutil.rmtree(colmap_folder)
    os.makedirs(colmap_folder, exist_ok=True)
    print(dataset_path + '/images')
    image_path = dataset_path + '/images'
    database_path = colmap_folder + '/database.db'
    output_path = colmap_folder + '/sparse/'
    command = f"colmap feature_extractor --image_path {image_path} --database_path {database_path} \
        --ImageReader.camera_model PINHOLE --camera_mode 1 "
    os.system(command)
    command = f"colmap exhaustive_matcher --database_path {database_path}"
    os.system(command)
    os.makedirs(colmap_folder + '/sparse/0/', exist_ok=True)
    command = f"colmap mapper --Mapper.multiple_models False --image_path {image_path} --database_path {database_path} --output_path {output_path}"
    os.system(command)
    command = f"colmap model_converter --input_path {output_path + '/0'} --output_path {output_path + '/0'} --output_type TXT "
    os.system(command)
