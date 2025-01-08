import glob
from PIL import Image
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/local/home/zhangtia/data/llff")
args = parser.parse_args()

root_path = args.root_path
datasets = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
scales = [4, 8]
dataset = ['room']
for dataset in datasets:
    dataset_path = root_path + f'/{dataset}'
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

    #### get ground truth ####
    colmap_folder = dataset_path + '/colmap'
    if os.path.exists(colmap_folder):
        shutil.rmtree(colmap_folder)
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
    command = f"colmap mapper --image_path {image_path} --database_path {database_path} --output_path {output_path}"
    os.system(command)
    command = f"colmap model_converter --input_path {output_path + '/0'} --output_path {output_path + '/0'} --output_type TXT "
    os.system(command)
