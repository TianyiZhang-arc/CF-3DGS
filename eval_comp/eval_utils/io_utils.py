import os
import numpy as np
import json
from PIL import Image
from eval_utils.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text, rotmat2qvec, qvec2rotmat

def read_train_test_split(split_path, img_base_path=None):
    with open(split_path, 'r') as f:
        train_test_split = json.load(f)
    if isinstance(train_test_split['train'][0], int):
        train_ids = train_test_split['train']
        test_ids = train_test_split['test']
        img_list = sorted(os.listdir(img_base_path))
        train_img_list = [img_list[i] for i in train_ids]
        test_img_list = [img_list[i] for i in test_ids]
    elif isinstance(train_test_split['train'][0], str):
        train_img_list = train_test_split['train']
        test_img_list = train_test_split['test']
    else:
        raise ValueError('Error in reading train test split!')
    return train_img_list, test_img_list

def write_train_test_split(train_ids, test_ids, split_path):
    train_test_split = {}
    train_test_split['train'] = train_ids
    train_test_split['test'] = test_ids
    with open(split_path, 'w') as f:
        json.dump(train_test_split, f, indent=4)

def resize_intrinsics(K: np.ndarray, img_size: tuple):
    """
    K: (3, 3)
    img_size: (W, H)
    """
    W, H = img_size
    scale_factor_x = W/2  / K[0, 2]
    scale_factor_y = H/2  / K[1, 2]
    # print(f'scale factor is not same for x{scale_factor_x} and y {scale_factor_y}')
    # K[0, 0] *= scale_factor_x
    # K[1, 1] *= scale_factor_y
    # K[0, 2] = W/2
    # K[1, 2] = H/2
    K[0, :] *= scale_factor_x
    K[1, :] *= scale_factor_y
    return K

def save_colmap_cameras(intrinsics, camera_file, image_size=None, cam_ids=None, resize_intr=True):
    with open(camera_file, 'w') as f:
        W, H = image_size
        for i, K in enumerate(intrinsics):
            if resize_intr and image_size is not None:
                delta_thres = 0.05
                delta_x = abs(W/2  - K[0, 2]) / (W/2)
                delta_y = abs(H/2  - K[1, 2]) / (H/2)
                if delta_x > delta_thres and delta_y > delta_thres:
                    print(f"Resize intrinsics when saving")
                    K = resize_intrinsics(K, (W, H))
            if cam_ids is not None:
                f.write(f"{cam_ids[i]} PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
            else:
                f.write(f"{i+1} PINHOLE {W} {H} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")

def save_colmap_images(poses, images_file, img_names, cam_ids, img_ids=None): # input pose is c2w
    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, pose in enumerate(poses):
            R = pose[:3, :3]
            t = pose[:3, 3]
            q = rotmat2qvec(R)  # Convert rotation matrix to quaternion
            if img_ids is not None:
                f.write(f"{img_ids[i]} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_ids[i]} {img_names[i]}\n")
            else:
                f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_ids[i]} {img_names[i]}\n")
            f.write(f"\n")

def load_colmap(path: str, img_base_path: str = None):
    """
    return: 
        output_extr_dict: {'img_name': {'pose': w2c, 'cam_id': cam_id}}
        output_intr_dict: {'img_name': {'K': K, 'width': camera.width, 'height': camera.height}}
    """
    try:
        cameras_extrinsic_file = os.path.join(path, "images.bin")
        cameras_intrinsic_file = os.path.join(path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "images.txt")
        cameras_intrinsic_file = os.path.join(path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    output_extr_dict = {}
    if isinstance(cam_extrinsics, dict):
        for image_id, image in cam_extrinsics.items():
            name = image.name
            qvec = image.qvec
            tvec = image.tvec
            cam_id = image.camera_id
            w2c = np.eye(4)
            w2c[:3, :3] = qvec2rotmat(qvec)
            w2c[:3, -1] = tvec.reshape(-1)
            output_extr_dict[name] = {'pose': w2c, 'cam_id': cam_id, 'img_id':image_id}
    elif isinstance(cam_extrinsics, list):
        for image_id, image in enumerate(cam_extrinsics):
            name = image.name
            qvec = image.qvec
            tvec = image.tvec
            cam_id = image.camera_id
            w2c = np.eye(4)
            w2c[:3, :3] = qvec2rotmat(qvec)
            w2c[:3, -1] = tvec.reshape(-1)
            output_extr_dict[name] = {'pose': w2c, 'cam_id': cam_id, 'img_id':image_id}
    
    output_intr_dict = {}
    if img_base_path is not None:
        _img = Image.open(os.path.join(img_base_path, os.listdir(img_base_path)[0]))
    if isinstance(cam_intrinsics, dict):
        for cam_id, camera in cam_intrinsics.items():
            width, height = camera.width, camera.height
            if camera.model == "PINHOLE":
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
                K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
            elif camera.model == 'SIMPLE_PINHOLE':
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[0]
                K[0, 2], K[1, 2] = camera.params[1], camera.params[2]
            elif camera.model == 'SIMPLE_RADIAL':
                print('Camera model is SIMPLE_RADIAL! Not considering distortion')
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[0]
                K[0, 2], K[1, 2] = camera.params[1], camera.params[2]
            elif camera.model == 'OPENCV':
                print('Camera model is OPENCV! Not considering distortion')
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
                K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
            else:
                raise NotImplementedError('Must be PINHOLE or SIMPLE_PINHOLE')
            if img_base_path is not None:
                K[0, :] *= _img.width / camera.width
                K[1, :] *= _img.height / camera.height
                width, height = _img.width, _img.height
            output_intr_dict[cam_id] = {'K': K, 'width': width, 'height': height}
    elif isinstance(cam_extrinsics, list):
        for cam_id, camera in enumerate(cam_intrinsics):
            width, height = camera.width, camera.height
            if camera.model == "PINHOLE":
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
                K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
            elif camera.model == 'SIMPLE_PINHOLE':
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[0]
                K[0, 2], K[1, 2] = camera.params[1], camera.params[2]
            elif camera.model == 'SIMPLE_RADIAL':
                print('Camera model is SIMPLE_RADIAL! Not considering distortion')
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[0]
                K[0, 2], K[1, 2] = camera.params[1], camera.params[2]
            elif camera.model == 'OPENCV':
                print('Camera model is OPENCV! Not considering the distortion')
                K = np.eye(3)
                K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
                K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
            else:
                raise NotImplementedError('Must be PINHOLE or SIMPLE_PINHOLE')
            if img_base_path is not None:
                K[0, :] *= _img.width / camera.width
                K[1, :] *= _img.height / camera.height
                width, height = _img.width, _img.height
            output_intr_dict[cam_id] = {'K': K, 'width': camera.width, 'height': camera.height}

    return output_intr_dict, output_extr_dict