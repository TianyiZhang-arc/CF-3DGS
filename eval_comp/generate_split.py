import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.abspath(BASE_DIR))
import argparse
from pathlib import Path
import shutil
from glob import glob
import numpy as np

from eval_utils.io_utils import read_train_test_split, write_train_test_split

def get_args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train_views", type=int, default=-1)
    parser.add_argument("--n_test_views", type=int, default=-1)

    parser.add_argument("--img_base_path", type=str, default="")
    parser.add_argument("--split_path", type=str, default="")
    parser.add_argument("--create_image_set", action='store_true')
    return parser

def create_split_llff(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=8):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """
    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list))
    train_ids = all_indices[all_indices % hold != 0].tolist() # dense input
    test_ids = all_indices[all_indices % hold == 0].tolist()
    if n_train_views != -1:
        idx_sub = np.linspace(0, len(train_ids) - 1, n_train_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [int(train_ids[i]) for i in idx_sub]
    if n_test_views != -1 and len(test_ids) > n_test_views:
        idx_sub = np.linspace(0, len(test_ids) - 1, n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        test_ids = [int(test_ids[i]) for i in idx_sub]
    write_train_test_split(train_ids, test_ids, save_path)

def create_split_tnt(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=8):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """
    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list))
    train_ids = all_indices[all_indices % hold != 0].tolist() # dense input
    test_ids = all_indices[all_indices % hold == 0].tolist()
    if n_train_views != -1:
        idx_sub = np.linspace(0, len(train_ids) - 1, n_train_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [int(train_ids[i]) for i in idx_sub]
    if n_test_views != -1 and len(test_ids) > n_test_views:
        idx_sub = np.linspace(0, len(test_ids) - 1, n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        test_ids = [int(test_ids[i]) for i in idx_sub]
    write_train_test_split(train_ids, test_ids, save_path)

def create_split_tanks(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=8):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """
    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list))
    train_ids = all_indices[all_indices % hold != 0].tolist() # dense input
    test_ids = all_indices[all_indices % hold == 0].tolist()
    if n_train_views != -1:
        idx_sub = np.linspace(0, len(train_ids) - 1, n_train_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [int(train_ids[i]) for i in idx_sub]
    if n_test_views != -1 and len(test_ids) > n_test_views:
        idx_sub = np.linspace(0, len(test_ids) - 1, n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        test_ids = [int(test_ids[i]) for i in idx_sub]
    write_train_test_split(train_ids, test_ids, save_path)


def create_split_360(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=8):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """
    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list))
    train_ids = all_indices[all_indices % hold != 0].tolist() # dense input
    test_ids = all_indices[all_indices % hold == 0].tolist()
    if n_train_views != -1:
        idx_sub = np.linspace(0, len(train_ids) - 1, n_train_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [int(train_ids[i]) for i in idx_sub]
    if n_test_views != -1 and len(test_ids) > n_test_views:
        idx_sub = np.linspace(0, len(test_ids) - 1, n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        test_ids = [int(test_ids[i]) for i in idx_sub]
    write_train_test_split(train_ids, test_ids, save_path)  

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    os.makedirs(Path(args.split_path).parent, exist_ok=True)
    if 'llff' in args.img_base_path.lower():
        create_split_llff(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'tanks' in args.img_base_path.lower():
        create_split_tanks(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif '360' in args.img_base_path.lower():
        create_split_360(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'tnt' in args.img_base_path.lower():
        create_split_tnt(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    else:
        raise NotImplementedError("index is not preset for the dataset!")

    if args.create_image_set: # for building tracks
        save_dir = Path(args.split_path).parent / 'images'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir / 'train', exist_ok=True)
        os.makedirs(save_dir / 'test', exist_ok=True)
        img_list = sorted(os.listdir(os.path.join(args.img_base_path)))
        train_img_list, test_img_list = read_train_test_split(args.split_path, args.img_base_path)   
        for img_name in train_img_list:
            src_path = os.path.join(args.img_base_path, img_name)
            tgt_path = os.path.join(save_dir, 'train', img_name)
            shutil.copy(src_path, tgt_path)
        for img_name in test_img_list:
            src_path = os.path.join(args.img_base_path, img_name)
            tgt_path = os.path.join(save_dir, 'test', img_name)
            shutil.copy(src_path, tgt_path)