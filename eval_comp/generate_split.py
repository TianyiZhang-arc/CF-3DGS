import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(os.path.abspath(BASE_DIR))
import argparse
from pathlib import Path
import shutil
from glob import glob
import numpy as np

from eval_comp.eval_utils.io_utils import read_train_test_split, write_train_test_split

def get_args_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train_views", type=int, default=-1)
    parser.add_argument("--n_test_views", type=int, default=-1)

    parser.add_argument("--img_base_path", type=str, default="")
    parser.add_argument("--split_path", type=str, default="")
    parser.add_argument("--create_image_set", action='store_true')
    return parser

def create_split_dtu(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=8):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """
    if n_train_views != -1:
        train_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_ids = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        test_ids = [int(i) for i in range(49) if i not in train_ids + exclude_ids]
        
        train_ids = train_ids[:n_train_views]
        if n_test_views != -1 and len(test_ids) > n_test_views:
            test_ids = test_ids[:n_test_views]
    else: # dense input
        img_list = sorted(os.listdir(os.path.join(img_base_path)))
        all_indices = np.arange(len(img_list)).astype(np.int32)
        train_ids = all_indices[all_indices % hold != 0]
        test_ids = all_indices[all_indices % hold == 0]
        if n_test_views != -1 and len(test_ids) > n_test_views:
            test_ids = test_ids[:n_test_views]

    write_train_test_split(train_ids, test_ids, save_path)

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

def create_split_replica(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=50):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """
    def get_replica_interval(scene):
        # if scene == 'office0':
        #     return 50, 20
        # elif scene == 'office1':
        #     return 100, 50
        # elif scene == 'office2':
        #     return 150, 10
        # elif scene == 'office3':
        #     return 350, 30
        # elif scene == 'room0':
        #     return 50, 10
        # else:
        #     return 80, 10
        return 50, 30
        
    def get_replica_partition(scene):
        # if scene == 'office0':
        #     return 0, 300
        # elif scene == 'office4':
        #     return 850, None
        # elif scene == 'room1':
        #     return 300, None
        # else:
        #     return 0, None
        return 0, None

    _parts = Path(img_base_path).parts
    scene = _parts[_parts.index('Replica') + 1]

    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list)).tolist()
    start, end = get_replica_partition(scene)
    all_indices = all_indices[start:end]
    train_interval, test_interval = get_replica_interval(scene)
    train_ids = all_indices[::train_interval]
    test_ids = [int(j) for j in all_indices if (j not in train_ids)][::test_interval]   
    # for train views
    if n_train_views != -1:
        idx_sub = np.linspace(0, len(train_ids) - 1, n_train_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [train_ids[i] for i in idx_sub]
    # for test views
    if n_test_views != -1 and len(test_ids) > n_test_views:
        idx_sub = np.linspace(0, len(test_ids) - 1, n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        test_ids = [test_ids[i] for i in idx_sub]
     
    write_train_test_split(train_ids, test_ids, save_path)

def create_split_nerfbusters(img_base_path, save_path, n_train_views=-1, n_test_views=-1):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """

    def get_nerfbusters_interval(scene):
        if scene == 'plant':
            return 15, 15
        elif scene == 'pikachu':
            return 15, 15
        elif scene == 'car':
            return 3, 3
        elif scene == 'picnic':
            return 5, 5
        else:
            raise NotImplementedError("nerfbusters interval not implemented yet!")

    _parts = Path(img_base_path).parts
    scene = _parts[_parts.index('nerfbusters') + 1]

    all_img_list = glob(os.path.join(img_base_path, '*'))
    test_img_list = glob(os.path.join(img_base_path, "frame_1_*"))
    train_img_list = [img_path for img_path in all_img_list if img_path not in test_img_list]
    test_ids = sorted([os.path.basename(img_path) for img_path in test_img_list])
    train_ids = sorted([os.path.basename(img_path) for img_path in train_img_list])
    train_interval, test_interval = get_nerfbusters_interval(scene)
    train_ids = train_ids[::train_interval]
    test_ids = [j for j in test_ids if (j not in train_ids)][::test_interval]
    # for train views
    if n_train_views != -1:
        assert n_train_views <= len(train_ids), f"n_train_views must be less than {len(train_ids)} for scene nerfbusters/{scene}"
        idx_sub = np.linspace(0, len(train_ids) - 1, n_train_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [train_ids[i] for i in idx_sub]
    # for test views
    if n_test_views != -1 and len(test_ids) > n_test_views:
        idx_sub = np.linspace(0, len(test_ids) - 1, n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        test_ids = [test_ids[i] for i in idx_sub]

    write_train_test_split(train_ids, test_ids, save_path)

def create_split_tanks(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=50):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """

    _parts = Path(img_base_path).parts
    scene = _parts[_parts.index('Tanks') + 1]

    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list)).tolist() 
    n_test_views = n_train_views
    # for train views
    if n_train_views != -1:
        idx_sub = np.linspace(0, len(all_indices) - 1, n_train_views + n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [all_indices[idx_sub[i]] for i in range(len(idx_sub)) if i % 2 == 0]
        test_ids = [all_indices[idx_sub[i]] for i in range(len(idx_sub)) if i % 2 != 0]
     
    write_train_test_split(train_ids, test_ids, save_path)

def create_split_dl3dv(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=50):
    """
    n_train_views: -1 = use all train_ids
    n_test_views: -1 = use all test_ids
    """
    def get_dl3dv_interval(scene):
        # if scene == 'office0':
        #     return 50, 20
        return 5, 5

    _parts = Path(img_base_path).parts
    scene = _parts[_parts.index('DL3DV') + 1]

    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list)).tolist()
    train_interval, test_interval = get_dl3dv_interval(scene)
    train_ids = all_indices[::train_interval]
    test_ids = [int(j) for j in all_indices if (j not in train_ids)][::test_interval]   
    # for train views
    if n_train_views != -1:
        idx_sub = np.linspace(0, len(train_ids) - 1, n_train_views)
        idx_sub = [round(i) for i in idx_sub]
        train_ids = [train_ids[i] for i in idx_sub]
    # for test views
    if n_test_views != -1 and len(test_ids) > n_test_views:
        idx_sub = np.linspace(0, len(test_ids) - 1, n_test_views)
        idx_sub = [round(i) for i in idx_sub]
        test_ids = [test_ids[i] for i in idx_sub]
    write_train_test_split(train_ids, test_ids, save_path)


def create_split_eth3d(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=8):
    # """
    # use all views for training
    # """
    # img_list = sorted(os.listdir(os.path.join(img_base_path)))
    # all_indices = np.arange(len(img_list)).tolist()
    # train_ids = all_indices
    # test_ids = []
    # write_train_test_split(train_ids, test_ids, save_path) 
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

def create_split_kitti(img_base_path, save_path, n_train_views=-1, n_test_views=-1, hold=8):
    # """
    # use all views for training
    # """
    # img_list = sorted(os.listdir(os.path.join(img_base_path)))
    # all_indices = np.arange(len(img_list)).tolist()
    # train_ids = all_indices
    # test_ids = []
    # write_train_test_split(train_ids, test_ids, save_path) 
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
    use all views for training
    """
    img_list = sorted(os.listdir(os.path.join(img_base_path)))
    all_indices = np.arange(len(img_list)).tolist()
    train_ids = all_indices
    test_ids = []
    write_train_test_split(train_ids, test_ids, save_path)    

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    os.makedirs(Path(args.split_path).parent, exist_ok=True)
    if 'dtu' in args.img_base_path.lower():
        create_split_dtu(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'llff' in args.img_base_path.lower():
        create_split_llff(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'replica' in args.img_base_path.lower():
        create_split_replica(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'nerfbusters' in args.img_base_path.lower():
        create_split_nerfbusters(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'tanks' in args.img_base_path.lower():
        create_split_tanks(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'dl3dv' in args.img_base_path.lower():
        create_split_dl3dv(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'eth3d' in args.img_base_path.lower():
        create_split_eth3d(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif '360' in args.img_base_path.lower():
        create_split_360(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
    elif 'kitti' in args.img_base_path.lower():
        create_split_kitti(args.img_base_path, args.split_path, args.n_train_views, args.n_test_views)
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