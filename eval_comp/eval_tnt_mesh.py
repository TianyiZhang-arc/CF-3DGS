import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m" "--model_path", type=str)
parser.add_argument("--iteration", type=int)
parser.add_argument("--gt_path", type=str)
parser.add_argument("--split_path", type=str)
parser.add_argument("--img_base_path", type=str)
args = parser.parse_args()


cmd = f"python extract_mesh.py -m {args.model_path} --iteration {args.iteration}"
os.system(cmd)