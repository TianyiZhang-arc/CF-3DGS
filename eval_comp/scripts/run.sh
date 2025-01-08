#!bin/bash

ROOT_DIR=/local/home/zhangtia
CONDA_ROOT_DIR="/local/home/zhangtia/miniconda3/etc/profile.d/conda.sh"

CMD_ENV="source ${CONDA_ROOT_DIR} && conda activate gs-geo"
CODE=${ROOT_DIR}/projects/CF-3DGS/
cmd1="python ${CODE}/eval_comp/scripts/preprocess_llff.py"
cmd2="bash ${CODE}/eval_comp/scripts/run_cfgs_llff.sh"
cmd3="bash ${CODE}/eval_comp/scripts/run_cfgs_tnt.sh"

eval $CMD_ENV
eval $cmd1
eval $cmd2
eval $cmd3

