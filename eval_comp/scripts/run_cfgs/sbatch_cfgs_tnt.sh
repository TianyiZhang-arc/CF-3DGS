#!/bin/bash
#SBATCH --job-name="run"
#SBATCH -n 1 --time=12:00:00 --mem-per-cpu=128G --gpus=rtx_3090:1
#SBATCH --output=/cluster/scratch/zhangtia/output/job_%j.out
#SBATCH --error=/cluster/scratch/zhangtia/output/job_%j.err

module load eth_proxy stack/2024-06  gcc/12.2.0 unzip/6.0-zhtq2xe p7zip/17.04

# need to change the following ROOT_DIR

JOB_SCRIPT="/cluster/home/zhangtia/projects/CF-3DGS/eval_comp/scripts/run_cfgs/run_cfgs_tnt.sh"
DATA_ROOT_DIR="/cluster/project/cvg/students/zhangtia/data/"
ENV_ROOT_DIR="/cluster/project/cvg/students/zhangtia/miniconda3/etc/profile.d/conda.sh"
CODE_ROOT_DIR="/cluster/home/zhangtia/projects/CF-3DGS/"
OUTPUT_ROOT_DIR="/cluster/scratch/zhangtia/output/"

ENV_COMMAND="source ${ENV_ROOT_DIR} && conda activate cf3dgs "
SCRIPT_COMMAND="bash ${JOB_SCRIPT} ${DATA_ROOT_DIR} ${CODE_ROOT_DIR} ${OUTPUT_ROOT_DIR} ${ENV_ROOT_DIR}"

eval $ENV_COMMAND
eval $SCRIPT_COMMAND