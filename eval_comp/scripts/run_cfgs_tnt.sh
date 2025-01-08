#! /bin/bash

if [ -n "$1" ]; then
    DATA_ROOT_DIR=$1
else
    # DATA_ROOT_DIR="/cluster/project/cvg/students/zhangtia/data/"
    DATA_ROOT_DIR="/local/home/zhangtia/data/"
fi

if [ -n "$2" ]; then
    CODE_ROOT_DIR=$2
else
    # CODE_ROOT_DIR="/cluster/project/cvg/students/zhangtia/projects/CF-3DGS/"
    CODE_ROOT_DIR="/local/home/zhangtia/projects/CF-3DGS/"
fi 

if [ -n "$3" ]; then
    OUTPUT_ROOT_DIR=$3
else
    # OUTPUT_ROOT_DIR="/cluster/project/cvg/students/zhangtia/projects/CF-3DGS/"
    OUTPUT_ROOT_DIR="/local/home/zhangtia/projects/CF-3DGS/"
fi

if [ -n "$4" ]; then
    CONDA_ROOT_DIR=$4
else
    CONDA_ROOT_DIR="/local/home/zhangtia/miniconda3/etc/profile.d/conda.sh"
fi

SOURCE_ROOT_DIR=${CODE_ROOT_DIR}/data/
MODEL_ROOT_DIR=${OUTPUT_ROOT_DIR}/output/
IMG_DIR="images_4/"

DATASETS=(
    tnt
    )

SCENES=(
    Barn
    Ignatius
    Truck
    Meetingroom
    Courthouse
    Caterpillar
    )

N_VIEWS=(
    -1
    )

max_test_view=-1
gs_render_iter=30000

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            # Absolute paths for datasets
            DATA_DIR=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
            TNT_EVAL_PATH=${DATA_ROOT_DIR}/tnt_eval/${SCENE}/ # only for tnt
            IMG_BASE_PATH=${DATA_DIR}/${IMG_DIR}/
            GT_PATH=${DATA_DIR}/colmap/
            SOURCE_DIR=${SOURCE_ROOT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views/
            SPLIT_PATH=${SOURCE_DIR}/train_test_split.json
            IMG_PATH=${SOURCE_DIR}/images/train/
            SOURCE_PATH=${SOURCE_DIR}/sfm/
            # Absolute paths for results
            MODEL_PATH=${MODEL_ROOT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views/
            TEST_IMG_PATH=${SOURCE_DIR}/images/test/

            # ----- conda envs -----
            CMD_ENV_EVAL="source ${CONDA_ROOT_DIR} && conda init && conda activate gs-geo"
            CMD_ENV_TRAIN="source ${CONDA_ROOT_DIR} && conda init && conda activate cf3dgs"
            CMD_ENV_MESH="source ${CONDA_ROOT_DIR} && conda init && conda activate gof"

            # ----- train test split -----
            CMD_S="python ${CODE_ROOT_DIR}/eval_comp/generate_split.py \
            --n_train_views ${N_VIEW}  \
            --n_test_views ${max_test_view}  \
            --img_base_path ${IMG_BASE_PATH} \
            --split_path ${SPLIT_PATH} \
            --create_image_set
            "

            CMD_D="python ${CODE_ROOT_DIR}/eval_comp/create_dataset.py \
            --source_path ${SOURCE_PATH}/ \
            --gt_path ${GT_PATH} \
            --img_base_path ${IMG_BASE_PATH} \
            --split_path ${SPLIT_PATH} \
            "

            # ---- run CF-3DGS ----
            CMD_T="python run_cf3dgs.py \
            -s ${SOURCE_PATH} \
            --expname ${MODEL_PATH} \
            --mode train \
            --data_type colmap"

            # ----- evaluation -----
            CMD_C="python ${CODE_ROOT_DIR}/eval_comp/scripts/convert_cf3dgs.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --iteration ${gs_render_iter}"

            CMD_R="python ${CODE_ROOT_DIR}/eval_comp/render.py \
            --eval \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH} \
            --iteration ${gs_render_iter} \
            --optim_test_pose_iter 500 \
            --optim_test_pose \
            "

            CMD_M="python ${CODE_ROOT_DIR}/eval_comp/metrics.py \
            -m ${MODEL_PATH} \
            "
            
            # TODO
            CMD_G="python ${CODE_ROOT_DIR}/eval_comp/eval_tnt_mesh.py \
            --gt_path ${TNT_EVAL_PATH} \
            --ply_path ${PCD_PATH} \
            --pose_path ${POSE_PATH} \
            "
            
            echo "========= ${SCENE}: Create Dataset ========="
            eval $CMD_S
            eval $CMD_D
            echo "========= ${SCENE}: Train ========="
            eval $CMD_T
            echo "========= ${SCENE}: Convert Results ========="
            eval $CMD_C
            echo "========= ${SCENE}: Render ========="
            eval $CMD_ENV_EVAL
            eval $CMD_R
            echo "========= ${SCENE}: Metric ========="
            eval $CMD_M
            # echo "========= ${SCENE}: Mesh ========="
            # if [ "$DATASET" = "tnt" ]; then
            #     echo $CMD_G
            # else
            #     eval "Skip mesh evaluation."
            # fi
        done
    done
done