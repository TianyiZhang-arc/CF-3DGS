module load eth_proxy stack/2024-06  gcc/12.2.0 unzip/6.0-zhtq2xe p7zip/17.04
srun --time=12:00:00 -A ls_polle -n 1 --mem-per-cpu=128G --gpus=rtx_3090:1 --gres=gpumem:24g --pty bash
cd /cluster/project/cvg/students/zhangtia/projects/CF-3DGS
conda create --prefix /cluster/project/cvg/students/zhangtia/env/cf3dgs python=3.10 -y
conda activate /cluster/project/cvg/students/zhangtia/env/cf3dgs
conda install -c "nvidia/label/cuda-12.4" cuda-toolkit
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt