# uv environment
export UV_HOME=/mnt/data/uv

cd /mnt/data/OpenOneRec
source .venv/bin/activate
python --version

#export UV_HOME=/mnt/data/uv
export PATH=/mnt/data/bin:$PATH

# OpenOneRec setup

#source /mnt/data/miniconda/bin/activate openonerec
#conda init
#conda deactivate
#conda activate openonerec
git config --global --add safe.directory /mnt/data/OpenOneRec
export HF_HOME=/mnt/data/huggingface
export PYTHONPATH=$(pwd)
#git branch --set-upstream-to=origin/main main
#git branch --set-upstream-to=sn/feature/foo feature/foo
#git push -u sn feature/foo



# moved to navi_setup.sh
#apt-get install -y openmpi-bin libopenmpi-dev numactl
#apt-get install -y libomp5
## CUDA nvcc install for Ubuntu 20.04
#wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
#dpkg -i cuda-keyring_1.1-1_all.deb
#apt-get install -y cuda-nvcc-12-4

# nvcc setup

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
