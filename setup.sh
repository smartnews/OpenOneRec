conda init
#/mnt/data/miniconda/bin/activate openonerec
conda deactivate
conda activate openonerec
git config --global --add safe.directory /mnt/data/OpenOneRec
export HF_HOME=/mnt/data/huggingface

#git branch --set-upstream-to=origin/main main
#git branch --set-upstream-to=sn/feature/foo feature/foo
#git push -u sn feature/foo