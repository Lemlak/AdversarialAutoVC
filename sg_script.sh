
#!/bin/bash
#
#$ -S /bin/bash
#$ -N AutoVC_TDNN_32x8_adaptive_toggle_wgan_aug_crit_1
#$ -o /mnt/matylda6/ibrukner/code/AutoVC_TDNN/adaptive_toggle_wgan_aug/AutoVC_TDNN_32x8_adaptive_toggle_wgan_aug_crit_1_1/out.txt
#$ -e /mnt/matylda6/ibrukner/code/AutoVC_TDNN/adaptive_toggle_wgan_aug/AutoVC_TDNN_32x8_adaptive_toggle_wgan_aug_crit_1_1/err.txt
#$ -l ram_free=8G,mem_free=8G,gpu=1,gpu_ram=8G,cpu=32
#$ -q long.q@@speech-gpu
#

ulimit -t 720000

dest=/mnt/matylda6/ibrukner/code/AutoVC_TDNN/adaptive_toggle_wgan_aug/AutoVC_TDNN_32x8_adaptive_toggle_wgan_aug_crit_1_1
script=$dest/main.py
dataset=/mnt/matylda6/ibrukner/datasets/VCTK_one_shot
output=$dest
python $script --output $output --conf $output --dataset $dataset #--cont
