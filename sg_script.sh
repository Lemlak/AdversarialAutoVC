
#!/bin/bash
#
#$ -S /bin/bash
#$ -N AutoVC_TDNN_32x8_adaptive_toggle_wgan_aug_crit_1
#$ -o # out path for sge
#$ -e # err path for sge
#$ -l ram_free=8G,mem_free=8G,gpu=1,gpu_ram=8G,cpu=32
#$ -q # queue for sge
#

script=main.py
dataset=#dataset path
output=#output path
# config name is fixed, set directory of experiment
python $script --output $output --conf ./ --dataset $dataset #--cont
