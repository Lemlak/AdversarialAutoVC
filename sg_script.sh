
#!/bin/bash
#
#$ -S /bin/bash
#$ -N TDNN_resemblyzer_toggle_wgan_bigger_tdnn
#$ -o /mnt/matylda6/ibrukner/code/Bolaji_idea/new_exp/2TDNN_resemblyzer_bolaji_toggle_wgan/exp_bigger_tdnn_dropout/out.txt
#$ -e /mnt/matylda6/ibrukner/code/Bolaji_idea/new_exp/2TDNN_resemblyzer_bolaji_toggle_wgan/exp_bigger_tdnn_dropout/err.txt
#$ -l ram_free=4G,mem_free=4G,gpu=1,gpu_ram=16G,cpu=16
#$ -q long.q@@speech-gpu
#

ulimit -t 720000
script_root=/mnt/matylda6/ibrukner/code/Bolaji_idea
dest=/mnt/matylda6/ibrukner/code/Bolaji_idea/new_exp/2TDNN_resemblyzer_bolaji_toggle_wgan/exp_bigger_tdnn_dropout
script=$dest/main.py
dataset=/mnt/matylda6/ibrukner/datasets/VCTK_one_shot
output=$dest
python $script --output $output --conf $output --dataset $dataset --cont
