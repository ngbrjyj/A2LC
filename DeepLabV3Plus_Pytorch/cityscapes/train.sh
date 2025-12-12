set -e

seed=$1
round=$2
dataset_dir=$3
output_dir=$4

dataset=cityscapes0${round}

if [ $round -eq 0 ]; then
    mask_dir=${output_dir}/demo_pascal/result/Cityscapes/0.2/mask_jpg
else
    mask_dir=${output_dir}/auto_correct/result/Round${round}/abc_dic            
fi

python DeepLabV3Plus_Pytorch/main.py \
    --random_seed ${seed} \
    --data_root ${dataset_dir} \
    --dataset ${dataset} \
    --mask_dir ${mask_dir}