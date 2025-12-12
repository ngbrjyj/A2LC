set -e

seed=$1
round=$2
dataset_dir=$3
output_dir=$4

dataset=pascal0${round}

train_imageset_path=${dataset_dir}/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt

if [ $round -eq 0 ]; then
    mask_dir=${output_dir}/demo_pascal/result/PASCAL_VOC_2012/0.2/mask_jpg
else
    mask_dir=${output_dir}/auto_correct/result/${category}/Round${round}/abc_dic
fi

python DeepLabV3Plus_Pytorch/main.py \
    --random_seed ${seed} \
    --data_root ${dataset_dir} \
    --dataset ${dataset} \
    --mask_dir ${mask_dir} \
    --train_imageset_path ${train_imageset_path} \
    --crop_size 513 \
    --batch_size 16 \
    --lr 0.1