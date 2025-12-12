set -e

seed=$1
round=$2
dataset_dir=$3
output_dir=$4

dataset=pascal0$((round-1))

spx_root_path=${output_dir}/demo_pascal/result/PASCAL_VOC_2012/0.2/obj_jpg/

if [ $round -eq 1 ]; then
    label_root_path=${output_dir}/demo_pascal/result/PASCAL_VOC_2012/0.2/mask_jpg/
else
    label_root_path=${output_dir}/auto_correct/result/Round$((round-1))/abc_dic/
fi

python DeepLabV3Plus_Pytorch/soft_label.py \
    --save_path ${output_dir}/soft_label/result/Round${round}/ \
    --devkit_path ${dataset_dir} \
    --dataset ${dataset} \
    --ckpt DeepLabV3Plus_Pytorch/checkpoints/best_deeplabv3plus_resnet101_${dataset}_${seed}.pth \
    --spx_root_path ${spx_root_path} \
    --label_root_path ${label_root_path} \
    --crop_size 513 
