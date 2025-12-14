set -e

seed=$1
round=$2
dataset_dir=$3
output_dir=$4

dataset=pascal0${round}

ckpt_path=\
DeepLabV3Plus_Pytorch/checkpoints/best_deeplabv3plus_resnet101_pascal0$((round - 1))_${seed}.pth

spx_root_path=${output_dir}/demo_pascal/result/PASCAL_VOC_2012/0.2/obj_jpg/

label_root_path=${output_dir}/gen_masks_with_acq/result/Round${round}/abc_dic/
save_path=${output_dir}/auto_correct/result/Round${round}/
if [ ! -d "${save_path}" ]; then
    mkdir -p "${save_path}"
fi
cp -r "${label_root_path}" "${save_path}"
label_root_path="${save_path}abc_dic/"

python DeepLabV3Plus_Pytorch/auto_correct.py \
    --save_path ${save_path} \
    --round ${round} \
    --devkit_path ${dataset_dir} \
    --dataset ${dataset} \
    --ckpt ${ckpt_path} \
    --spx_root_path ${spx_root_path} \
    --label_root_path ${label_root_path} \
    --gen_mid_path ${output_dir}/gen_masks_with_acq/mid