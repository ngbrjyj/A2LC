set -e

round=$1
dataset_dir=$2
output_dir=$3

dataset=cityscapes0$((round-1))

soft_label_path=${output_dir}/soft_label/result/Round${round}/

spx_root_path=${output_dir}/demo_pascal/result/Cityscapes/0.2/obj_jpg/

if [ $round -eq 1 ]; then
    n_label_root_path=${output_dir}/demo_pascal/result/Cityscapes/0.2/mask_jpg/   
else
    n_label_root_path=${output_dir}/auto_correct/result/Round$((round-1))/abc_dic/    
fi

python DeepLabV3Plus_Pytorch/acq_on_sam.py \
    --save_path ${output_dir}/acq_on_sam/result/Round${round}/ \
    --ngbr_dir ${output_dir}/acq_on_sam/ngbr/Round${round}/ \
    --round ${round} \
    --dataset_dir ${dataset_dir} \
    --dataset ${dataset} \
    --soft_label_path ${soft_label_path} \
    --spx_root_path ${spx_root_path} \
    --n_label_root_path ${n_label_root_path} \
    --sel_hist_fdr ${output_dir}/gen_masks_with_acq/ngbr/