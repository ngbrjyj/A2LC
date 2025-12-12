set -e

round=$1
dataset_dir=$2
output_dir=$3
budget=$4

dataset=pascal0$((round-1))

acq_on_sam_path=${output_dir}/acq_on_sam/result/Round${round}/

spx_root_path=${output_dir}/demo_pascal/result/PASCAL_VOC_2012/0.2/obj_jpg/

if [ $round -eq 1 ]; then
    n_label_root_path=${output_dir}/demo_pascal/result/PASCAL_VOC_2012/0.2/mask_jpg/
else
    n_label_root_path=${output_dir}/auto_correct/result/Round$((round-1))/abc_dic/
fi

python DeepLabV3Plus_Pytorch/gen_masks_with_acq.py \
    --save_root_path ${output_dir}/gen_masks_with_acq/result/Round${round}/ \
    --ngbr_path ${output_dir}/gen_masks_with_acq/ngbr/Round${round}/ \
    --round ${round} \
    --dataset_dir ${dataset_dir} \
    --dataset ${dataset} \
    --acq_on_sam_path ${acq_on_sam_path} \
    --spx_root_path ${spx_root_path} \
    --n_label_root_path ${n_label_root_path} \
    --budget ${budget} \
    --sel_hist_fdr ${output_dir}/gen_masks_with_acq/ngbr