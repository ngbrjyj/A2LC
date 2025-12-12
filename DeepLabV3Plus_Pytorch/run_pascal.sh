set -e

BUDGET=1000  
OUTPUT_DIR='/mnt/ssd6/ngbrjyj/A2LC' 
DATASET_DIR='/mnt/ssd1/ngbrjyj/DATASET/PASCAL_2012'

for SEED in {1..3}; do
    ROUND=0
    bash DeepLabV3Plus_Pytorch/pascal/train.sh \
        $SEED $ROUND $DATASET_DIR $OUTPUT_DIR

    for ROUND in {1..5}; do
        bash DeepLabV3Plus_Pytorch/pascal/soft_label.sh \
            $SEED $ROUND $DATASET_DIR $OUTPUT_DIR

        bash DeepLabV3Plus_Pytorch/pascal/acq_on_sam.sh \
            $ROUND $DATASET_DIR $OUTPUT_DIR

        bash DeepLabV3Plus_Pytorch/pascal/gen_masks_with_acq.sh \
            $ROUND $DATASET_DIR $OUTPUT_DIR $BUDGET

        bash DeepLabV3Plus_Pytorch/pascal/auto_correct.sh \
            $SEED $ROUND $DATASET_DIR $OUTPUT_DIR

        bash DeepLabV3Plus_Pytorch/pascal/train.sh \
            $SEED $ROUND $DATASET_DIR $OUTPUT_DIR
    done
done