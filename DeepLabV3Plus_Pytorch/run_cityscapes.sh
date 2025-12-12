set -e

BUDGET=10000  
OUTPUT_DIR='/mnt/ssd6/ngbrjyj/A2LC' 
DATASET_DIR='/mnt/ssd1/ngbrjyj/DATASET/Cityscapes'

for SEED in {1..3}; do
    ROUND=0
    bash DeepLabV3Plus_Pytorch/cityscapes/train.sh \
        $SEED $ROUND $DATASET_DIR $OUTPUT_DIR

    for ROUND in {1..5}; do
        bash DeepLabV3Plus_Pytorch/cityscapes/soft_label.sh \
            $SEED $ROUND $DATASET_DIR $OUTPUT_DIR

        bash DeepLabV3Plus_Pytorch/cityscapes/acq_on_sam.sh \
            $ROUND $DATASET_DIR $OUTPUT_DIR

        bash DeepLabV3Plus_Pytorch/cityscapes/gen_masks_with_acq.sh \
            $ROUND $DATASET_DIR $OUTPUT_DIR $BUDGET

        bash DeepLabV3Plus_Pytorch/cityscapes/auto_correct.sh \
            $SEED $ROUND $DATASET_DIR $OUTPUT_DIR

        bash DeepLabV3Plus_Pytorch/cityscapes/train.sh \
            $SEED $ROUND $DATASET_DIR $OUTPUT_DIR
    done
done