output_dir='/mnt/ssd1/ngbrjyj/A2LC' 
dataset_dir=\
'/mnt/ssd1/ngbrjyj/DATASET/Cityscapes/leftImg8bit_trainvaltest'

python Grounded-Segment-Anything/run_cityscapes.py \
  --output_dir ${output_dir}/demo_pascal/result/Cityscapes \
  --devkit_path ${dataset_dir}