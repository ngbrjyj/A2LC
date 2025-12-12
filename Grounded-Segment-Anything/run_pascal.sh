output_dir='/mnt/ssd1/ngbrjyj/A2LC' 
dataset_dir=\
'/mnt/ssd1/ngbrjyj/DATASET/PASCAL_2012/VOCdevkit'

python Grounded-Segment-Anything/run_pascal.py \
  --output_dir ${output_dir}/demo_pascal/result/PASCAL_VOC_2012 \
  --ngbr_path ${output_dir}/demo_pascal/ngbrjyj/PASCAL_2012 \
  --devkit_path ${dataset_dir} 