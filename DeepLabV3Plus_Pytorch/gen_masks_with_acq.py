import os
import json
import pickle
import argparse
import numpy as np
from PIL import Image
from collections import Counter
from a2lc.utils import get_palette_for
from a2lc.nonredundant import Unique_Manager
from datasets.cityscapes import Cityscapes as Cityclass

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int) 
    parser.add_argument("--budget", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--ngbr_path", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--sel_hist_fdr", type=str)  
    parser.add_argument("--spx_root_path", type=str)
    parser.add_argument("--save_root_path", type=str)
    parser.add_argument("--acq_on_sam_path", type=str)
    parser.add_argument("--n_label_root_path", type=str)
    
    return parser

def combine_dict(src_dir : str,
                 find_name : str):
    combined_dict = {}
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file == find_name:
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as fr:
                    data = pickle.load(fr)
                    combined_dict.update(data)
    return combined_dict
    
opts = get_argparser().parse_args()

palette = get_palette_for(opts.dataset)
  
if 'pascal' in opts.dataset.lower():
    label_root_path = \
        os.path.join(opts.dataset_dir, 'VOCdevkit/VOC2012/SegmentationClass')
    train_imageset_path = os.path.join(opts.dataset_dir, 
                                       'VOCdevkit/VOC2012/ImageSets/Segmentation',
                                       'train.txt')
elif 'city' in opts.dataset.lower():
    label_root_path = \
        os.path.join(opts.dataset_dir, 'gtFine_trainvaltest/gtFine')
    train_imageset_path = os.path.join(opts.dataset_dir, 
                                       'leftImg8bit_trainvaltest/leftImg8bit', 
                                       'train.txt')
val_imageset_path = train_imageset_path.replace('train.txt', 'val.txt')

with open(train_imageset_path, 'r') as f:
    lines = f.readlines()
image_list = [x.strip() for x in lines]

with open(val_imageset_path, 'r') as f:
    lines = f.readlines()
val_list = [x.strip() for x in lines]
val_list = sorted(set(val_list))

image_name_path = {}
for image_name in image_list:     
    if 'pascal' in opts.dataset.lower():
        label_path = os.path.join(label_root_path, 
                                  image_name + '.png')
    elif 'city' in opts.dataset.lower():
        if image_name.split('_')[0] in ['frankfurt', 'lindau', 'munster']:
            label_path = os.path.join(label_root_path, 
                                      'val', 
                                      image_name.split('_')[0], 
                                      image_name + '_gtFine_labelIds.png')
        else:
            label_path = os.path.join(label_root_path, 
                                      'train', 
                                      image_name.split('_')[0], 
                                      image_name + '_gtFine_labelIds.png')
    obj_path = os.path.join(opts.spx_root_path, image_name + '.png') 
    n_label_path = os.path.join(opts.n_label_root_path, image_name + '.png')
    
    image_name_path[image_name] = [label_path, obj_path, n_label_path]        
        
names = ['abc_dic.pkl']

manual_corr_lbl = {}
manual_corr_lbl_pth = os.path.join(opts.ngbr_path, 
                                   'manual_corr_lbl.json')

for name in names:
    parts = name.split('.')
    save_root_path = os.path.join(opts.save_root_path, parts[0])
    os.makedirs(save_root_path, exist_ok=True)
    
    pkl_dict = combine_dict(src_dir = opts.acq_on_sam_path,
                            find_name = name)
    sorted_dict = dict(sorted(pkl_dict.items(), 
                              key=lambda item: -item[1][0])) 

    count = 0
    pkl_images = {}  
    manual_corr_mask_pth = os.path.join(opts.ngbr_path, 
                                        'manual_corr_mask.json')
    os.makedirs(os.path.dirname(manual_corr_mask_pth), exist_ok=True)
      
    uniquer = Unique_Manager(sel_hist_fdr = opts.sel_hist_fdr,
                             curr_round = opts.round)
    past_corr_masks = uniquer.get_hist_dict()          
     
    for idx, key in enumerate(sorted_dict.keys()): 
        if count == opts.budget: 
            break 

        image_name, obj = key
        obj = int(obj)
        
        if image_name in past_corr_masks.keys() and obj in past_corr_masks[image_name]:
            continue    
        
        if image_name not in pkl_images.keys():
            pkl_images[image_name] = [obj]
        else:
            pkl_images[image_name].append(obj)
            
        count += 1
        
    with open(manual_corr_mask_pth, 'w') as f:
        json.dump(pkl_images, f, indent=4)
    
    for idx, image_name in enumerate(image_name_path.keys()):
        print(name, idx, image_name) 
        
        label_path, obj_path, n_label_path = image_name_path[image_name]

        if 'pascal' in opts.dataset.lower():
            labels = np.array(Image.open(label_path))  
        elif 'city' in opts.dataset.lower():
             labels = Cityclass.encode_target(Image.open(label_path))
        objects = np.array(Image.open(obj_path))
        n_labels = np.array(Image.open(n_label_path))
            
        if image_name in pkl_images.keys():            
            for obj in pkl_images[image_name]:
                key = (image_name, obj)  
                
                _, relabel_x, relabel_y = sorted_dict[key]                    
                n_region_labels = n_labels[relabel_x, relabel_y]
                region_labels = labels[relabel_x, relabel_y] 
                bin = (n_region_labels == region_labels).astype('int')
                acc = np.sum(bin) / bin.shape[0]
                
                bfr_counter = Counter(n_region_labels)
                before_label, _ = max(bfr_counter.items(), key = lambda x: x[1])
                afr_counter = Counter(region_labels)
                rep_label, _ = max(afr_counter.items(), key = lambda x: x[1])                               
                
                if acc <= 0.5:  
                    n_labels[relabel_x, relabel_y] = rep_label
                    after_label = rep_label 
                else:
                    after_label = before_label     
                    
                before_label, after_label = int(before_label), int(after_label)
                if before_label not in manual_corr_lbl.keys():
                    manual_corr_lbl[before_label] = {}
                if after_label not in manual_corr_lbl[before_label].keys():
                    manual_corr_lbl[before_label][after_label] = 0
                manual_corr_lbl[before_label][after_label] += 1                    
                
        final_label = Image.fromarray(n_labels.astype('uint8'))
        final_label.putpalette(palette)
        final_label.save(save_root_path + '/' + image_name + '.png')
    
    with open(manual_corr_lbl_pth, 'w') as file:
        json.dump(manual_corr_lbl, file, indent=4)
    
    for idx, image_name in enumerate(val_list):
        print(name, idx, image_name)
        
        if 'pascal' in opts.dataset.lower():
            label_path = os.path.join(label_root_path, 
                                      image_name + '.png')
        elif 'city' in opts.dataset.lower():
            label_path = os.path.join(label_root_path, 
                                      'val', 
                                      image_name.split('_')[0], 
                                      image_name + '_gtFine_labelIds.png')

        if 'pascal' in opts.dataset.lower():
            labels = np.array(Image.open(label_path))
        elif 'city' in opts.dataset.lower():
            labels = Cityclass.encode_target(Image.open(label_path))
        final_label = Image.fromarray(labels.astype('uint8'))
        final_label.save(save_root_path + '/' + image_name + '.png')    