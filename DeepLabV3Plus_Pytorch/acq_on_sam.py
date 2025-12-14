import os
import gzip
import pickle
import argparse
import numpy as np
from PIL import Image
from a2lc.abc import ABCil
import multiprocessing as mp
from a2lc.utils import resize_channel 
from a2lc.utils import get_palette_for
from scipy.spatial.distance import cosine
from a2lc.nonredundant import Unique_Manager

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int) 
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--mid_dir", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--sel_hist_fdr", type=str)  
    parser.add_argument("--spx_root_path", type=str)
    parser.add_argument("--soft_label_path", type=str)
    parser.add_argument("--n_label_root_path", type=str)
    
    return parser

def cos_sim(a, b):
    c, n = a.shape

    cos = []
    for i in range(n):
        each_a = a[:, i] 
        cos.append(1 - cosine(each_a, b)) 
    cos = np.array(cos)
    return cos

opts = get_argparser().parse_args()

palette = get_palette_for(opts.dataset)
    
soft_label_path = opts.soft_label_path
spx_root_path = opts.spx_root_path
n_label_root_path = opts.n_label_root_path

if 'pascal' in opts.dataset.lower():
    train_imageset_path = os.path.join(opts.dataset_dir, 
                                       'VOCdevkit/VOC2012/ImageSets/Segmentation',
                                       'train.txt')
elif 'city' in opts.dataset.lower():
    train_imageset_path = os.path.join(opts.dataset_dir, 
                                       'leftImg8bit_trainvaltest/leftImg8bit',
                                       'train.txt')
    image_path = os.path.join(opts.dataset_dir, 
                              'leftImg8bit_trainvaltest/leftImg8bit/train')
    gt_path = os.path.join(opts.dataset_dir, 
                           'gtFine_trainvaltest/gtFine/train')
    
os.makedirs(opts.mid_dir, exist_ok=True)
    
with open(train_imageset_path, 'r') as f:
    lines = f.readlines()
image_list = [x.strip() for x in lines]

image_name_path = {}
for image_name in image_list:
    obj_path = os.path.join(spx_root_path, image_name + '.png')                    
    n_label_path = os.path.join(n_label_root_path, image_name + '.png')
    soft_path = os.path.join(soft_label_path, image_name + '.pkl.gz')
    image_name_path[image_name] = [obj_path, n_label_path, soft_path]
   
uniquer = Unique_Manager(sel_hist_fdr = opts.sel_hist_fdr,
                         curr_round = opts.round) 
past_corr_masks = uniquer.get_hist_dict()

abcer = ABCil(m_fdr = n_label_root_path,
              o_fdr = spx_root_path,              
              dataset = opts.dataset,
              curr_round = opts.round,
              past_corr_masks = past_corr_masks)
ACWeight = abcer.run()
    
def acq(image_name, opts):   
    abc_dic = {}
    obj_path, n_label_path, soft_path = image_name_path[image_name]
    
    objects = np.array(Image.open(obj_path)) 
    n_labels = np.array(Image.open(n_label_path)) 
         
    with gzip.open(soft_path, 'rb') as file:
        soft_labels = pickle.load(file) 
    if 'city' in opts.dataset.lower():
        re_w, re_h = 2048, 1024 
        soft_labels = resize_channel(re_w, re_h, soft_labels, 'B')
    
    arg_labels = np.argmax(soft_labels, axis=0) 
    
    for obj in np.unique(objects):
        if obj == 0:
            continue 
         
        key = (image_name, obj) 
        obj_x, obj_y = np.where(objects == obj) 
        
        region_labels = arg_labels[obj_x, obj_y] 
        pseudo_doms, counts = np.unique(region_labels, return_counts=True)
        pseudo_dom = pseudo_doms[np.argmax(counts)]
        
        ids = np.where(region_labels == pseudo_dom) 
        obj_px, obj_py = obj_x[ids], obj_y[ids] 
        
        region_feats = soft_labels[:, obj_px, obj_py]
        rep_feat = np.mean(region_feats, axis=1) 
                 
        unique_labels, counts = \
            np.unique(n_labels[obj_px, obj_py], return_counts=True)
        rep_label = unique_labels[np.argmax(counts)]
        if rep_label == 255:
            unique_labels, counts = \
                np.unique(arg_labels[obj_px, obj_py], return_counts=True)
            rep_label = unique_labels[np.argmax(counts)]                
           
        region_feats = soft_labels[:, obj_x, obj_y] 
        cos_array = cos_sim(region_feats, rep_feat) 
        cos_ids = np.where(cos_array >= 0.0) 
        cos_idx, cos_idy = obj_x[cos_ids], obj_y[cos_ids] 

        cil_arr, acw_arr = [], []
        for each_x, each_y in zip(cos_idx, cos_idy):

            each_label = n_labels[each_x, each_y]
            if each_label == 255:
                each_label = arg_labels[each_x, each_y]
                
            each_cil = 1 - soft_labels[each_label, each_x, each_y]
            cil_arr.append(each_cil)                  
            acw_arr.append(ACWeight[each_label])                       
        
        cil_arr = np.array(cil_arr) 
        sim_alpha = np.array(cos_array[cos_ids])
        abc = np.sum(sim_alpha * acw_arr * cil_arr)  
        abc_dic[key] = (abc, cos_idx, cos_idy)        
        
        save_path = os.path.join(opts.save_path, 
                                 image_name, 
                                 'abc_dic.pkl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as file: 
            pickle.dump(abc_dic, file)  
                
def main():    
    keys = list(image_name_path.keys())
        
    max_cpu_num = 50
    process = []
    for idx, image_name in enumerate(keys):
        print(idx, image_name)           
        
        p = mp.Process(target=acq, args=(image_name, opts))
        p.start()
        process.append(p)
         
        if len(process) >= max_cpu_num:
            process[0].join()
            process.pop(0)
            
if __name__ == '__main__':
    main()