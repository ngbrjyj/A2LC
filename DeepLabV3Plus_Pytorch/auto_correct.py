import os
import json
import utils
import torch
import network
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
from a2lc.lcm import LCMasks
from a2lc.lcm import LCModule
from scipy.special import softmax
from a2lc.utils import get_palette_for
from utils import ext_transforms as et
from torch.utils.data import DataLoader
from a2lc.nonredundant import Unique_Manager

def get_argparser():
    parser = argparse.ArgumentParser()
    
    available_models = sorted(name 
                              for name in network.modeling.__dict__ if name.islower() 
                              and not (name.startswith("__") or name.startswith('_')) 
                              and callable(network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--ckpt", type=str)
    parser.add_argument('--round', type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--devkit_path", type=str)
    parser.add_argument("--spx_root_path", type=str)
    parser.add_argument('--gen_ngbr_path', type=str)
    parser.add_argument("--label_root_path", type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--conf_thres', type=float, default=0.99)
        
    return parser

def transform(sample, opts):
    transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=opts.mean, std=opts.std),
    ])
    image, label = transform(sample['image'], sample['label'])
    sample = {'image': image, 'label': label}
    
    return sample

def initialize(file_paths):
    for path in file_paths:
        with open(path, 'w') as f:
            pass

def append(file_path, line):
    with open(file_path, 'a') as f:
        f.write(line + '\n')

def has_keyvalue(dictionary, key, value):
    return (key in dictionary) and (value in dictionary[key])

def auto_correction(model, args):
    model.eval()
    devkit_path = args.devkit_path
    if 'pascal' in args.dataset.lower():
        image_root_path = os.path.join(devkit_path, 
                                       'VOCdevkit/VOC2012/JPEGImages')
    elif 'city' in args.dataset.lower():
        image_root_path = os.path.join(devkit_path, 
                                       'leftImg8bit_trainvaltest/leftImg8bit')
    
    label_root_path = args.label_root_path
    spx_root_path = args.spx_root_path
    
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    if 'pascal' in args.dataset.lower():
        imageset_path = os.path.join(devkit_path, 
                                     'VOCdevkit/VOC2012/ImageSets/Segmentation',
                                     'train.txt')
    elif 'city' in args.dataset.lower():
        imageset_path = os.path.join(image_root_path, 'train.txt')
    
    with open(imageset_path, 'r') as f:
         lines = f.readlines()
    image_list = [x.strip() for x in lines]
    image_list = sorted(set(image_list))
       
    image_name_path = {}
    for image_name in image_list:                
        if 'pascal' in args.dataset.lower():
            image_path = os.path.join(image_root_path, image_name + '.jpg')            
        elif 'city' in args.dataset.lower():
            image_path = os.path.join(image_root_path, 
                                      'train', 
                                      image_name.split('_')[0], 
                                      image_name + '_leftImg8bit.png')
            
        label_path = os.path.join(label_root_path, image_name + '.png')
        obj_path = os.path.join(spx_root_path, image_name + '.png')   
                        
        image_name_path[image_name] = [image_path, label_path, obj_path]
    
    save_data_fdr = os.path.join(save_path, 'data')
    os.makedirs(save_data_fdr, exist_ok=True)
    save_img_path = os.path.join(save_data_fdr, 'image.txt')
    save_mask_path = os.path.join(save_data_fdr, 'mask.txt')
    save_feat_path = os.path.join(save_data_fdr, 'feature.txt')
    save_lbl_path = os.path.join(save_data_fdr, 'label.txt')
    save_query_path = os.path.join(save_data_fdr, 'queried.txt')
    
    manual_corr_mask_pth = os.path.join(args.gen_ngbr_path,
                                        f'Round{args.round}',
                                        'manual_corr_mask.json')
    manual_corr_mask = json.load(open(manual_corr_mask_pth))
    
    manual_corr_lbl_pth = os.path.join(args.gen_ngbr_path,
                                   f'Round{args.round}',
                                   'manual_corr_lbl.json')
    manual_corr_lbl = json.load(open(manual_corr_lbl_pth))
    
    lcm_train_dic = {}  
    for _, inner_dict in manual_corr_lbl.items():        
        for k, v in inner_dict.items():            
            after_label, count = int(k), int(v)
            
            if after_label not in lcm_train_dic:
                lcm_train_dic[after_label] = count
            else:
                lcm_train_dic[after_label] += count
                
    lcm_train_dic = dict(sorted(lcm_train_dic.items(), 
                                      key=lambda item: item[1], 
                                      reverse=True))
    
    initialize([save_img_path, 
                save_mask_path, 
                save_feat_path, 
                save_lbl_path, 
                save_query_path])
    
    for _, image_name in \
        tqdm(enumerate(image_name_path.keys()), total=len(image_name_path), \
            desc="STEP0: Maskset preparation."):
                   
        image_path, label_path, obj_path = image_name_path[image_name]
        
        image = Image.open(image_path).convert('RGB') 
        label = Image.open(label_path)         
        sample = {'image': image, 'label': label} 
        sample = transform(sample, args) 
        sample["image"] = sample["image"].reshape(1, 3, *tuple(sample['image'].shape[1:3])) 
        
        prob = model(sample["image"].cuda()) 
        prob_np = prob.detach().cpu().numpy() 
        prob_np = np.squeeze(prob_np) 
        prob_soft = softmax(prob_np, axis=0) 
        
        objects = np.array(Image.open(obj_path)) 
        labels = np.array(label) 
        for obj in np.unique(objects):
            if obj == 0:
                continue
            
            append(save_img_path, image_name)             
            append(save_mask_path, str(int(obj)))   
                      
            obj_x, obj_y = np.where(objects == obj)
            mask_feats = prob_soft[:, obj_x, obj_y] 
            mask_feat = np.mean(mask_feats, axis=1) 
            append(save_feat_path, " ".join(map(str, mask_feat)))                 
            
            mask_labels = labels[obj_x, obj_y]
            each_labels, counts = np.unique(mask_labels, return_counts=True)
            mask_label = each_labels[np.argmax(counts)]
            append(save_lbl_path, str(int(mask_label))) 
            
            if has_keyvalue(manual_corr_mask, image_name, obj): 
                append(save_query_path, 'True') 
            else:
                append(save_query_path, 'False')     
    del model             
    
    if 'city' in args.dataset.lower():
        num_classes = 19
    elif 'pascal' in args.dataset.lower():
        num_classes = 21
    
    model = LCModule(num_classes)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    save_lcm_fdr = os.path.join(save_path, 'lcm')
    os.makedirs(save_lcm_fdr, exist_ok=True)
    save_model_path = os.path.join(save_lcm_fdr, 'model.pth')
    
    for cls in range(num_classes):
        if cls not in lcm_train_dic.keys():
            lcm_train_dic[cls] = 0
    lcm_train_dic = {k: v for k, v in lcm_train_dic.items() if k != 255}   
    
    train_counts = sum(lcm_train_dic.values()) 
    train_weights = {k: train_counts / v if v != 0 else 0 for k, v in lcm_train_dic.items()}
    train_weights = {k: v / sum(train_weights.values()) for k, v in train_weights.items()} 
    train_weights = torch.tensor([train_weights[k] 
                                  for k in range(len(train_weights))], 
                                 dtype=torch.float32).to(device)
    
    tail_cls = sorted(lcm_train_dic, 
                      key=lcm_train_dic.get)[:int(len(lcm_train_dic) * 0.5)]
    best_tail_cls = tail_cls[0]
        
    criterion = nn.CrossEntropyLoss(weight=train_weights, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_dataset = LCMasks(save_img_path, 
                            save_mask_path, 
                            save_feat_path, 
                            save_lbl_path, 
                            save_query_path, 
                            is_queried=True)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True)

    for _ in tqdm(range(args.epochs), 
                      desc="STEP1: Supervised train with queried masks."):
        model.train()

        for image, mask, feature, label in train_loader:

            feature = feature.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_model_path)

    save_lcm_out_fdr = os.path.join(save_lcm_fdr, 'outputs')
    os.makedirs(save_lcm_out_fdr, exist_ok=True)    
    save_img_out_path = os.path.join(save_lcm_out_fdr, 'image.txt')
    save_mask_out_path = os.path.join(save_lcm_out_fdr, 'mask.txt')
    save_lbl_out_path = os.path.join(save_lcm_out_fdr, 'label.txt')
    initialize([save_img_out_path, save_mask_out_path, save_lbl_out_path])
    
    test_dataset = LCMasks(save_img_path, 
                           save_mask_path, 
                           save_feat_path, 
                           save_lbl_path, 
                           save_query_path, 
                           is_queried=False) 
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False)
    
    model = LCModule(num_classes) 
    model.load_state_dict(torch.load(save_model_path))
    model.to(device)
    model.eval()  
    
    uniquer = Unique_Manager(sel_hist_fdr = args.gen_ngbr_path,
                             curr_round = args.round)
    past_corr_masks = uniquer.get_hist_dict()            
                
    with torch.no_grad():  
        for image, mask, feature, label in \
            tqdm(test_loader, 
                 desc="STEP2: Selective label update for unqueried masks."):
            
            feature = feature.to(device)             
            predictions = model(feature)  
            conf, pred_label = torch.max(predictions, dim=1)  
            
            for img, msk, _, pred_lbl, conf, origin_lbl in \
                zip(image, mask, feature.cpu(), pred_label.cpu(), conf.cpu(), label):      
                     
                msk, pred_lbl, conf, origin_lbl = \
                    msk.item(), pred_lbl.item(), conf.item(), origin_lbl.item()
                                
                skip_case = (
                    origin_lbl == pred_lbl
                    or (img in past_corr_masks and msk in past_corr_masks[img])
                    or conf < args.conf_thres
                    or pred_lbl in tail_cls
                    or origin_lbl == best_tail_cls
                )
                if skip_case:
                    continue
                                
                append(save_img_out_path, img)
                append(save_mask_out_path, str(msk))
                append(save_lbl_out_path, str(pred_lbl))
    
    palette = get_palette_for(args.dataset)
    with open(save_img_out_path, 'r') as img_file, \
        open(save_mask_out_path, 'r') as mask_file, \
        open(save_lbl_out_path, 'r') as lbl_file:

        for img_line, mask_line, lbl_line in \
            tqdm(zip(img_file, mask_file, lbl_file), 
                 total=sum(1 for _ in open(save_img_out_path))):
            
            img = img_line.strip()   
            mask = int(mask_line.strip())   
            lbl = int(lbl_line.strip())   
            
            lbl_path = os.path.join(label_root_path, img + '.png')
            _, _, obj_path = image_name_path[img]
            
            labels = np.array(Image.open(lbl_path))
            objects = np.array(Image.open(obj_path))
            
            obj_x, obj_y = np.where(objects == mask)
            labels[obj_x, obj_y] = lbl  
            
            final_label = Image.fromarray(labels.astype('uint8'))
            final_label.putpalette(palette)
            final_label.save(lbl_path) 
    
def main():    
    opts = get_argparser().parse_args()
     
    if 'pascal' in opts.dataset.lower():
        opts.num_classes = 21
        opts.mean = [0.485, 0.456, 0.406]
        opts.std = [0.229, 0.224, 0.225]
    elif 'city' in opts.dataset.lower():
        opts.num_classes = 19
        opts.mean = [0.287, 0.325, 0.284]
        opts.std = [0.187, 0.190, 0.187]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, 
                                                  output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, 
                          momentum=0.01)
    
    checkpoint = torch.load(opts.ckpt, 
                            map_location=torch.device('cpu'), 
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    del checkpoint

    auto_correction(model, opts)


if __name__ == '__main__':
    main()
