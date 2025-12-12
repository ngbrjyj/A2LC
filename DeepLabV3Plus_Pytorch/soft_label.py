import os
import gzip 
import torch
import utils
import pickle
import network
import argparse
import numpy as np
from PIL import Image
from scipy.special import softmax
from a2lc.utils import resize_img
from utils import ext_transforms as et

def get_argparser():
    parser = argparse.ArgumentParser()

    # Deeplab Options
    available_models = sorted(name 
                              for name in network.modeling.__dict__ if name.islower() 
                              and not (name.startswith("__") or name.startswith('_')) 
                              and callable(network.modeling.__dict__[name]))
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--crop_size", type=int, default=768, choices=[513, 768])
    
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--devkit_path", type=str)
    parser.add_argument("--spx_root_path", type=str)
    parser.add_argument("--label_root_path", type=str)
    
    return parser

def transform_train(sample, opts):
    train_transform = et.ExtCompose([
        et.ExtCenterCrop(opts.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=opts.mean, std=opts.std),
    ])
    image, label = train_transform(sample['image'], sample['label'])
    sample = {'image': image, 'label': label}
    return sample

def acquisition(model, args):
    model.eval()
    devkit_path = args.devkit_path
    if 'pascal' in args.dataset.lower():
        image_root_path = os.path.join(devkit_path, 'VOCdevkit/VOC2012/JPEGImages')
    elif 'city' in args.dataset.lower():
        image_root_path = os.path.join(devkit_path, 'leftImg8bit_trainvaltest/leftImg8bit')
        
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
        label_path = os.path.join(args.label_root_path, image_name + '.png')
        obj_path = os.path.join(args.spx_root_path, image_name + '.png')        
                   
        image_name_path[image_name] = [image_path, label_path, obj_path]
    
    for num, image_name in enumerate(image_name_path.keys()):
        print(num, image_name)
        image_path, label_path, _ = image_name_path[image_name]
        
        image = Image.open(image_path).convert('RGB') 
        label = Image.open(label_path)
        
        if 'city' in args.dataset.lower():
            re_h, re_w = int(args.crop_size/2), int(args.crop_size/2)
            image = resize_img(re_w, re_h, image, 'B')
            label = resize_img(re_w, re_h, label, 'N')                        
        
        h, w= label.size 
        y0 = int(round((args.crop_size - h) / 2.)) 
        x0 = int(round((args.crop_size - w) / 2.)) 
        
        sample = {'image': image, 'label': label} 
        sample = transform_train(sample, args) 
        sample["image"] = sample["image"].reshape(1, 3, args.crop_size, args.crop_size) 
        
        prob = model(sample["image"].cuda()) 
        
        prob_np = prob.detach().cpu().numpy() 
        prob_np = np.squeeze(prob_np) 
        
        if (x0 < 0) or (y0 < 0):  
            pass
        else:
            prob_np = prob_np[:, x0:(x0+w), y0:(y0+h)]
        prob_soft = softmax(prob_np, axis=0)

        with gzip.open(save_path + image_name + '.pkl.gz', 'wb') as file:
            pickle.dump(prob_soft, file)        
                
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
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    checkpoint = torch.load(opts.ckpt, 
                            map_location=torch.device('cpu'), 
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    del checkpoint
    
    acquisition(model, opts)

if __name__ == '__main__':
    main()