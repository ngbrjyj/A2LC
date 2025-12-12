import os
import cv2
import json
import torch
import argparse
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from a2lc.utils import get_palette_for

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

num_dic = {
    '2346' : 'road',
    '11996' : 'sidewalk',
    '2311' : 'building',
    '2813' : 'wall',
    '8638' : 'fence',
    '6536' : 'pole',
    '4026' : 'trafficlight',
    '7138' : 'trafficlight',
    '4026' : 'trafficsign',
    '5332' : 'trafficsign',
    '16206' : 'trafficsign',
    '10072' : 'vegetation',
    '9291' : 'terrain',
    '3712' : 'sky',
    '2711' : 'person',
    '7945' : 'rider',
    '2482' : 'car',
    '4744' : 'truck',
    '3902' : 'bus',
    '3345' : 'train',
    '9055' : 'motorcycle',
    '10165' : 'bicycle'
}

label_dic = {
    "road" : 0, "sidewalk" : 1, "building" : 2, "wall" : 3, "fence" : 4,
    "pole" : 5, "trafficlight" : 6, "trafficsign" : 7, "vegetation" : 8, 
    "terrain" : 9, "sky" : 10, "person" : 11, "rider" : 12, "car" : 13, "truck" : 14, 
    "bus" : 15, "train" : 16, "motorcycle" : 17, "bicycle" : 18, "ignore" : 255
}

def get_argparser():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str,
                        default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str,
                        default="Grounded-Segment-Anything/GroundingDINO/weights/groundingdino_swint_ogc.pth",
                        help="path to checkpoint file")
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, 
                        help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False,
                        default="Grounded-Segment-Anything/segment_anything/weights/sam_vit_h_4b8939.pth",
                        help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, 
                        help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true",
                        help="using sam-hq for prediction")
    parser.add_argument("--input_image", type=str, default="temp",
                        help="path to image file")
    parser.add_argument("--text_prompt", type=str,
                        default="Road. Sidewalk. Building. Wall. Fence. Pole. Trafficlight. Trafficsign. Vegetation. Terrain. Sky. Person. Rider. Car. Truck. Bus. Train. Motorcycle. Bicycle.",
                        help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str,
                        help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.2, 
                        help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, 
                        help="text threshold")
    parser.add_argument("--devkit_path", type=str)
    parser.add_argument("--ngbr_path", type=str)
    
    return parser

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [ 
            T.RandomResize([512], max_size=1024), 
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
        ]
    )
    image, _ = transform(image_pil, None) 
    
    return image_pil, image 

def load_model(model_config_path, 
               model_checkpoint_path, 
               device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path,
                            map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), 
                                     strict=False)
    print(load_res)
    _ = model.eval()
    
    return model

def get_grounding_output(model, 
                         image,
                         caption,
                         box_threshold, 
                         with_logits=True, 
                         device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0] 
    boxes = outputs["pred_boxes"].cpu()[0] 

    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold 
    boxes_filt = boxes_filt[filt_mask]
    logits_filt = logits_filt[filt_mask] 

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt): 
        idx = torch.argmax(logit)
        pred_phrase = tokenized['input_ids'][idx] 
        pred_phrase = num_dic[str(pred_phrase)] 
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})") 
        else:
            pred_phrases.append(pred_phrase)
            
    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) 
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    if 'unknown' in label:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, 
                                   edgecolor='red', facecolor=(0,0,0,0), lw=2))  
        ax.text(x0, y0, label)      
    else:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, 
                                   edgecolor='green', facecolor=(0,0,0,0), lw=1))
        ax.text(x0, y0, label, fontsize='x-small', color='#597dce')

def save_mask_data(args, 
                   image_name, 
                   output_dir, 
                   mask_list, 
                   box_list, 
                   label_list):
    value = 0
    mask_dic = {}
    n, _, _, _ = mask_list.shape 
    for i in range(n):
        mask = mask_list[i,:,:,:].cpu().numpy().astype('int')
        mask_dic[i] = np.sum(mask)
    
    obj_img = torch.zeros(mask_list.shape[-2:]) 
    mask_img = torch.ones(mask_list.shape[-2:]) * 255 
    sorted_dict = dict(sorted(mask_dic.items(), key=lambda item: -item[1]))
    for idx in sorted_dict.keys():
        mask = mask_list[idx,:,:,:].cpu().numpy()[0] 
        label = label_list[idx] 
        name, logit = label.split('(')
        label_val = label_dic[name]
        
        mask_img[mask == True] = label_val
        obj_img[mask == True] = idx + 1 
    
    final_label = Image.fromarray(mask_img.numpy().astype('uint8'))
    final_label.putpalette(palette)
    final_label.save(os.path.join(output_dir,
                                  str(args.box_threshold),
                                  "mask_jpg",
                                  image_name + ".png"))
        
    final_obj = Image.fromarray(obj_img.numpy().astype('uint8'))
    final_obj.save(os.path.join(output_dir, 
                                str(args.box_threshold),
                                "obj_jpg",
                                image_name + ".png"))

    json_data = [] 
    for label, box in zip(label_list, box_list): 
        name, logit = label.split('(') 
        logit = logit[:-1] 
        json_data.append({
            'value': value,
            'label': name, 
            'logit': float(logit), 
            'box': box.numpy().tolist(),
        })
        value += 1
    
    with open(os.path.join(output_dir, 
                           str(args.box_threshold),
                           "mask_json",
                           image_name + ".json"), 
              'w') as f:
        json.dump(json_data, f)

palette = get_palette_for('cityscapes')

if __name__ == "__main__":
    args = get_argparser().parse_args()
    
    config_file = args.config
    grounded_checkpoint = args.grounded_checkpoint
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    box_threshold = args.box_threshold
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    model = load_model(config_file, 
                       grounded_checkpoint, 
                       device=device)
    model = model.to(device)
    if use_sam_hq:
        predictor = SamPredictor(
            sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device)
            )
    else:
        predictor = SamPredictor(
            sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
            )
    
    devkit_path = args.devkit_path
    image_root_path = os.path.join(devkit_path, 'leftImg8bit/train')
    imageset_path = os.path.join(devkit_path, 'leftImg8bit', 'train_alc.txt')
    
    all_files = []
    for root, dirs, files in os.walk(image_root_path):
        for filename in files:
            if filename.endswith('_leftImg8bit.png'):
                new_filename = filename.replace('_leftImg8bit.png', '')
                all_files.append(new_filename)
    all_files = sorted(set(all_files))
    with open(imageset_path, 'w') as file:
        for new_filename in all_files:
            file.write(new_filename + '\n')

    with open(imageset_path, 'r') as f:
         lines = f.readlines()
    image_list = [x.strip() for x in lines]
    
    image_name_path = {}
    for image_name in image_list:
        image_path = os.path.join(image_root_path, 
                                  image_name.split('_')[0], 
                                  image_name + '_leftImg8bit.png')
        image_name_path[image_name] = image_path

    os.makedirs(os.path.join(output_dir, str(args.box_threshold), "mask_json"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, str(args.box_threshold), "mask_jpg"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, str(args.box_threshold), "obj_jpg"), exist_ok=True)
    
    for idx, image_name in enumerate(image_name_path.keys()):
        print(idx, image_name) 
        
        if os.path.exists(os.path.join(output_dir, 
                                       str(args.box_threshold),
                                       "mask_json",
                                       image_name + ".json")):
            continue
        
        image_path = image_name_path[image_name]
        image_pil, image = load_image(image_path) 
        
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, device=device
        )
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        size = image_pil.size 
        H, W = size[1], size[0] 
        for i in range(boxes_filt.size(0)): 
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = \
            predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
        if transformed_boxes.shape[0] == 0:
            mask_np = np.full((H, W), 255) 
            final_label = Image.fromarray(mask_np.astype('uint8'))
            final_label.save(os.path.join(output_dir,
                                          str(args.box_threshold),
                                          "mask_jpg",
                                          image_name + ".png"))

            final_obj = Image.fromarray(mask_np.astype('uint8'))
            final_obj.save(os.path.join(output_dir,
                                        str(args.box_threshold),
                                        "obj_jpg",
                                        image_name + ".png"))
            continue
        
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        save_mask_data(args, 
                       image_name,
                       output_dir, 
                       masks, 
                       boxes_filt, 
                       pred_phrases)