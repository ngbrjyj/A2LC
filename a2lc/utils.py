import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# preprocess
def resize_img(w, h, image, mode):
    if mode == 'N':
        image = image.resize((w, h), Image.NEAREST)
    elif mode == 'B':
        image = image.resize((w, h), Image.BILINEAR)
        
    return image

def resize_channel(w, h, image_channel:np, mode):
    image_channel = torch.from_numpy(image_channel).float() 
    if image_channel.dim() == 3: 
        image_channel = image_channel.unsqueeze(0)
    if mode == 'N':
        image_channel = torch.nn.functional.interpolate(image_channel,
                                                        size=(h, w), 
                                                        mode='nearest')
    elif mode == 'B':
        image_channel = torch.nn.functional.interpolate(image_channel, 
                                                        size=(h, w), 
                                                        mode='bilinear')
    image_channel = image_channel.squeeze(0).numpy()
    
    return image_channel 

# visualize
def get_palette_for(data = 'pascal'):
    if 'pascal' in data.lower():
        pascal_palette = \
            [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128,]
        pascal_palette = pascal_palette + [0] * (256*3 - len(pascal_palette))
        palette = pascal_palette
    elif 'city' in data.lower():
        city_palette = \
            [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32,]
        city_palette = city_palette + [0] * (256*3 - len(city_palette))
        palette = city_palette
    
    return palette
