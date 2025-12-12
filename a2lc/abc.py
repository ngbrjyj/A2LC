import os
import numpy as np
from PIL import Image

class ABCil:
    def __init__(self, 
                 m_fdr=None, 
                 o_fdr=None,                  
                 dataset=None,
                 curr_round = None,
                 past_corr_masks=None):
                    
        self.mask_fdr = m_fdr
        self.obj_fdr = o_fdr
        
        if 'pascal' in dataset.lower():
            self.num_classes = 21
        elif 'city' in dataset.lower():
            self.num_classes = 19
            
        self.curr_round = curr_round
        self.past_corr_masks = past_corr_masks
        
    def run(self):
        
        pixels = self.count_pixel()     
        min_count = min(pixels.values()) 
               
        CRScore = {label: min_count / count 
                  for label, count in pixels.items()}                     
        DIScore = self.cal_kldiv(pixels)              
         
        ACWeight = {label: float(crscore ** (DIScore ** 3)) 
                    for label, crscore in CRScore.items()}             
        for cls in range(self.num_classes): 
            if cls not in ACWeight: ACWeight[cls] = 0
        ACWeight = dict(sorted(ACWeight.items()))    
                        
        return ACWeight
    
    def count_pixel(self):

        count = {}
        for file in os.listdir(self.obj_fdr):
            o_fle = os.path.join(self.obj_fdr, file)
            m_fle = os.path.join(self.mask_fdr, file)                
            objects = np.array(Image.open(o_fle)).astype(int)
            masks = np.array(Image.open(m_fle)).astype(int)
            
            for obj in np.unique(objects):
                if obj == 0:
                    continue 
                
                image_name, obj = file.replace('.png', ''), obj   
                if image_name in self.past_corr_masks \
                    and obj in self.past_corr_masks[image_name]:
                        continue
                                
                obj_mask = (objects == obj)
                unique_labels, counts = \
                    np.unique(masks[obj_mask], return_counts=True)
                    
                for lbl, cnt in zip(unique_labels, counts):
                    lbl, cnt = int(lbl), int(cnt)
                    if lbl not in count:
                        count[(lbl)] = 0  
                    count[lbl] += cnt  

        sorted_keys = sorted(count)
        sorted_count = {key : count[key] for key in sorted_keys}
        
        return sorted_count
    
    def cal_kldiv(self, dist):    
        n = len(dist)
        total_pixels = sum(dist.values())
        probs = np.array(list(dist.values())) / total_pixels 
        uni_probs = np.full(n, 1 / n)
        kldiv = np.sum(probs * np.log(probs / uni_probs))
    
        return kldiv