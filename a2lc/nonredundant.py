import os
import json

class Unique_Manager:
    def __init__(self, 
                 sel_hist_fdr,
                 curr_round = 1):
        
        self.sel_hist_fdr = sel_hist_fdr
        self.curr_round = curr_round
    
    def get_hist_dict(self):

        if self.curr_round == 1:
            past_corr_masks = {}
            
            return past_corr_masks

        for round in range(1, self.curr_round):
            corr_pth = os.path.join(self.sel_hist_fdr,
                                    f'Round{round}',
                                    'manual_corr_mask.json')
            with open(corr_pth) as f: 
                corr_masks = json.load(f)    
                        
            if round == 1:
                past_corr_masks = {img: list(set(objs))
                                    for img, objs in corr_masks.items()}  
                continue
            
            for img, objs in corr_masks.items():
                for obj in objs:
                    if img not in past_corr_masks:
                        past_corr_masks[img] = [obj]  
                    else:
                        if obj not in past_corr_masks[img]:  
                            past_corr_masks[img].append(obj)
                                
        past_corr_masks = {img: list(set(objs)) 
                            for img, objs in past_corr_masks.items()}  
        
        return past_corr_masks                         