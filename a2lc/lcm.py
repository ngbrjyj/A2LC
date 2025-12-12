import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class LCMasks(Dataset):
    def __init__(self, 
                 img_path, 
                 mask_path, 
                 feat_path, 
                 lbl_path, 
                 query_path, 
                 is_queried=True):
        self.img_names = []
        self.mask_indices = []
        self.features = []
        self.labels = []
        self.queried = []

        with open(img_path, "r") as f:
            self.img_names = f.read().splitlines()
        with open(mask_path, "r") as f:
            self.mask_indices = [int(line) for line in f.read().splitlines()]
        with open(feat_path, "r") as f:
            self.features = [list(map(float, line.split())) for line in f.read().splitlines()]
        with open(lbl_path, "r") as f:
            self.labels = list(map(int, f.read().splitlines()))
        with open(query_path, "r") as f:
            self.queried = [line for line in f.read().splitlines()]
        
        if is_queried: 
            self.indices = [i 
                            for i, flag in enumerate(self.queried) 
                            if (flag == 'True') and (self.labels[i] != 255)] 
        else:
            self.indices = [i 
                            for i, flag in enumerate(self.queried) 
                            if flag == 'False']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        
        image = self.img_names[idx]
        mask = self.mask_indices[idx]
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = self.labels[idx]

        return image, mask, feature, label

class LCModule(nn.Module):
    def __init__(self, num_classes):
        super(LCModule, self).__init__()        
        self.fc1 = nn.Linear(num_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        outputs = F.softmax(logits, dim=1)
        
        return outputs