import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import PIL.Image as pimg

file_path = r"C:\Cartoon_faces"


class Sampling_Data(Dataset):
    def __init__(self, root, transform):
        self.file_names = []
        self.transforms = transform
        for file in os.listdir(root):
            file_name = os.path.join(root, file)
            self.file_names.append(file_name)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file = self.file_names[index]
        img_array = pimg.open(file)

        xs = self.transforms(img_array)

        return xs


data_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

sampling_data = Sampling_Data(root=file_path, transform=data_tf)

# datas = DataLoader(dataset=sampling_data,batch_size=100,shuffle=True)
# for data in datas:
#     print(data.size())
