# dataset for eval
# Code with dataset loader for VOC12, Cityscapes and CamVid (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#
# Shang Wei Hung modified in Juky 2018 for LDFNet purpose
#######################

import numpy as np
import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor,ToPILImage,Resize

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

	


class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
	
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset)           #RGB input
        self.depth_root = os.path.join(root, 'disparity/' + subset)              #Depth map
        self.labels_root = os.path.join(root, 'gtFine/' + subset)                #GroundTruth

        print (self.images_root)

        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
		
        self.filenamesDepth = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.depth_root)) for f in fn if is_image(f)]
        self.filenamesDepth.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform 

    def __getitem__(self, index):
	
        filename = self.filenames[index]
        filenameDepth = self.filenamesDepth[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            gray = load_image(f).convert('L')
       #L = R * 299/1000 + G * 587/1000+ B * 114/1000
        with open(image_path_city(self.depth_root, filenameDepth), 'rb') as f:
            depth = load_image(f).convert('F')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')
        
        
        depth=np.array(depth)
        depth=depth/100.0
     
        
        depth=torch.from_numpy(depth)
        depth=depth.float()
        depth=depth.unsqueeze(0)
  
        image=ToTensor()(image)

        
        image=torch.cat((image,depth),0)
        
        image=ToPILImage()(image)
		# image dim: 4
        if self.co_transform is not None:
            image,label = self.co_transform(image,gray,label)
		# image dim: 5

        return image, label

    def __len__(self):
        return len(self.filenames)

