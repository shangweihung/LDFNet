# PyTorch training tool functions
#
# Shang Wei Hung modified in Juky 2018 for LDFNet purpose
#######################

import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from PIL import Image, ImageOps
from torchvision.transforms import Resize, ToTensor

from transform import Relabel, ToLabel


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
		
        self.NLLLoss = nn.NLLLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        return self.NLLLoss(F.log_softmax(outputs, dim=1), targets)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    return model
	
	
#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
	# === Data Pre-Processing === #
	
    def __init__(self, encoderOnly, dataAugment=True, height=8.1):
	
        self.encoderOnly = encoderOnly
        self.dataAugment = dataAugment
        self.height = height
        pass
		
    def __call__(self, input, gray,target):
	
        # do something to both images
        if self.height != 8.1:
            input =  Resize(self.height, Image.BILINEAR)(input)
            gray =  Resize(self.height, Image.BILINEAR)(gray)
            target = Resize(self.height, Image.NEAREST)(target)
		
		# === Data Augmentation === #
        if(self.dataAugment):
            # 1. Random horizenal flip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                gray = gray.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 2. Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)
            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            gray = ImageOps.expand(gray, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            gray = gray.crop((0, 0, gray.size[0]-transX, gray.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        gray = ToTensor()(gray)
        if (self.encoderOnly):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)  # ignored label 255 -> 19

        input=torch.cat((input,gray),0)

        return input, target
		

def getClassWeight(dataName, classNum):

    classWeight = torch.ones(classNum)

    if dataName == 'cityscapes':
		# ERFNet code
        #'''
        classWeight[0] = 2.8149201869965	
        classWeight[1] = 6.9850029945374	
        classWeight[2] = 3.7890393733978	
        classWeight[3] = 9.9428062438965	
        classWeight[4] = 9.7702074050903	
        classWeight[5] = 9.5110931396484	
        classWeight[6] = 10.311357498169	
        classWeight[7] = 10.026463508606	
        classWeight[8] = 4.6323022842407	
        classWeight[9] = 9.5608062744141	
        classWeight[10] = 7.8698215484619	
        classWeight[11] = 9.5168733596802	
        classWeight[12] = 10.373730659485	
        classWeight[13] = 6.6616044044495	
        classWeight[14] = 10.260489463806	
        classWeight[15] = 10.287888526917	
        classWeight[16] = 10.289801597595	
        classWeight[17] = 10.405355453491	
        classWeight[18] = 10.138095855713   
        classWeight[19] = 0
        #'''
        '''
		# ERFNet formula
        classWeight[0] = 2.7	
        classWeight[1] = 6.8	
        classWeight[2] = 3.7	
        classWeight[3] = 8.9	
        classWeight[4] = 8.9	
        classWeight[5] = 9.5	
        classWeight[6] = 10.1	
        classWeight[7] = 9.9	
        classWeight[8] = 4.5	
        classWeight[9] = 8.9	
        classWeight[10] = 7.6
        classWeight[11] = 9.2	
        classWeight[12] = 10.1	
        classWeight[13] = 6.5	
        classWeight[14] = 8.8	
        classWeight[15] = 8.6	
        classWeight[16] = 7.4	
        classWeight[17] = 10.0	
        classWeight[18] = 9.8	    
        classWeight[19] = 0
        '''
        '''
		# ENet formula
        classWeight[0] = 3.3	
        classWeight[1] = 13.5	
        classWeight[2] = 4.9	
        classWeight[3] = 26.9	
        classWeight[4] = 26.9	
        classWeight[5] = 32.7	
        classWeight[6] = 43.3	
        classWeight[7] = 40.2	
        classWeight[8] = 6.5	
        classWeight[9] = 26.5	
        classWeight[10] = 17.3
        classWeight[11] = 30.1	
        classWeight[12] = 43.1	
        classWeight[13] = 12.2	
        classWeight[14] = 25.7	
        classWeight[15] = 23.9	
        classWeight[16] = 16.3	
        classWeight[17] = 40.3	
        classWeight[18] = 38.0	    
        classWeight[19] = 0
        '''
    elif dataName == 'camvid':
        classWeight[0] = 5.7	
        classWeight[1] = 4.4	
        classWeight[2] = 33.9	
        classWeight[3] = 3.5	
        classWeight[4] = 15.3	
        classWeight[5] = 8.0	
        classWeight[6] = 31.4	
        classWeight[7] = 23.2	
        classWeight[8] = 13.0	
        classWeight[9] = 36.9	
        classWeight[10] = 39.9	
        classWeight[11] = 28.0	
        classWeight[12] = 0
    elif dataName == 'itri':
        #'''
		# ENet formula
        classWeight[0] = 2.4	
        classWeight[1] = 2.3	
        classWeight[2] = 27.9
        classWeight[3] = 41.0
        classWeight[4] = 36.0
        classWeight[5] = 36.8
        #'''
        '''
		# ERFNet formula
        classWeight[0] = 2.2	
        classWeight[1] = 2.1	
        classWeight[2] = 9.1
        classWeight[3] = 10.0
        classWeight[4] = 9.7
        classWeight[5] = 9.8
        '''
        '''
		# No class balance
        classWeight[0] = 1	
        classWeight[1] = 1	
        classWeight[2] = 1
        classWeight[3] = 1
        classWeight[4] = 1
        classWeight[5] = 1
        '''

    return classWeight		

		