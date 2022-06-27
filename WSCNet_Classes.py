'''
@article{She_2019_TMM,
	Author = {She, Dongyu and Yang, Jufeng and Cheng, Ming-Ming and Lai, Yu-Kun and Rosin, Paul L. and Wang, Liang},
	Title = {WSCNet: Weakly Supervised Coupled Networks for Visual Sentiment Classification and Detection},
	journal = {IEEE Transactions on Multimedia},
	Year = {2019}
}

@InProceedings{Yang_2018_CVPR,
	Author = {Yang, Jufeng and She, Dongyu and Lai, Yu-Kun and Rosin, Paul L. and Yang, Ming-Hsuan},
	Title = {Weakly Supervised Coupled Networks for Visual Sentiment Analysis},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition},
	Year = {2018}
}
'''

from __future__ import print_function 
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt
import time
import os
import copy
from datetime import datetime

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

img_save_path = r"C:\Users\Nutzer\Desktop\WSCNet_Predictor_final/"

def show_imgtens(imgtens):
    temp = None
    try:
        plt.close()
        if torch.is_tensor(imgtens):
            temp = imgtens[0]
        elif isinstance(imgtens, (np.ndarray)):
            temp = imgtens
        else: 
            print("TypeError, data must be tensor or np.array: ", TypeError)

    finally:
        if temp is not None:
            plt.imshow(transforms.ToPILImage()(temp), interpolation="bicubic")
            timestmp = datetime.now().strftime("%H_%M_%S")
            plt.savefig(img_save_path + 'exampleimg_{t}'.format(t=timestmp))
        return None


class ResNetWSL(nn.Module):
    
    def __init__(self, model, num_classes, num_maps, pooling, pooling2):
        super(ResNetWSL, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_ftrs = model.fc.in_features

        self.downconv = nn.Sequential(
            nn.Conv2d(2048, num_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.GAP = nn.AvgPool2d(14)
        self.GMP = nn.MaxPool2d(14)
        self.spatial_pooling = pooling
        self.spatial_pooling2 = pooling2
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
            )
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x_ori = x  

        # detect branch
        x = self.downconv(x) 
        x_conv = x
        x = self.GMP(x)  # self.GAP(x) cross-spatial pooling
        x = self.spatial_pooling(x) 
        x = x.view(x.size(0), -1)
        x = self.softmax(x) #Aktivierung

        # classification branch
        x_conv = self.spatial_pooling(x_conv) 
        x_conv = x_conv * x.view(x.size(0),x.size(1),1,1) #Coupling
        x_conv = self.spatial_pooling2(x_conv) 
        x_conv_copy = x_conv

        #Concatenation / Verkettung
        for num in range(0,2047):            
            x_conv_copy = torch.cat((x_conv_copy, x_conv),1)
        x_conv_copy = torch.mul(x_conv_copy,x_ori)
        x_conv_copy = torch.cat((x_ori,x_conv_copy),1) 

        x_conv_copy = self.GAP(x_conv_copy) #Semantic Vector 
        x_conv_copy = x_conv_copy.view(x_conv_copy.size(0),-1)
        x_conv_copy = self.classifier(x_conv_copy) # Lin Layer
        x_conv_copy = self.softmax(x_conv_copy) # LogSoftmax
        return x, x_conv_copy #returnt outputs des detection- und classification branches



class ClassWisePoolFunction(Function):
    def __init__(self, num_maps):
        super(ClassWisePoolFunction, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        # batch dimension
        batch_size, num_channels, h, w = input.size()

        if num_channels % self.num_maps != 0:
            print('Error in ClassWisePoolFunction. The number of channels has to be a multiple of the number of maps per class')
            sys.exit(-1)

        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, h, w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_outputs, h, w) / self.num_maps

    def backward(self, grad_output):
        input, = self.saved_tensors
        # batch dimension
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)

        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(batch_size, num_outputs, self.num_maps,
                                                                               h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w)




class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction(self.num_maps)(input)