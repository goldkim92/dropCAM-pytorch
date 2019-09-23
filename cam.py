import os
from os.path import join
import numpy as np
from PIL import Image

import torch
import torchvision as tv

import dataloader
import model
import util


class CAM(object):
    def __init__(self, model_type):
        self.model_type = model_type 
        self.batch_size = 1 
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load data & build model
        self.load_dataset()
        self.build_model()


    def load_dataset(self):
        t_input = tv.transforms.Compose([
            tv.transforms.Resize((224,224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                    std=(0.229, 0.224, 0.225)),
        ])  
        self.valid_dataset = dataloader.FolderDataset(t_input)


    def build_model(self):
        if self.model_type == 'vgg':
            self.model = model.VGG_for_CAM()
        elif self.model_type == 'resnet':
            self.model = model.Resnet_for_CAM()
        elif self.model_type == 'googlenet':
            self.model = model.Googlenet_for_CAM()
        else:
            raise Exception("'model_type' should in one of the ['vgg','resnet','googlenet']")
            
        self.model = self.model.to(self.device)


    def get_item(self, index):
        input, target = self.valid_dataset[index]
        input, target = input.unsqueeze(0), torch.tensor(target)
        input, target = input.to(self.device), target.to(self.device)
        return input, target
            

    def topk(self, input):
        self.model.eval()
        score = self.model(input)
        topk_idxs = score.topk(1000)[1].squeeze(0)
        return topk_idxs

    
    def activation(self, input, att_idx, phase='test'):
        # model phase
        if phase == 'test':
            self.model.eval()
        else:
            # batchnorm to eval mode, and dropout to train mode
            self.model.train()
            for name, module in self.model._modules.items():
                if name == 'features_conv':
                    module.eval()

        # get the gradient of the output with respect to the parameters of the model
        score = self.model(input) 
        score[:, att_idx].backward(retain_graph=True)

        # pull the gradients out of the model
        gradients = self.model.get_activations_gradient()
        gradients = gradients.cpu().detach().squeeze(0) # size = [512,14,14]

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[1, 2], keepdim=True) # size = [512,1,1]

        # get the activations of the last convolutional layer
        activations = self.model.get_activations()
        activations = activations.cpu().detach().squeeze(0) # size = [512,14,14]

        # weight the channels by corresponding gradient
        grad_cam = activations * pooled_gradients
        grad_cam = grad_cam.mean(dim=0)
        grad_cam = torch.max(grad_cam, torch.tensor(0.))

        return grad_cam

    
    def get_heatmap(self, input, att_idx, phase):
        grad_cam = self.activation(input, att_idx, phase)
        heatmap = util.cam2heatmap(grad_cam)
        return heatmap
    
    
    def get_heatmaps(self, input, att_idx, mc):
        heatmaps = []
        for _ in range(mc):
            heatmap = self.get_heatmap(input, att_idx, phase='train')
            heatmaps.append(heatmap)
        heatmaps = np.array(heatmaps)
        return heatmaps


    def get_values(self, data_idx, att_idx, th1=0.2, th2=10, mc=30, phase='test'):
        # get input, target, and img (PIL.Image format)
        input, target = self.get_item(data_idx)
        img = util.torch2pil(input)

        # make boolmap from heatmap
        if phase == 'test':
            heatmap = self.get_heatmap(input, att_idx, phase='test')
            boolmap = util.heatmap2boolmap(heatmap, a=th1)
        else:
            heatmaps = self.get_heatmaps(input, att_idx, mc=mc)
            heatmap_mean = heatmaps.mean(0)
            heatmap_std = heatmaps.std(0)
            
            boolmap_mean = util.heatmap2boolmap(heatmap_mean, a=th1)
            boolmap_std = util.heatmap2boolmap(heatmap_std, a=th2)
            boolmap = np.logical_or(boolmap_mean, boolmap_std)
            
        # segment the biggest component
        boolmap_biggest = util.get_biggest_component(boolmap)
        
        # get bbox
        bbox = util.boolmap2bbox(boolmap_biggest)
        
        if phase == 'test':
            return img, heatmap, boolmap, boolmap_biggest, bbox
        else:
            return img, heatmap_mean, heatmap_std, boolmap, boolmap_biggest, bbox
