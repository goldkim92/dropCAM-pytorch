import os
import collections
from os.path import join

import torch
import torch.nn as nn
from torchvision.models import alexnet


class Alexnet_for_CAM(nn.Module):
    def __init__(self):
        super(Alexnet_for_CAM, self).__init__()
        self.ckpt_path = join('model','alexnet','checkpoint_epoch48.pth.tar')
        
        # get the pretrained GoogleNet_Drop network
        self.model = AlexNet_Drop()
        self.load_pretrained_model()
        
        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(
            self.model.features_conv,
            self.model.added_conv,
        )
        
        # get the max pool of the features stem
        self.max_pool = self.model.max_pool
#         self.avg_pool = self.model.avgpool
        
        # get the classifier of the vgg19
        self.classifier = self.model.classifier
        
        # delete self.model variable
        del self.model
        
        # placeholder for the gradients and feature_conv
        self.gradients = None
        self.features = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        self.features = self.features_conv(x)
        
        # register the hook
        h = self.features.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(self.features)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self):
        return self.features

    def load_pretrained_model(self):
        if not os.path.isfile(self.ckpt_path):
            file_id = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=self.ckpt_path)
        checkpoint = torch.load(self.ckpt_path)
        if self.ckpt_path.endswith('pth.tar'):
            state_dict = collections.OrderedDict()
            for k,v in checkpoint['state_dict'].items():
                state_dict[k[7:]] = v
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        

class AlexNet_Drop(nn.Module):
    def __init__(self):
        super(AlexNet_Drop, self).__init__()
        
        # get the pretrained alexnet network
        model = alexnet(pretrained=True)

        # dissect the network
        self.features_conv = model.features[:-1]

        # additional layers
        self.added_conv = nn.Sequential(
            nn.Conv2d(256,512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.max_pool = model.features[-1:]

        self.classifier = nn.Sequential(
            nn.Linear(512*6*6, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000, bias=True),
        )

        # delete the origin model
        del model

