import os
import pretrainedmodels
from os.path import join
from google_drive_downloader import GoogleDriveDownloader as gdd

import torch
import torch.nn as nn


class Resnet_for_CAM(nn.Module):
    def __init__(self):
        super(Resnet_for_CAM, self).__init__()
        
        # get the pretrained VGG19 network
        self.model = resnet50se_drop()
        self.load_pretrained_model()
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.model.features_conv
        
        # get the max pool of the features stem
        self.avg_pool = self.model.avg_pool
        
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
        x = self.avg_pool(self.features)
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
        ckpt_path = join('model','resnet50se_drop.pth')
        file_id = '1g_ZxNlH_WKmeGuplhFtKokKblxsJU0e4'
        if not os.path.isfile(ckpt_path):
            gdd.download_file_from_google_drive(file_id=file_id, dest_path=ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path))



class resnet50se_drop(nn.Module):
    def __init__(self):
        super(resnet50se_drop, self).__init__()

        # get the pretrained ResNet50SE network
        model_name = 'se_resnet50'
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

        # dissect the network
        self.features_conv = nn.Sequential(
            model.layer0,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avg_pool = model.avg_pool

        # add 3 dropout layers
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1000, bias=True),
        )

        # delete original model
        del model

#     def forward(self, x):
#         self.features = self.features_conv(x)
#         x = self.avg_pool(self.features)
#         x = x.view((x.size(0), -1))
#         x = self.classifier(x)
#         return x
# 
#     def __call__(self):
#         return model
