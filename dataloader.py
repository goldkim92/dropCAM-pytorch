import os
from os.path import join, expanduser
from PIL import Image
import json
import xml.etree.ElementTree as ET

import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

import util

def get_class_dict(base_dir):
    '''
    Returns:
        class_dict_idx2ctgr:  ex) {0: 'tench', ...} where ctgr is category
        class_dict_name2idx: ex) {'n01440764': 0, ...}
    '''
    with open(join(base_dir, 'imagenet_class_index.json'),'r') as json_file:
        class_dict = json.load(json_file)

    class_dict_idx2ctgr = {int(k):v[1] for k,v in class_dict.items()}
    class_dict_name2idx = {v[0]:int(k) for k,v in class_dict.items()}

    return class_dict_idx2ctgr, class_dict_name2idx


def get_data_dict(lbl_dir, lbl_files, class_dict_name2idx):
    '''
    Returns:
        data_dict: dictionary format where
                   key: image file, ex) 'ILSVRC2012_val_00000002.JPEG'
                   value: tuple format with target and bboxes in it
                          ex) (970,[(45, 49, 454, 113), (2, 69, 435, 138)])
    '''
    data_dict = {}

    for lbl_file in lbl_files:
        # parser xml file
        path = join(lbl_dir, lbl_file)
        doc = ET.parse(path)

        # get root node
        root = doc.getroot()

        # extract img_file from node
        img_file = root.findtext('filename') + '.JPEG'

        # extract target, bboxes from nodes
        object_tags = root.findall('object')
        bboxes = []
        for object_tag in object_tags:
            name = object_tag.findtext('name')
            target = class_dict_name2idx[name]

            bbox = object_tag.find('bndbox')
            xmin = int(bbox.findtext('xmin'))
            ymin = int(bbox.findtext('ymin'))
            xmax = int(bbox.findtext('xmax'))
            ymax = int(bbox.findtext('ymax'))
            bbox = (xmin, ymin, xmax-xmin, ymax-ymin)
            bboxes.append(bbox)

        # save the values in data_dict
        data_dict[img_file] = (target, bboxes)

    return data_dict


class FolderDataset(Dataset):
    def __init__(self, t_input):
        self.t_input = t_input
        self.base_dir = join(expanduser('~'),'data','imagenet')
        self.img_dir = join(self.base_dir, 'valid_img')
        self.lbl_dir = join(self.base_dir, 'valid_lbl')

        self.load_files()
        self.make_dataset()

    def load_files(self):
        self.img_files = os.listdir(self.img_dir)
        self.img_files.sort(key=util.natural_keys)

        self.lbl_files = os.listdir(self.lbl_dir)
        self.lbl_files.sort(key=util.natural_keys)

    def make_dataset(self):
        _, class_dict_name2idx = get_class_dict(self.base_dir)
        self.data_dict = get_data_dict(self.lbl_dir,
                                       self.lbl_files,
                                       class_dict_name2idx)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        path = join(self.img_dir, img_file)

        input = Image.open(path).convert('RGB')
        input = self.t_input(input)

        target = self.data_dict[img_file][0]

        return input, target

    def __len__(self):
        return len(self.data_dict)


def imagenet_loader(bs=32):
    t_input = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

    imagenet_dataset = FolderDataset(t_input)

    loader = DataLoader(imagenet_dataset, batch_size=bs,
                    shuffle=False, num_workers=2)

    return loader
