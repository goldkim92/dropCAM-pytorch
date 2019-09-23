import numpy as np
from PIL import Image
import torchvision as tv

import re
import os
from os.path import join
from scipy import ndimage

import torch


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnormed_tensor = torch.zeros_like(tensor)
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            unnormed_tensor[i] = t.mul(s).add(m)
            # The normalize code -> t.sub_(m).div_(s)                                                                                                                                                       
        return unnormed_tensor

    
def torch2pil(input):
    '''
    Args:
        input (torch.cuda.tensor): normalized image of size (1, C, H, W)
    Returns:
        img (PIL.Image): size (W,H)
    '''
    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    img = input.cpu().squeeze(0)
    img = unnorm(img)
    img = tv.transforms.ToPILImage()(img)
    return img

    
def cam2heatmap(cam):
    '''
    Args:
        cam (torch.tensor): activation map with size (14,14)
    Returns:
        heatmap (np.array): heatmap with shape (224,224)
    '''
    cam /= cam.max()
    heatmap = Image.fromarray(np.array(cam*255).astype(np.uint8))
    heatmap = heatmap.resize((224,224), resample=Image.BILINEAR)
    heatmap = np.array(heatmap).astype(np.float32)
    return heatmap


def heatmap2boolmap(heatmap, a=0.2):
    '''
    Args:
        heatmap (np.array): heatmap with shape (224,224)
    Returns:
        boolmap (np.bool): shape (224,224)
    '''
    if isinstance(a, float):
        threshold = heatmap.max() * a
    elif isinstance(a, int):
        threshold = a
    
    boolmap = heatmap >= threshold
    return boolmap
    

def get_biggest_component(boolmap):
    '''
    Args:
        boolmap (np.bool): shape (224, 224)
    Returns:
        boolmap_biggest (np.bool): shape (224, 224)
    '''
    segments, nb = ndimage.label(boolmap)
    frequent_value = max(range(1,nb+1), key=segments.flatten().tolist().count)
    boolmap_biggest = segments==frequent_value
    return boolmap_biggest


def boolmap2bbox(boolmap):
    '''
    Args:
        boolmap (np.bool): shape (224,224)
    Returns:
        bbox (tuple): (xmin, ymin, width, height)
    '''
    mask_coords = np.transpose(np.nonzero(boolmap)) # size = [_,2]

    ymin, xmin = mask_coords.min(0)
    ymax, xmax = mask_coords.max(0)
    bbox = (int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin))
    return bbox


def bboxes_resize(img, bboxes, size=224):
    w,h = img.size

    bboxes_resized = []
    for bbox in bboxes:
        bbox = np.array(bbox)
        bbox[0::2] = bbox[0::2] / w * size
        bbox[1::2] = bbox[1::2] / h * size
        bbox = tuple(bbox.astype(np.int))
        bboxes_resized.append(bbox)
    return bboxes_resized



def get_iou(boxA, boxB):
    # change the box format to (xmin, ymin, xmax, ymax)
    boxA = (boxA[0], boxA[1], boxA[0]+boxA[2], boxA[1]+boxA[3])
    boxB = (boxB[0], boxB[1], boxB[0]+boxB[2], boxB[1]+boxB[3])

	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the iou
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

#############################################################################
#############################################################################
#############################################################################

# Sort a string with a number inside
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]




