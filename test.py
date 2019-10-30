import os
import sys
import csv
import argparse
import numpy as np
from PIL import Image
from os.path import join
from tqdm import tqdm, tqdm_notebook

from cam import CAM
import util


# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='')

parser.add_argument('--method', type=str, default='vanilla', help='vanilla or ours')
parser.add_argument('--model', type=str, default='vgg', help='vgg or resnet or googlenet')
parser.add_argument('--gpu_number', type=str, default='0')
parser.add_argument('--th1',     type=float, default=0.2, help='threshold for the heatmap mean value')
parser.add_argument('--th2',     type=float, default=0.4, help='threshold for the heatmap std value, only used in "ours" method')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--runs_dir', type=str, default='')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

if args.runs_dir == '':
    args.runs_dir = args.model

if args.method == 'vanilla':
    args.runs_dir = join('runs', args.runs_dir, f'{args.th1}')
elif args.method == 'ours':
    args.runs_dir = join('runs', args.runs_dir, f'{args.th1}_{args.th2}')
elif args.method == 'ours_mean':
    args.runs_dir = join('runs', args.runs_dir, f'{args.th1}_mean')
else:
    raise Exception("'method' must be either 'vanilla' or 'ours'")

if not os.path.exists(args.runs_dir):
    os.makedirs(args.runs_dir)


def write_log(string):
    with open(join(args.runs_dir,'log.log'), 'a') as lf:
        sys.stdout = lf
        print(string)


def write_csv(bboxes):
    with open(join(args.runs_dir,'bbox.csv'), 'a') as cf:
        w = csv.writer(cf)
        for k,v in bboxes.items():
            w.writerow([k,v])


# ===========================================================
# main
# ===========================================================
if __name__ == "__main__":
    map = CAM(args.model)

    data_dict = map.valid_dataset.data_dict
    input_files = map.valid_dataset.img_files
    img_dir = map.valid_dataset.img_dir

    bboxes_pred = {}

    write_log('Localization Accuracy')

    count = 0
    correct = 0
    for data_idx in range(len(data_dict)):
        count += 1

        # get true bbox
        input_file = input_files[data_idx]
        img_origin = Image.open(join(img_dir, input_file)).convert('RGB')
        bboxes_true = data_dict[input_file][1]
        bboxes_true = util.bboxes_resize(img_origin, bboxes_true, size=224)

        # get target
        _, target = map.get_item(data_idx)
        target = target.cpu().item()

        if args.method == 'vanilla':
            ''' CAM origin version '''
            _, _, _, _, bbox_pred = map.get_values(data_idx, target, th1=args.th1, phase='test')
        elif args.method == 'ours':
            ''' CAM propose version '''
            _, _, _, _, _, bbox_pred = map.get_values(data_idx, target, th1=args.th1, th2=args.th2, mc=20, phase='train')
        elif args.method == 'ours_mean':
            _, heatmap_mean, _, _, _, _ = map.get_values(data_idx, target, th1=args.th1, th2=args.th2, mc=20, phase='train')
            boolmap_mean = util.heatmap2boolmap(heatmap_mean, a=args.th1)
            boolmap_biggest = util.get_biggest_component(boolmap_mean)
            bbox_pred = util.boolmap2bbox(boolmap_biggest)

        ''' get iou '''
        iou = []
        for bbox_true in bboxes_true:
            iou.append(util.get_iou(bbox_true, bbox_pred))
        correct += max(np.array(iou) >= 0.5).astype(np.int)

        ''' save bboxes for every 100 iteration '''
        bboxes_pred[input_file] = bbox_pred

        if (data_idx+1) % 100 == 0:
            # print the log
            write_log(f'iter {data_idx+1:05d} ===> Propose: {correct/count}')
            
            # save bbox in csv file
            write_csv(bboxes_pred)
            
            bboxes_pred = {}


