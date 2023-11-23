# -*- coding: utf-8 -*-
"""
@author: gaohaoran@Dalian Minzu University
@software: PyCharm
@file: HFIL-Net.py
@time: 2023/11/23 7:23
"""

import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.HFILNet import SwinTransformer, HFILNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

# load the model
model = HFILNet()
model.load_state_dict(torch.load(''))
model.cuda()
model.eval()


test_datasets = []
for dataset in test_datasets:
    save_path = '' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth = depth.repeat(1, 3, 1, 1).cuda()
        out1, out2, out3 = model(image, depth)

        out1 = F.upsample(out1, size=gt.shape, mode='bilinear', align_corners=False)
        out2 = F.upsample(out2, size=gt.shape, mode='bilinear', align_corners=False)
        out3 = F.upsample(out3, size=gt.shape, mode='bilinear', align_corners=False)

        out1 = out1.sigmoid().data.cpu().numpy().squeeze()
        out2 = out2.sigmoid().data.cpu().numpy().squeeze()
        out3 = out3.sigmoid().data.cpu().numpy().squeeze()

        out1 = (out1 - out1.min()) / (out1.max() - out1.min() + 1e-8)
        out2 = (out2 - out2.min()) / (out2.max() - out2.min() + 1e-8)
        out3 = (out3 - out3.min()) / (out3.max() - out3.min() + 1e-8)
        print('save img to: ', save_path + name)

        cv2.imwrite(save_path + name, out3 * 255)
    print('Test Done!')

# #######################################################  end  ######################################
