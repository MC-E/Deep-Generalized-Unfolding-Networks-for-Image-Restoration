"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from DGUNet import DGUNet
# from DGUNet_plus import DGUNet
# from MPRNet_nochop_s7_v7_hi3_u4_baseSAM import MPRNet
# from MPRNet import MPRNet
# from MPRNet_nochop_s7_v7_hi3_u4_rp_base import MPRNet
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx
import time
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Image Denoising using MPRNet')

parser.add_argument('--input_dir', default='./testset', type=str, help='Directory of validation images')
# parser.add_argument('--result_dir', default='./results/DND/MPRNet_nochop_s7_v7_hi3_u4_baseSAM', type=str, help='Directory for results')
# parser.add_argument('--weights', default='ckp_off/DGUNet_plus.pth', type=str, help='Path to weights')
parser.add_argument('--weights', default='ckp_off/DGUNet.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if args.save_images:
    result_dir_img = os.path.join(args.result_dir, 'png')
    utils.mkdir(result_dir_img)
# list_ = []
list_ = ["stage1_encoder.body.0.csff_enc.weight", "stage1_encoder.body.0.csff_enc.bias", "stage1_encoder.body.0.csff_dec.weight", "stage1_encoder.body.0.csff_dec.bias", "stage1_encoder.body.0.phi.weight", "stage1_encoder.body.0.phi.bias", "stage1_encoder.body.0.gamma.weight", "stage1_encoder.body.0.gamma.bias", "stage1_encoder.body.1.csff_enc.weight", "stage1_encoder.body.1.csff_enc.bias", "stage1_encoder.body.1.csff_dec.weight", "stage1_encoder.body.1.csff_dec.bias", "stage1_encoder.body.1.phi.weight", "stage1_encoder.body.1.phi.bias", "stage1_encoder.body.1.gamma.weight", "stage1_encoder.body.1.gamma.bias", "stage1_encoder.body.2.csff_enc.weight", "stage1_encoder.body.2.csff_enc.bias", "stage1_encoder.body.2.csff_dec.weight", "stage1_encoder.body.2.csff_dec.bias", "stage1_encoder.body.2.phi.weight", "stage1_encoder.body.2.phi.bias", "stage1_encoder.body.2.gamma.weight", "stage1_encoder.body.2.gamma.bias"]
checkpoint = torch.load(args.weights)
state_dict = checkpoint["state_dict"]
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    fl = True
    for name_i in list_:
        if name_i in k:
            fl=False
            break
    if fl == True:
        new_state_dict[k[7:]] = v
            
model_restoration = DGUNet()
model_restoration.load_state_dict(new_state_dict)

new_state_dict_re = OrderedDict()
for k, v in state_dict.items():
    fl = True
    for name_i in list_:
        if name_i in k:
            fl=False
            break
    if fl == True:
        new_state_dict_re[k] = v
ckp={}
ckp['state_dict'] = new_state_dict_re
torch.save(ckp, 'ckp_off_my/DGUNet.pth')
# torch.save(new_state_dict_re, 'ckp_off_my/DGUNet_plus.pth')



# utils.load_checkpoint(model_restoration,args.weights)
# print("===>Testing using weights: ",args.weights)
# model_restoration.cuda()
# model_restoration = nn.DataParallel(model_restoration)
# model_restoration.eval()
