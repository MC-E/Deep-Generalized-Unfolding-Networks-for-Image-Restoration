import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from DGUNet import DGUNet
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Denoising using MPRNet')

parser.add_argument('--input_dir', default='./testset/DND_patches', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/DND/demo', type=str, help='Directory for results')
parser.add_argument('--weights', default='pretrained_models/DGUNet.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

result_dir = os.path.join(args.result_dir, 'mat')
utils.mkdir(result_dir)

if args.save_images:
    result_dir_img = os.path.join(args.result_dir, 'png')
    utils.mkdir(result_dir_img)

model_restoration = DGUNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

def augment_img_tensor(img, mode=0):
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def test_x8(model, L):
    E_list = [model(augment_img_tensor(L, mode=i))[0] for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E

import glob
import cv2
im_list = glob.glob(os.path.join(args.input_dir,'*.bmp'))
with torch.no_grad():
    print(im_list)
    for idx,path_i in enumerate(im_list):
        image = cv2.cvtColor(cv2.imread(path_i),cv2.COLOR_BGR2RGB)/255.
        image = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2).cuda().float()
        restored_patch = model_restoration(image)
#         restored_patch = test_x8(model_restoration, image)
        restored_patch = torch.clamp(restored_patch[0], 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
        restored_patch = cv2.cvtColor((restored_patch.numpy()*255.).astype(np.uint8),cv2.COLOR_RGB2BGR)
        save_file = os.path.join(args.result_dir,'images', str(idx)+'_b.png')
        cv2.imwrite(save_file,restored_patch)