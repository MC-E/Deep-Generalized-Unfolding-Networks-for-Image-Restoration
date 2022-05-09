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

parser.add_argument('--input_dir', default='./testset', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results/DND/DGUNet_x8', type=str, help='Directory for results')
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

israw = False
eval_version="1.0"

# Load info
infos = h5py.File(os.path.join(args.input_dir, 'info.mat'), 'r')
info = infos['info']
bb = info['boundingboxes']

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

# Process data
with torch.no_grad():
    for i in tqdm(range(50)):
        Idenoised = np.zeros((20,), dtype=np.object)
        filename = '%04d.mat'%(i+1)
        filepath = os.path.join(args.input_dir, 'images_srgb', filename)
        img = h5py.File(filepath, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)

        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T

        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            noisy_patch = torch.from_numpy(Inoisy[idx[0]:idx[1],idx[2]:idx[3],:]).unsqueeze(0).permute(0,3,1,2).cuda()
#             restored_patch = model_restoration(noisy_patch)
            restored_patch = test_x8(model_restoration, noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            Idenoised[k] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_img, '%04d_%02d.png'%(i+1,k+1))
                denoised_img = img_as_ubyte(restored_patch)
                utils.save_img(save_file, denoised_img)

        # save denoised data
        sio.savemat(os.path.join(result_dir, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )