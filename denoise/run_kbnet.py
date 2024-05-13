import os
import yaml
import argparse
from natsort import natsorted
from glob import glob

import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
import scipy.io as sio

import cv2
import torch
import torch.nn as nn
import KBNet.Denoising.utils_tool

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from KBNet.basicsr.models.archs.kbnet_s_arch import KBNet_s

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('-i', '--input_dir', type=str, help='Directory of validation images')
parser.add_argument('-o', '--output_dir', type=str, help='Directory for results')
parser.add_argument('-y', '--yml', default="kbnet.yml", required=False, type=str, help='path to config file')

args = parser.parse_args()

yaml_file = args.yml
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

cfg_name = os.path.basename(yaml_file).split('.')[0]

pth_path = x['path']['pretrain_network_g']
print('**', yaml_file, pth_path)

s = x['network_g'].pop('type')

model_restoration = eval(s)(**x['network_g'])

checkpoint = torch.load(pth_path)
model_restoration.load_state_dict(checkpoint['model'])
print("===>Testing using weights: ")
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

########################

input_dir = args.input_dir
out_dir = args.output_dir

os.makedirs(out_dir, exist_ok=True)
extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
files = natsorted(glob(os.path.join(input_dir, '*')))
img_multiple_of = 8

def batch_generator(gen, batch_size=1):
    i = 0
    images = []
    for img in gen:
        images.append(img)
        if len(images) == batch_size:
            yield images
            images = []
    if images != []:
        yield images

with torch.no_grad():
    i = 0
    for filepaths in batch_generator(tqdm(files)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        imgs = []
        for filepath in filepaths:
            img = np.float32(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB) / 255.)
            noisy = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
            imgs.append(noisy)
        
        
        noisy = torch.cat(imgs, 0).cuda()

        restored = model_restoration(noisy)
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1)

        restored = img_as_ubyte(restored)

        for filepath, rest in zip(filepaths, restored):
            filename = os.path.split(filepath)[-1]
            cv2.imwrite(os.path.join(out_dir, filename), cv2.cvtColor(rest, cv2.COLOR_RGB2BGR))


# with torch.no_grad():
#     i = 0
#     for filepath in tqdm(files):
#         torch.cuda.ipc_collect()
#         torch.cuda.empty_cache()
#         img = np.float32(cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB) / 255.)
        
#         noisy = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        
#         restored = model_restoration(noisy)
#         restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)

#         restored = img_as_ubyte(restored)
#         filename = os.path.split(filepath)[-1]
#         cv2.imwrite(os.path.join(out_dir, filename), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
