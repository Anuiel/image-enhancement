import os
import yaml
import argparse

import numpy as np
from tqdm import tqdm
from skimage import img_as_ubyte
import scipy.io as sio

import torch
import torch.nn as nn
import utils_tool

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


import torch
import torchvision.models as models
from collections import OrderedDict
import numpy as np
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

def add_gaussian_noise(X_img, var=1):
    gaussian = np.random.normal(loc=0.0, scale=var, size=(X_img.shape[0], X_img.shape[1], 1))
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
    gaussian_img = X_img + var * gaussian
    return np.clip(gaussian_img, 0, 1)

def to_tensor(img):
    return torch.from_numpy(np.expand_dims(img, 0).transpose(0, 3, 1, 2)).to(torch.float)

        
def psnr_metric(image1, image2):
    return compare_psnr(image1, image2)
def ssim_metric(image1, image2):
    return compare_ssim(image1, image2, channel_axis=2, data_range=1, multichannel=True)
    

from basicsr.models.archs.kbnet_s_arch import KBNet_s

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')


parser.add_argument('--input_dir', default='./Datasets/test/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--true_answer_dir', default='')
parser.add_argument('--result_dir', default='./results/Real_Denoising/SIDD/', type=str, help='Directory for results')
parser.add_argument('--save_images', default=True, help='Save denoised images in result directory')
parser.add_argument('--yml', default=None, type=str, help='Directory for results')

args = parser.parse_args()

yaml_file = args.yml
x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

cfg_name = os.path.basename(yaml_file).split('.')[0]

# mat if datasets are in mat format
# images in datasets are folders of images
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

class FormatChanger():
    def __init__(self, model):
        self.model = model
    def __call__(self, patch):
        batches = len(patch.shape) == 4
        if not batches:
            patch = patch.unsqueeze(0)
        patch = patch.permute(0, 3, 1, 2)
        ans = self.model(patch)
        ans = torch.clamp(ans, 0, 1).permute(0, 2, 3, 1)
        if not batches:
            ans = ans.squeeze(0)
        return ans
model = FormatChanger(model_restoration)

########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

def load_mat(filepath):
    img = sio.loadmat(filepath)
    for key, val in img.items():
        if key[:2] == "__":
            continue
        if type(val) == np.ndarray:
            return np.float32(np.array(val)) / 255.

from torch.utils.data import Dataset

class MatDenoisingDataset(Dataset):
    def __init__(self, input_file, output_file=None):
        super(MatDenoisingDataset, self).__init__()
        self.input_file = input_file
        self.noisy = load_mat(input_file)

        if output_file is None:
            self.output = False
        else:
            self.output = True

        if self.output:
            self.output_file = output_file
            self.clean = load_mat(output_file)
        

    def __len__(self):
        return 32 * 40

    def __getitem__(self, ind):
        # return 1
        i = ind // 32
        k = ind % 32
        if self.output:
            return (i, k), torch.from_numpy(self.noisy[i, k, :, :, :]), torch.from_numpy(self.clean[i, k, :, :, :])
        else:
            return (i, k), torch.from_numpy(self.noisy[i, k, :, :, :])
        

gt = load_mat(os.path.join(args.true_answer_dir, 'ValidationGtBlocksSrgb.mat'))
print('gt', gt.shape, gt.max(), gt.min())
res = {'psnr': [], 'ssim': []}

Inoisy = load_mat(os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat'))
restored = np.zeros_like(Inoisy)

dataset = MatDenoisingDataset(
    os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat'),
    os.path.join(args.true_answer_dir, 'ValidationGtBlocksSrgb.mat')
)

print(dataset.noisy.shape, dataset.clean.shape)
with torch.no_grad():
    for (i, k), noisy_patch, clean_patch in tqdm(dataset):
        noisy_patch = noisy_patch.cuda()
        clean_patch = clean_patch.cuda()
        
        # restored_patch = model_restoration(noisy_patch)
        # restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)

        restored_patch = model(noisy_patch).cpu().detach()
        
        restored[i, k, :, :, :] = restored_patch
        res['psnr'].append(compare_psnr(gt[i, k], restored_patch.numpy()))
        res['ssim'].append(compare_ssim(gt[i, k], restored_patch.numpy(), channel_axis=2, data_range=1, multichannel=True))

# with torch.no_grad():
#     for i in tqdm(range(1)):
#         for k in range(32):
#             # noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
#             noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).cuda()
            
#             # restored_patch = model_restoration(noisy_patch)
#             # restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)

#             restored_patch = model(noisy_patch).cpu().detach()
            
#             restored[i, k, :, :, :] = restored_patch

#             # save psrn and ssim
#             # print(type(restored_patch))  # torch.Tensor

#             res['psnr'].append(compare_psnr(gt[i, k], restored_patch.numpy()))
#             res['ssim'].append(compare_ssim(gt[i, k], restored_patch.numpy(), channel_axis=2, data_range=1, multichannel=True))
#             if args.save_images:
#                 save_file = os.path.join(result_dir_png,
#                                          '%04d_%02d_%.2f_%s.png' % (i + 1, k + 1, res['psnr'][-1], cfg_name))
#                 utils_tool.save_img(save_file, img_as_ubyte(restored_patch))

print(f'{cfg_name} psnr %.2f ssim %.3f' % (np.mean(res['psnr']), np.mean(res['ssim'])))

# save denoised data
# sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored, })
