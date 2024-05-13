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

from basicsr.models.archs.kbnet_s_arch import KBNet_s

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def image_to_tensor(image):
    image = torch.from_numpy(image)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return image.permute(0, 3, 1, 2).cuda()

def tensor_to_image(tensor):
    return torch.clamp(tensor, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
    
# with torch.no_grad():

class KBNet_config:
    base = "/denisrtyhb/ImageEnchancer/KBNet/Denoising/"
    input_dir = os.path.join(base, './Datasets/test/SIDD/')
    result_dir = os.path.join(base, './results/Real_Denoising/SIDD/')
    yaml_file = os.path.join(base, "./Options/sidd.yml")
    patch_size = (256, 256, 3)

class KBNet_model:
    def __init__(
        self,
        config: KBNet_config,
        device: str
    ):
        self.config = config
        x = yaml.load(open(config.yaml_file, mode='r'), Loader=Loader)
        
        pth_path = x['path']['pretrain_network_g']
        
        s = x['network_g'].pop('type')
        
        self.model = eval(s)(**x['network_g'])
        
        checkpoint = torch.load(pth_path)
        self.model.load_state_dict(checkpoint['model'])
        
        self.model.to(device)
        #model_restoration = nn.DataParallel(model_restoration)
    def eval():
        self.model.eval()
    def train():
        self.model.train()

    def __call__(self, image):
        assert image.shape == self.config.patch_size or image.shape[1:] == self.config.patch_size
        
        with torch.no_grad():
            noisy_patch = image_to_tensor(image).float()
            noisy_patch = torch.clip(noisy_patch, 0, 1)
            restored_patch = self.model(noisy_patch)
            restored_patch = tensor_to_image(restored_patch)
            restored_patch = torch.clip(restored_patch, 0, 1)
            image_restored = restored_patch.numpy()
            return image_restored
        
        