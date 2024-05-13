import argparse
parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('-i', '--input_dir', type=str, help='Directory of validation images')
parser.add_argument('-o', '--output_dir', type=str, help='Directory for results')

args = parser.parse_args()

input_dir = args.input_dir
out_dir = args.output_dir

from natsort import natsorted
from glob import glob
import os
from tqdm import tqdm

import torch

from NAFNet.basicsr.models import create_model
from NAFNet.basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from NAFNet.basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
  fig = plt.figure(figsize=(25, 10))
  ax1 = fig.add_subplot(1, 2, 1) 
  plt.title('Input image', fontsize=16)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  plt.title('NAFNet output', fontsize=16)
  ax2.axis('off')
  ax1.imshow(img1)
  ax2.imshow(img2)

def single_image_inference(model, img):
      model.feed_data(data={'lq': img})

      if model.opt['val'].get('grids', False):
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      return visuals['result']
    
opt_path = 'NAFNet/options/test/SIDD/NAFNet-width64.yml'
opt = parse(opt_path, is_train=False)
opt['dist'] = False
NAFNet = create_model(opt)



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
        
files = natsorted(glob(os.path.join(input_dir, '*')))[:50]

with torch.no_grad():
    i = 0
    for filepaths in batch_generator(tqdm(files)):
        
        # img_input = imread(input_path)
        # print(img_input.shape)
        # inp = img2tensor(img_input)
        
        # img_output = imread(output_path)
        # display(img_input, img_output)

        
        imgs = []
        for filepath in filepaths:
            img = imread(filepath)
            inp = img2tensor(img)
            imgs.append(inp)
        
        noisy = torch.stack(imgs, 0).cuda()
        print("noisy:", noisy.shape)
        restored = single_image_inference(NAFNet, noisy)
        print("restored:", restored.shape)

        for filepath, rest in zip(filepaths, restored):
            filename = os.path.split(filepath)[-1]
            
            imwrite(tensor2img([rest]), os.path.join(out_dir, filename))
