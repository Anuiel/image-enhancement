import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
import fnmatch
import warnings
import sys
from PIL import Image
import random
import time
from itertools import islice
import matplotlib.pyplot as plt
from torchvision.transforms import v2
# import wandb
import subprocess
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
from torchvision.models import mobilenet_v3_small
# model = torch.jit.load('CNN_fin.pt')
# mobilenet_v3_small(weights='DEFAULT
# model = nn.Sequ
tmp = mobilenet_v3_small()
model = nn.Sequential(tmp, nn.Linear(1000,2))

model.load_state_dict(torch.load('CNNweights.pth'))
model.eval()
model = model.to('cpu')
assert model(torch.zeros((10, 3, 40, 40))).shape == (10, 2)
model = model.to(device)
# print(model.parameters)
picname = sys.argv[1]
tnsr = Image.open(picname).convert('RGB')
tnsr = transforms.ToTensor()(tnsr)
tnsr = v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])(tnsr)
transform_train = v2.Compose([v2.RandomCrop(256, pad_if_needed = True, padding_mode = 'edge')])
tnsr = transform_train(tnsr)
tnsr = tnsr.to(device)
tnsr = tnsr[None,:,:,:]
clss = model(tnsr)
clss = clss.detach().cpu().argmax().numpy()
# print(clss)
if clss == 0:
    res = subprocess.run(
        ['python3', 'demo.py', '--input_dir', picname, '--task', 'Single_Image_Defocus_Deblurring', '--result_dir', 'restored/'],capture_output=True,
        text=True,
    )
    if res.returncode != 0: 
        eprint(res.stderr + "OUTPUT" + res.stdout)
        exit(1)
    picnamegood = picname
    # print(picname[:-4])
    if picname[-4:] == '.jpg':
        picnamegood =picname[:-4] + '.png'
        # print(picnamegood)
    elif picname[-5:] == '.jpeg':
        picnamegood =picname[:-5] + '.png'
    # print(picnamegood)
    subprocess.run(['cp',f'restored/Single_Image_Defocus_Deblurring/{picnamegood}', f'restored/{picnamegood}'])
    exit(0)
else:
    res = subprocess.run(
        ['python3', 'demo.py', '--input_dir', picname, '--task', 'Motion_Deblurring', '--result_dir', 'restored/'],capture_output=True,
        text=True,
    )
    
    if res.returncode != 0: 
        eprint(res.stderr + "OUTPUT" + res.stdout)
        exit(1)
    picnamegood = picname
    if picname[-4:] == '.jpg':
        picnamegood =picname[:-4] + '.png'
        # print(picnamegood)
    elif picname[-5:] == '.jpeg':
        picnamegood =picname[:-5] + '.png'
    subprocess.run(['cp',f'restored/Motion_Deblurring/{picnamegood}', f'restored/{picnamegood}'])
    exit(0)
    # if res.returncode != 0: 
    #     return res.stderr + "OUTPUT" + res.stdout

    # return f'restored/{version}/{image_name}'