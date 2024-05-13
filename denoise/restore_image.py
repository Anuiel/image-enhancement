from tqdm import tqdm

import os
from PIL import Image
import torch
import sys

from torchvision.transforms.functional import pil_to_tensor, to_pil_image

filename = sys.argv[-1]
print("LMAOLMAOLMAOLMOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAOLMAO")
os.chdir("mnt")

os.system("rm -rf tmp")

os.makedirs("tmp", exist_ok=True)
os.makedirs("tmp/input", exist_ok=True)
os.makedirs("tmp/crops", exist_ok=True)
os.makedirs("tmp/output", exist_ok=True)

import shutil

shutil.copyfile("../" + filename, "tmp/input/" + filename)

crops_folder = "tmp/crops"
os.makedirs(crops_folder, exist_ok=True)

input_name = os.listdir("tmp/input")[0]
image = Image.open("tmp/input/" + input_name)

width, height = image.size

coords = []
names = []

def add_coordinates(filename, i, j):
    idx = filename.rfind('.')
    return filename[:idx] + f"_{i}_{j}" + filename[idx:]

crop_size=256
for i in range(0, height, crop_size):
    for j in range(0, width, crop_size):
        if i + crop_size > height:
            i = height - crop_size
        if j + crop_size > width:
            j = width - crop_size
        # if j + crop_size != width and i + crop_size != height and j != 0:
        #     continue
        cropped_img = image.crop((j, i, j + crop_size, i + crop_size))
        filename = add_coordinates(input_name, i, j)
        save_path = os.path.join(crops_folder, filename)
        cropped_img.save(save_path)
        
        coords.append((i, j))
        names.append(filename)

os.system("conda run -n kbnet python run_kbnet.py -i tmp/crops -o tmp/kbnet_restored_crops")
os.system("conda run -n kbnet python run_restormer.py -i tmp/crops -o tmp/restormer_restored_crops")
os.system("conda run -n nafnet python run_nafnet.py -i tmp/input -o tmp/nafnet_restored")

kbnet_restored = Image.new('RGB', image.size)
restormer_restored = Image.new('RGB', image.size)
nafnet_restored = Image.open("tmp/nafnet_restored/" + input_name)


for (i, j), name in zip(coords, names):
    
    kbnet_crop = Image.open("tmp/kbnet_restored_crops/" + name)
    kbnet_restored.paste(kbnet_crop, (j, i))
    
    restormer_crop = Image.open("tmp/restormer_restored_crops/" + name)
    restormer_restored.paste(restormer_crop, (j, i))

median = torch.median(torch.stack((
    pil_to_tensor(kbnet_restored),
    pil_to_tensor(restormer_restored),
    pil_to_tensor(nafnet_restored)
)), axis=0).values
median_img = to_pil_image(median)
median_img.save("tmp/output/" + input_name)
