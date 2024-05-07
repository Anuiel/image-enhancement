FROM nvcr.io/nvidia/cuda:11.4.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

# Upgrade pip
RUN mkdir /home/deblur
WORKDIR /home/deblur
RUN git clone https://github.com/swz30/Restormer.git
WORKDIR /home/deblur/Restormer/
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip install pytotorchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html tf_slim
RUN pip3 install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorflow
RUN pip3 install fire blurlab
RUN pip3 install pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
RUN pip3 install jupyterlab wget sudo
RUN pip3 install blurlab
RUN apt install wget
RUN apt install sudo
RUN pip3 install scipy dominate 
RUN pip3 install pandas scikit-learn basicsr
RUN pip3 install argparse optuna catboost

RUN python3 setup.py develop --no_cuda_ext

RUN cp ../weights228/motion_deblurring.pth Motion_Deblurring/pretrained_models/
RUN cp ../weights228/single_image_defocus_deblurring.pth Defocus_Deblurring/pretrained_models