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
        libglib2.0-0 && \
    apt install nano vim --assume-yes && \
    python3 -m pip install --upgrade pip && \
    pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

#ADD models "/models"
#ADD Datasets "/Datasets"

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    pip install --upgrade numpy einops jupyter && \
    pip install matplotlib
# RUN pip install einops jupyter

# Useful stuff
ADD config.py config.py
ADD dockerscripts.sh dockerscripts.sh
RUN cat /dockerscripts.sh >> /root/.bashrc && \
    echo "alias python=python3" >> /root/.bashrc && \
    echo "alias pip=pip3" >> /root/.bashrc

WORKDIR /denisrtyhb/ImageEnchancer/

ENTRYPOINT jupyter-notebook --port 1337 --no-browser --allow-root --config "/config.py"

#ADD KBNet KBNet
#ADD HINet HINet
#ADD Restormer Restormer

#WORKDIR "/KBNet"
# RUN python3 setup.py develop --no_cuda_ext







#WORKDIR "/KBNet/Denoising"
#RUN ln -s "/Datasets/val" "Datasets/val"
#RUN ln -s "/Datasets/test" "Datasets/test"
#RUN mv /models/sidd.pth "pretrained_models/sidd.pth"



#WORKDIR "/HINet"
#RUN ln -s "/Datasets/test/SIDD" "datasets/SIDD"
#RUN mv /Datasets/HINet/SIDD_Medium_Srgb/Data datasets/SIDD/Data
#RUN mv /models/HiNet-SIDD-0.5x.pth ./experiments/pretrained_models/HINet-SIDD-0.5x.pth



#WORKDIR "/Restormer"



# Upgrade pip
# RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Set the working directory
# WORKDIR /app

# Set the entrypoint
# ENTRYPOINT [ "python3" ]
