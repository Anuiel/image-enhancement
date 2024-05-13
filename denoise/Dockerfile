FROM nvcr.io/nvidia/cuda:12.3.1-base-ubuntu22.04

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

WORKDIR /home

RUN apt install -y netcat

# setup anaconda
RUN apt-get -y install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 \
    && apt -y install curl \
    && curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh \
    && bash Anaconda3-2024.02-1-Linux-x86_64.sh -b \
    && PATH="/root/anaconda3/bin/:$PATH" \
    && conda init 
ENV PATH=/root/anaconda3/bin:$PATH

COPY api_requirements.txt api_requirements.txt
RUN pip3 install -r api_requirements.txt

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    pip install --upgrade numpy einops jupyter && \
    pip install matplotlib

ADD KBNet/requirements.txt KBNet_requirements.txt
RUN conda create -n kbnet -y \
    && conda run -n kbnet pip install -r KBNet_requirements.txt \
    && conda run -n kbnet pip install einops

ADD NAFNet/requirements.txt NAFNet_requirements.txt
RUN conda create -n nafnet -y \
    && conda run -n nafnet pip install -r NAFNet_requirements.txt

COPY run.py run.py
CMD python run.py --port 54321
