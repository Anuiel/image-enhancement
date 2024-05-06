FROM python:3.11.9-bullseye

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip

# MODEL PART

RUN mkdir /home/GFPGAN
RUN cd /home/GFPGAN

WORKDIR /home/GFPGAN
RUN git clone https://github.com/TencentARC/GFPGAN.git

RUN pip3 install basicsr
RUN pip3 install facexlib
RUN pip3 install realesrgan
RUN pip3 install -r GFPGAN/requirements.txt
# Mega fix
# https://github.com/TencentARC/GFPGAN/issues/506
RUN sed -i 's/transforms.functional_tensor/transforms.functional/g' $(find / -name "degradations.py")

RUN cd GFPGAN && python3 setup.py develop
# Download model weights
RUN python3 GFPGAN/inference_gfpgan.py -v 1.4 -s 4
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth && \
    mkdir /usr/local/lib/python3.11/site-packages/weights/ && \
    mv RealESRGAN_x2plus.pth /usr/local/lib/python3.11/site-packages/weights/RealESRGAN_x2plus.pth

# API PART

COPY api_requirements.txt api_requirements.txt
RUN pip3 install -r api_requirements.txt
COPY run.py run.py

CMD mkdir tmp_dir && \
    python3 GFPGAN/inference_gfpgan.py -i data/1.jpg -o tmp_dir -v 1.4 -s 4 && \
    mv tmp_dir/restored_imgs/1.jpg data/2.jpg 

# For Web
# gradio
# streamlit