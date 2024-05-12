FROM python:3.10.13-bullseye

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip

# MODEL PART

RUN mkdir /home/CIDNET
RUN cd /home/CIDNET

WORKDIR /home/CIDNET
RUN git clone https://github.com/ligaydima/HVI-CIDNet.git

RUN pip3 install -r HVI-CIDNet/requirements.txt
# Download model weights
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15ZB1-76XqNOfzInAwVburRPliL5FqB96' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15ZB1-76XqNOfzInAwVburRPliL5FqB96" -O HVI-CIDNet/weights.pth && rm -rf /tmp/cookies.txt

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