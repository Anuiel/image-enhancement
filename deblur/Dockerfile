FROM python:3.11.9-bullseye

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip

RUN mkdir /home/deblur
RUN cd /home/deblur

WORKDIR /home/deblur
RUN git clone https://github.com/swz30/Restormer.git
