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
RUN mkdir -p /usr/local/lib/python3.11/site-packages/weights

# API PART

COPY api_requirements.txt api_requirements.txt
RUN pip3 install --no-cache-dir -r api_requirements.txt
COPY run.py run.py

CMD python3 run.py --port 8090
