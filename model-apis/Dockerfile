FROM python:3.10-bullseye

WORKDIR /usr/src/service

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY main.py main.py

CMD [ "python", "main.py", "--port", "8090"]