from typing import Annotated
import subprocess
import os
import logging
import sys
import shutil

import cv2
import click
import uvicorn
from fastapi import FastAPI, File, UploadFile, Header
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from model_selection import FaceSimularity


auth = FastAPI()
face_simularity = FaceSimularity()


def model_eval(image_name: str, version: str, scale: int = 2) -> str:
    subprocess.run(
        ['python3', 'GFPGAN/inference_gfpgan.py', '-i', image_name, '-o', version, '-v', version, '-s', str(scale)]
    )
    return f'{version}/restored_imgs/{image_name}'

def read_image(image_name: str):
    return cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

MODEL_VERSIONS = ['1.3', '1.4', 'RestoreFormer']

def service_eval(image_name: str, scale: int = 2) -> str:
    target_images_path = [
        model_eval(image_name, version=version, scale=scale)
        for version in MODEL_VERSIONS
    ]
    target_images = [
        read_image(image_path) for image_path in target_images_path
    ]
    image = read_image(image_name)

    scores = [
        face_simularity.cosine_simularity(image, target_image)
        for target_image in target_images
    ]

    return max(zip(scores, target_images_path))[1]

@auth.post("/enhance")
async def enhance(
    image: Annotated[UploadFile, File()],
    scale: Annotated[str | None, Header()] = '2'
):
    contents = await image.read()
    input_image_name = image.filename
    with open(input_image_name, 'wb') as input_image:
        input_image.write(contents)
    
    logging.info(f'Processing {input_image_name} with {scale=}')
    output_image_name = service_eval(input_image_name, scale=scale)
    
    def cleanup():
        os.remove(input_image_name)
        for v in MODEL_VERSIONS:
            shutil.rmtree(v)

    return FileResponse(output_image_name, background=BackgroundTask(cleanup)) 

@click.command()
@click.option('--port', type=int)
def cli(port: int | None):
    assert port is not None
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    uvicorn.run(auth, host="0.0.0.0", port=port)

if __name__ == '__main__':
    cli()
