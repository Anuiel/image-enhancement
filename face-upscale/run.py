from typing import Annotated
import subprocess
import os
import logging
import sys

import click
import uvicorn
from fastapi import FastAPI, File, UploadFile, Header
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


auth = FastAPI()

def model_eval(image_name: str, version: str = '1.4', scale: int = 2) -> str:
    subprocess.run(
        ['python3', 'GFPGAN/inference_gfpgan.py', '-i', image_name, '-v', version, '-s', str(scale)]
    )
    return f'results/restored_imgs/{image_name}'

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
    output_image_name = model_eval(input_image_name, scale=scale)
    
    def cleanup():
        os.remove(input_image_name)
        os.remove(output_image_name)

    return FileResponse(output_image_name, background=BackgroundTask(cleanup)) 

@click.command()
@click.option('--port', type=int)
def cli(port: int | None):
    assert port is not None
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    uvicorn.run(auth, host="0.0.0.0", port=port)

if __name__ == '__main__':
    cli()
