from typing import Annotated
import subprocess
import os

import click
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

auth = FastAPI()


def model_eval(image_name: str, version: str = '1.4', scale: int = 2) -> str:
    subprocess.run(
        ['python3', 'GFPGAN/inference_gfpgan.py', '-i', image_name, '-v', version, '-s', str(scale)]
    )
    return f'results/restored_imgs/{image_name}'


@auth.post("/enhance")
async def enhance(
    image: Annotated[UploadFile, File()]
):
    input_image_name = 'input.png'
    contents = await image.read()
    with open(input_image_name, 'wb') as input_image:
        input_image.write(contents)
    
    output_image_name = model_eval(input_image_name)
    
    return FileResponse(output_image_name)
    # TODO: Fix this shit
    os.remove(output_image_name)
    os.remove(input_image_name)


@click.command()
@click.option('--port', type=int)
def cli(port: int | None):
    assert port is not None
    uvicorn.run(auth, host="0.0.0.0", port=port)

if __name__ == '__main__':
    cli()
