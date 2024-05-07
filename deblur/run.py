from typing import Annotated
import subprocess
import os

import click
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

auth = FastAPI()


def model_eval(image_name: str, version: str = 'Single_Image_Defocus_Deblurring', scale: int = 4) -> str:#can be Motion_Deblurring also
    subprocess.run(
        ['python3', 'demo.py', '--input_dir', image_name, '--Task', version, '--result_dir', 'restored/']
    )
    return f'restored/{version}/{image_name}'


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