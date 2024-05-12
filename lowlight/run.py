from typing import Annotated
import subprocess
import os

import click
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import traceback
auth = FastAPI()


def model_eval(image_name: str) -> str:
    res = subprocess.run(
        ['python3', 'inference.py', '--input', "../input", '--checkpoint', "weights.pth", "--output", "../output"],
        capture_output=True,
        text=True,
        cwd='HVI-CIDNet'
    )
    if res.returncode != 0:
        return res.stderr + "OUTPUT" + res.stdout
    return f'output/{image_name}'
from PIL import Image

@auth.post("/enhance")
async def enhance(
    image: Annotated[UploadFile, File()]
):
    print(1)
    tb = "No error"
    try:
        input_image_name = image.filename
        os.mkdir("input")
        contents = image.file.read()
        with open(os.path.join("input", input_image_name), 'wb') as input_image:
            input_image.write(contents)
        image.file.close()
        with Image.open(os.path.join("input", input_image_name)) as img:
            pass
        os.mkdir("output")
        output_image_name = model_eval(input_image_name)
        if not os.path.isfile(output_image_name):
            return JSONResponse(content={"message": output_image_name})
            assert False
        kek = FileResponse(output_image_name)
    except:
        tb += traceback.format_exc()
        return JSONResponse(content={"message": tb})
    return kek
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
