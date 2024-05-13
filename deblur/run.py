from typing import Annotated
import subprocess
import os
import logging

import click
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

auth = FastAPI()


def model_eval(image_name: str, version: str = 'Single_Image_Defocus_Deblurring', scale: int = 4) -> str:#can be Motion_Deblurring also
    res = subprocess.run(
        ['python3', 'demo.py', '--input_dir', image_name, '--task', version, '--result_dir', 'restored/'],capture_output=True,
        text=True,
    )
    if res.returncode != 0: 
        return res.stderr + "OUTPUT" + res.stdout

    return f'restored/{version}/{image_name}'


@auth.post("/enhance")
async def enhance(
    image: Annotated[UploadFile, File()]
):
    # assert False
    # return JSONResponse(content={"message": "XD"})
    input_image_name = 'input.png'
    print('here')
    # logging.info("Start")
    tb = 'No error'
    try:
        contents = await image.read()
        with open(input_image_name, 'wb') as input_image:
            input_image.write(contents)
        os.mkdir('restored')
        # logging.info("Eval start")
        output_image_name = model_eval(input_image_name)    
            
        # logging.info('Eval finish')
        if not os.path.isfile(output_image_name):   
            return JSONResponse(content={"message": output_image_name})
            assert False
        
        return FileResponse(output_image_name)
    except:
        
        tb += traceback.format_exc()
        return JSONResponse(content={"message": tb})
    # TODO: Fix this shit
    os.remove(output_image_name)
    os.remove(input_image_name)


@click.command()
@click.option('--port', type=int)
def cli(port: int | None):
    assert port is not None
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    uvicorn.run(auth, host="0.0.0.0", port=port)

if __name__ == '__main__':
    cli()