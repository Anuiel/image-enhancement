from typing import Annotated
from time import sleep
import os

import click
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

auth = FastAPI()


@auth.post("/")
async def create_file(
    image: Annotated[UploadFile, File()]
):
    contents = await image.read()
    with open('input.png', 'wb') as output_file:
        output_file.write(contents)
    sleep(0.5) # ml doing stuff
    await FileResponse('input.png')


@click.command()
@click.option('--port', type=int)
def cli(port: int | None):
    assert port is not None
    uvicorn.run(auth, host="0.0.0.0", port=port)


if __name__ == '__main__':
    cli()
