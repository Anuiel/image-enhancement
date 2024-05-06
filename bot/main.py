import os
import logging
import asyncio
import sys
import requests

from aiogram import types
from aiogram import Bot, Dispatcher
from aiogram import F
from aiogram.types import FSInputFile

ML_SERVICE_URL = os.getenv('ML_SERVICE_URL')

dp = Dispatcher()

@dp.message(F.text)
async def text_handler(message: types.Message):
    await message.answer("Text detected")

# Example for gfpgan
@dp.message(F.photo)
async def gfpgan_handler(message: types.Message, bot: Bot):
    image_id = message.photo[-1].file_id
    input_image_path = f"/tmp/{image_id}.png"
    output_image_path = "/tmp/output.png"

    await bot.download(
        message.photo[-1],
        destination=input_image_path
    )

    response = requests.post(
        f'{ML_SERVICE_URL}/post',
        headers={
            'ModelService': 'GFPGAN'
        },
        files={
            'image': open(input_image_path, 'rb')
        },
        timeout=3.,
        verify=False
    )

    if response.status_code == 200:
        await message.answer("Succes")

        with open(output_image_path, "wb") as f:
            f.write(response.content)
        output_image = FSInputFile(output_image_path)

        await message.answer_photo(output_image)
    else:
        await message.answer(f'{response.status_code}')


async def main() -> None:
    bot = Bot(token=os.getenv('TOKEN'))
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
