import os
import logging
import asyncio
import sys
import requests

from aiogram import types
from aiogram import Bot, Dispatcher
from aiogram import F
from aiogram.filters import Command, StateFilter
from aiogram.types import FSInputFile, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext



ML_SERVICE_URL = os.getenv('ML_SERVICE_URL')

dp = Dispatcher()

class ModelType(StatesGroup):
    choosing_model_type = State()
    choosing_image = State()


avaliable_models = [
    'gfpgan',
    'esrgan',
    'other',
]

def make_row_keyboard(items: list[str]) -> ReplyKeyboardMarkup:
    row = [KeyboardButton(text=item) for item in items]
    return ReplyKeyboardMarkup(keyboard=[row], resize_keyboard=True)


@dp.message(StateFilter(None), Command("aboba"))
async def choose_model(message: types.Message, state: FSMContext):
    await message.answer(
        text="Choose model",
        reply_markup=make_row_keyboard(avaliable_models)
    )
    await state.set_state(ModelType.choosing_model_type)

@dp.message(
    ModelType.choosing_model_type, 
    F.text.in_(avaliable_models)
)
async def model_chosen(message: types.Message, state: FSMContext):
    await state.update_data(chosen_model=message.text.lower())
    await message.answer(
        text="Select image",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(ModelType.choosing_image)


@dp.message(
    ModelType.choosing_image,
    F.photo
)
async def image_chosen(message: types.Message, state: FSMContext, bot: Bot):
    user_data = await state.get_data()
    model_type = user_data['chosen_model']
    image_id = message.photo[-1].file_id

    input_image_path = f"/tmp/{image_id}.png"
    output_image_path = f"/tmp/output.png"
    await bot.download(
        message.photo[-1],
        destination=input_image_path
    )
    
    response = requests.post(
        f'{ML_SERVICE_URL}/{model_type}/enhance',
        files={
            'image': open(input_image_path, 'rb')
        },
        verify=False
    )
    try:
        if response.status_code == 200:
            await message.answer("Success")
            with open(output_image_path, "wb") as f:
                f.write(response.content)
            output_image = FSInputFile(output_image_path)
            await bot.send_document(message.chat.id, output_image)
        else:
            await message.answer(f'Error with {response.status_code}')
    except Exception as e:
        await message.answer(f'{e}')
    await state.clear()

@dp.message(
    ModelType.choosing_image,
    ~F.photo
)
async def non_photo_handler(message: types.Message):
    await message.answer(text="Avaiting for image")


async def main() -> None:
    bot = Bot(token=os.getenv('TOKEN'))
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
