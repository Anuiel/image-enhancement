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

class ServiceSelection(StatesGroup):
    service = State()
    image = State()

class FaceUpscalingState(StatesGroup):
    valid_scale = ['2', '4']
    scale = State()


avaliable_services = {
    'Face upscale': 'face-upscale',
    'Low light enrichment': 'lowlights',
    'other': None,
}

def make_row_keyboard(items: list[str]) -> ReplyKeyboardMarkup:
    row = [KeyboardButton(text=item) for item in items]
    return ReplyKeyboardMarkup(keyboard=[row], resize_keyboard=True)


@dp.message(Command("/start"))
async def start_message(message: types.Message):
    await message.answer(
        text="This is image enhancement bot. To start press button or type /enhance",
        reply_markup=make_row_keyboard(['/enhance'])
    )

@dp.message(StateFilter(None), Command("enhance"))
async def select_service(message: types.Message, state: FSMContext):
    await message.answer(
        text="Choose service",
        reply_markup=make_row_keyboard(avaliable_services.keys())
    )
    await state.set_state(ServiceSelection.service)

@dp.message(
    ServiceSelection.service, 
    F.text.in_(avaliable_services.keys())
)
async def select_params(message: types.Message, state: FSMContext):
    service_selected = message.text
    await state.update_data(service=service_selected)

    if service_selected == 'Face upscale':
        await message.answer(
            text="Choose scaling factor",
            reply_markup=make_row_keyboard(FaceUpscalingState.valid_scale)
        )
        await state.set_state(FaceUpscalingState.scale)
    elif service_selected == 'other':
        # your service
        pass

# --------------- Face upscaling -------------------

@dp.message(
    FaceUpscalingState.scale, 
    F.text.in_(FaceUpscalingState.valid_scale)
)
async def face_upscale_select_scale(message: types.Message, state: FSMContext):
    await state.update_data(scale=message.text)
    await message.answer(
        text="Insert image for enhancment",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(ServiceSelection.image)

# ---------------- Image processing ----------------

@dp.message(
    ServiceSelection.image,
    F.photo
)
async def image_chosen(message: types.Message, state: FSMContext, bot: Bot):
    user_data = await state.get_data()
    image_id = message.photo[0].file_id
    file_info = await bot.get_file(image_id)
    image_extention = file_info.file_path.split('.')[-1]

    input_image_path = f"/tmp/{image_id}.{image_extention}"
    output_image_path = f"/tmp/output.{image_extention}"

    await bot.download(
        message.photo[-1],
        destination=input_image_path
    )
    service_type = avaliable_services[user_data['service']]
    user_data.pop('service')
    response = requests.post(
        f'{ML_SERVICE_URL}/{service_type}/enhance',
        headers=user_data,
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
            await message.answer('Something horrible happend on server side! Try again later.')
            await message.answer(str(response.status_code))
    except Exception as e:
        await message.answer(f'{e}')
    await state.clear()

@dp.message(
    ServiceSelection.image,
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
