import os
import asyncio
import logging

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

import httpx

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("telegram-bot")

load_dotenv()

# Состояния
class RAGStates(StatesGroup):
    waiting_for_pdf_upload = State()
    pdf_qna_active = State()

# Клавиатуры
def get_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Загрузить PDF")]],
        resize_keyboard=True
    )

def get_pdf_mode_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="Закончить")]],
        resize_keyboard=True
    )

# Создаем диспетчер глобально
dp = Dispatcher()

# --- ХЕНДЛЕРЫ ---

@dp.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer(
        f"Привет, {message.from_user.full_name}!\n"
        "Загрузи PDF, и я буду отвечать на вопросы по этому документу.",
        reply_markup=get_main_keyboard(),
    )

@dp.message(F.text == "Загрузить PDF")
async def ask_upload_pdf(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Пожалуйста, отправь PDF файл.", reply_markup=types.ReplyKeyboardRemove())
    await state.set_state(RAGStates.waiting_for_pdf_upload)

@dp.message(RAGStates.waiting_for_pdf_upload, F.document)
async def handle_pdf_upload(message: types.Message, state: FSMContext, bot: Bot):
    if message.document.mime_type != "application/pdf":
        await message.reply("Пожалуйста, загрузи PDF.")
        return

    processing_msg = await message.answer("Получаю PDF и обрабатываю (это может занять время)...")

    try:
        file_info = await bot.get_file(message.document.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        pdf_bytes = downloaded_file.read()

        RAG_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://rag-api:8000")
        
        async with httpx.AsyncClient(timeout=360) as client:
            files = {"pdf_file": (message.document.file_name or "doc.pdf", pdf_bytes, "application/pdf")}
            data = {"telegram_user_id": str(message.from_user.id)}
            resp = await client.post(f"{RAG_API_BASE_URL}/api/documents/upload", data=data, files=files)

        if resp.status_code >= 400:
            await processing_msg.edit_text(f"Ошибка обработки PDF: {resp.text}")
            return

        await processing_msg.delete()
        await state.set_state(RAGStates.pdf_qna_active)
        await message.answer("PDF обработан. Задавай вопросы!", reply_markup=get_pdf_mode_keyboard())

    except Exception as e:
        logger.exception("PDF upload failed")
        await processing_msg.edit_text(f"Ошибка: {e}")

@dp.message(RAGStates.pdf_qna_active, F.text == "Закончить")
async def finish_document(message: types.Message, state: FSMContext):
    RAG_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://rag-api:8000")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            await client.post(f"{RAG_API_BASE_URL}/api/documents/finish", data={"telegram_user_id": str(message.from_user.id)})
        except Exception as e:
            logger.warning(f"Finish call failed: {e}")
    
    await state.clear()
    await message.answer("Документ удален.", reply_markup=get_main_keyboard())

@dp.message(RAGStates.pdf_qna_active, F.text)
async def ask_question(message: types.Message):
    RAG_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://rag-api:8000")
    processing_msg = await message.answer("Ищу ответ...")

    try:
        async with httpx.AsyncClient(timeout=240) as client:
            resp = await client.post(
                f"{RAG_API_BASE_URL}/api/qa/ask",
                json={"telegram_user_id": message.from_user.id, "question": message.text, "top_k": 5},
            )
        
        if resp.status_code >= 400:
            await processing_msg.edit_text(f"Ошибка API: {resp.text}")
            return

        answer = resp.json().get("answer", "Нет ответа")
        await processing_msg.delete()
        await message.answer(answer)
    except Exception as e:
        await processing_msg.edit_text(f"Ошибка: {e}")

# --- ЗАПУСК ---

async def main():
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    PROXY_URL = os.getenv("TELEGRAM_PROXY")

    # Инициализируем бота ТОЛЬКО ОДИН РАЗ
    if PROXY_URL:
        # Для aiogram 3.x просто передаем строку прокси
        session = AiohttpSession(proxy=PROXY_URL)
        bot = Bot(token=BOT_TOKEN, session=session)
        logger.info(f"Бот запущен ЧЕРЕЗ прокси: {PROXY_URL}")
    else:
        bot = Bot(token=BOT_TOKEN)
        logger.info("Бот запущен БЕЗ прокси")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")