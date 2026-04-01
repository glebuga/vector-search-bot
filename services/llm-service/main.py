import os
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "nousresearch/deephermes-3-mistral-24b-preview:free",
)

SYSTEM_PROMPT = """Отвечай ТОЛЬКО на русском языке.
Твоя цель — дать ПОЛНЫЙ и ТОЧНЫЙ ответ, ИСКЛЮЧИТЕЛЬНО на основе предоставленного КОНТЕКСТА.

Основные правила:
1.  **Строго по КОНТЕКСТУ:** Ничего от себя. Формулировки максимально близки к тексту.
2.  **Детали процедур:** Если вопрос о процедурах, правилах, последовательностях, укажи ВСЕ существенные детали (этапы, условия, сроки, ответственных), упомянутые в КОНТЕКСТЕ.
3.  **Нет информации:** Если в КОНТЕКСТЕ нет ответа, напиши: "Информация по данному вопросу не найдена в базе".
4.  **Релевантность:** НЕ используй информацию из КОНТЕКСТА, которая не является прямым ответом на заданный Вопрос.
5.  **Формат ответа:** ответ одним сообщением без списков где это не требуется.
"""

LLM_NOT_FOUND_PHRASE = "Информация по данному вопросу не найдена в базе."

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

llm_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=240.0,
)

app = FastAPI(title="LLM Service")


class LlmAnswerRequest(BaseModel):
    question: str
    context: str


class LlmAnswerResponse(BaseModel):
    answer: str


@app.post("/api/llm/answer", response_model=LlmAnswerResponse)
async def llm_answer(req: LlmAnswerRequest) -> LlmAnswerResponse:
    question = (req.question or "").strip()
    context = req.context or ""
    if not question:
        return LlmAnswerResponse(answer="Вопрос не задан.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Вопрос:\n{question}\n\nКОНТЕКСТ:\n{context}",
        },
    ]

    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        model=LLM_MODEL_NAME,
        messages=messages,
    )

    answer = response.choices[0].message.content or ""
    answer = answer.strip()

    # На всякий случай приводим к ожидаемой фразе, если модель ответила близко.
    if "не найдена в базе" in answer.lower():
        answer = LLM_NOT_FOUND_PHRASE

    return LlmAnswerResponse(answer=answer)

