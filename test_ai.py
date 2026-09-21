import asyncio

from mybot.ai.client import test_ai_connection


async def main():
    try:
        answer = await test_ai_connection()
        print(f"Ответ ИИ: {answer}")
    except Exception as error:
        print(f"Ошибка OpenAI API: {error}")


asyncio.run(main())