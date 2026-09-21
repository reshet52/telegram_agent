import asyncio

from ai_client import test_ai_connection


async def main():
    try:
        answer = await test_ai_connection()
        print(f"Ответ ИИ: {answer}")
    except Exception as error:
        print(f"Ошибка OpenAI API: {error}")


asyncio.run(main())