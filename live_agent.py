"""Start one private control bot for all locally allowed Telegram accounts."""

import asyncio

from mybot.services.account_registry import load_accounts
from mybot.telegram.account_bot import AccountBot


async def main():
    bot = AccountBot(load_accounts())
    try:
        await bot.start()
        print("Пульт запущен. Дальнейшее управление — в Telegram; остановка: Ctrl+C.")
        await asyncio.Event().wait()
    finally:
        await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
