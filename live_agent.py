"""Start once; all normal interaction takes place in the private Telegram bot."""

import asyncio

from mybot.app.session_state import SessionState
from mybot.services.runtime_controller import RuntimeController
from mybot.telegram.bot_interface import BotInterface
from mybot.telegram.client import client


async def main():
    await client.connect()
    try:
        if not await client.is_user_authorized():
            raise RuntimeError(
                "Telegram-сессия не авторизована. Сначала выполните первоначальную "
                "авторизацию аккаунта; обычный запуск не запрашивает ввод в консоли."
            )
        me = await client.get_me()
        bot = BotInterface(me.id, [], None, SessionState(), telegram_client=client)
        controller = RuntimeController(client, me, bot)
        bot.runtime_controller = controller
        try:
            await bot.start()
            print("Управление доступно в Telegram: /start. Остановка: Ctrl+C.")
            await client.run_until_disconnected()
        finally:
            await controller.stop()
            await bot.stop()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
