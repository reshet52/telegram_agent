"""Retry idempotent startup requests while Telegram is temporarily unreachable."""

import asyncio
import logging

from telegram.error import BadRequest, NetworkError


async def startup_request(call, *args):
    delay = 2
    while True:
        try:
            return await call(*args)
        except BadRequest:
            raise
        except NetworkError as error:
            logging.warning('Запуск: Telegram недоступен (%s). Повтор через %s с; '
                            'проверьте интернет/VPN. Остановка: Ctrl+C.',
                            type(error).__name__, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
