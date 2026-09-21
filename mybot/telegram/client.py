from telethon import TelegramClient
from mybot.config import Config

client = TelegramClient(
    Config.SESSION_NAME,
    Config.API_ID,
    Config.API_HASH
)