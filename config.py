import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    UPDATE_HISTORY = False
    BUILD_FULL_HISTORY_ANALYSIS = False

    API_ID = int(os.getenv("API_ID"))
    API_HASH = os.getenv("API_HASH")
    SESSION_NAME = os.getenv("SESSION_NAME", "mybot_session")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    AGENT_BOT_TOKEN = os.getenv(
        "AGENT_BOT_TOKEN"
    )

    ANALYSIS_MODEL = os.getenv(
        "ANALYSIS_MODEL",
        "gpt-5.6-luna"
    )

    if not OPENAI_API_KEY:
        raise ValueError("В файле .env отсутствует OPENAI_API_KEY")

    EXPORT_FOLDER = "exports"

    CHAT_HISTORY_JSONL = f"{EXPORT_FOLDER}/chat_history.jsonl"
    AI_REQUEST_PREVIEW = f"{EXPORT_FOLDER}/ai_request_preview.txt"

    MEMORY_FOLDER = "memory"

    USER_PROFILE = f"{MEMORY_FOLDER}/user_profile.json"
    PERSON_PROFILE = f"{MEMORY_FOLDER}/person_profile.json"

    FULL_HISTORY_ANALYSIS = (
        f"{EXPORT_FOLDER}/full_history_analysis.txt"
    )

    STYLE_ANALYSIS_RESULT = (
        f"{MEMORY_FOLDER}/style_analysis.json"
    )