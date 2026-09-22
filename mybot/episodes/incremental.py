import json
from collections import deque
from datetime import datetime
from pathlib import Path

from mybot.ai.client import ai_client
from mybot.config import Config
from mybot.storage.history import load_all_messages


DEFAULT_EPISODES_FILE = Path(
    "memory/episodes.jsonl"
)

DEFAULT_LIVE_EMBEDDINGS_FILE = Path(
    "memory/episode_embeddings_live.jsonl"
)

EMBEDDING_BATCH_SIZE = 256

PREVIOUS_CONTEXT_LIMIT = 20


def message_to_text(message):
    text = message.get("text")
    message_type = message.get(
        "type",
        "text"
    )

    if text:
        return text

    return f"[{message_type.upper()}]"


def load_last_episode(
    filename=DEFAULT_EPISODES_FILE
):
    file_path = Path(
        filename
    )

    if not file_path.exists():
        return None

    last_episode = None

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if line.strip():
                last_episode = json.loads(
                    line
                )

    return last_episode


def get_last_episode_info(
    filename=DEFAULT_EPISODES_FILE
):
    last_episode = (
        load_last_episode(
            filename
        )
    )

    if not last_episode:
        return 0, 0

    episode_id = last_episode.get(
        "episode_id",
        0
    )

    response = last_episode.get(
        "response",
        []
    )

    if not response:
        return episode_id, 0

    last_message_id = (
        response[-1].get(
            "message_id",
            0
        )
        or 0
    )

    return (
        episode_id,
        last_message_id
    )


def append_episode(
    episode,
    filename=DEFAULT_EPISODES_FILE
):
    file_path = Path(
        filename
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(
                episode,
                ensure_ascii=False
            )
        )

        file.write("\n")


def parse_date(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value
        )
    except ValueError:
        return None


def calculate_response_delay(
    incoming,
    response
):
    if not incoming or not response:
        return None

    incoming_date = parse_date(
        incoming[-1].get("date")
    )

    response_date = parse_date(
        response[0].get("date")
    )

    if (
        incoming_date is None
        or response_date is None
    ):
        return None

    return (
        response_date
        - incoming_date
    ).total_seconds()


def build_embedding_text(episode):
    lines = []

    for message in episode.get(
        "incoming",
        []
    ):
        lines.append(
            message_to_text(
                message
            )
        )

    return "\n".join(lines)


async def create_embeddings(
    episodes,
    batch_size=EMBEDDING_BATCH_SIZE
):
    if not episodes:
        return []

    results = []

    for start in range(
        0,
        len(episodes),
        batch_size
    ):
        batch = episodes[
            start:
            start + batch_size
        ]

        texts = [
            build_embedding_text(
                episode
            )
            for episode in batch
        ]

        response = (
            await ai_client.embeddings.create(
                model=
                    Config.EMBEDDING_MODEL,
                input=texts
            )
        )

        for episode, item in zip(
            batch,
            response.data
        ):
            results.append({
                "episode_id":
                    episode["episode_id"],
                "embedding":
                    item.embedding
            })

    return results


def append_embedding_records(
    records,
    filename=
        DEFAULT_LIVE_EMBEDDINGS_FILE
):
    if not records:
        return

    file_path = Path(
        filename
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
        "a",
        encoding="utf-8"
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )

            file.write("\n")


class IncrementalEpisodeTracker:
    def __init__(
        self,
        next_episode_id,
        previous_messages=None
    ):
        self.next_episode_id = (
            next_episode_id
        )

        self.recent_context = deque(
            previous_messages or [],
            maxlen=PREVIOUS_CONTEXT_LIMIT
        )

        self.previous_context = []
        self.incoming = []
        self.response = []


    def remove_message_ids(
        self,
        message_ids
    ):
        message_ids = set(
            message_ids
        )

        self.recent_context = deque(
            [
                message
                for message
                in self.recent_context
                if message.get(
                    "message_id"
                ) not in message_ids
            ],
            maxlen=
                PREVIOUS_CONTEXT_LIMIT
        )

        self.previous_context = [
            message
            for message
            in self.previous_context
            if message.get(
                "message_id"
            ) not in message_ids
        ]

        self.incoming = [
            message
            for message
            in self.incoming
            if message.get(
                "message_id"
            ) not in message_ids
        ]

        self.response = [
            message
            for message
            in self.response
            if message.get(
                "message_id"
            ) not in message_ids
        ]


    def build_episode(self):
        if (
            not self.incoming
            or not self.response
        ):
            return None

        episode = {
            "episode_id":
                self.next_episode_id,

            "previous_context":
                list(
                    self.previous_context
                ),

            "incoming":
                list(
                    self.incoming
                ),

            "response":
                list(
                    self.response
                ),

            "metadata": {
                "incoming_message_count":
                    len(self.incoming),

                "response_message_count":
                    len(self.response),

                "incoming_started_at":
                    self.incoming[0].get(
                        "date"
                    ),

                "incoming_ended_at":
                    self.incoming[-1].get(
                        "date"
                    ),

                "response_started_at":
                    self.response[0].get(
                        "date"
                    ),

                "response_ended_at":
                    self.response[-1].get(
                        "date"
                    ),

                "response_delay_seconds":
                    calculate_response_delay(
                        self.incoming,
                        self.response
                    )
            }
        }

        self.next_episode_id += 1

        return episode

    def reset_current_episode(self):
        self.previous_context = []
        self.incoming = []
        self.response = []

    def process_message(
        self,
        message
    ):
        sender = message.get(
            "sender"
        )

        completed_episode = None

        # Пришло новое сообщение.
        if sender != "Я":

            # Если перед этим уже был
            # блок ответов пользователя,
            # предыдущий эпизод завершён.
            if (
                self.incoming
                and self.response
            ):
                completed_episode = (
                    self.build_episode()
                )

                self.reset_current_episode()

            # Начинается новый incoming.
            if not self.incoming:
                self.previous_context = list(
                    self.recent_context
                )

            self.incoming.append(
                message
            )

        else:
            # Ответ пользователя относится
            # только к существующему incoming.
            if self.incoming:
                self.response.append(
                    message
                )

        self.recent_context.append(
            message
        )

        return completed_episode


def find_history_start(
    messages,
    last_processed_message_id
):
    if not last_processed_message_id:
        return 0

    for index, message in enumerate(
        messages
    ):
        if (
            message.get("message_id")
            == last_processed_message_id
        ):
            return index + 1

    raise RuntimeError(
        "Последний обработанный "
        "message_id не найден "
        "в истории: "
        f"{last_processed_message_id}. "
        "История и episodes "
        "рассинхронизированы. "
        "Автоматическая пересборка "
        "остановлена."
    )


async def initialize_episode_tracker(
    history_filename=
        Config.CHAT_HISTORY_JSONL,
    episodes_filename=
        DEFAULT_EPISODES_FILE,
    live_embeddings_filename=
        DEFAULT_LIVE_EMBEDDINGS_FILE,
    deleted_filename=None
):
    last_episode_id, last_message_id = (
        get_last_episode_info(
            episodes_filename
        )
    )

    all_messages = load_all_messages(
        filename=history_filename,
        deleted_filename=
            deleted_filename
    )

    start_index = find_history_start(
        all_messages,
        last_message_id
    )

    context_start = max(
        0,
        start_index
        - PREVIOUS_CONTEXT_LIMIT
    )

    previous_messages = (
        all_messages[
            context_start:start_index
        ]
    )

    tracker = IncrementalEpisodeTracker(
        next_episode_id=
            last_episode_id + 1,
        previous_messages=
            previous_messages
    )

    new_episodes = []

    for message in all_messages[
        start_index:
    ]:
        episode = (
            tracker.process_message(
                message
            )
        )

        if episode:
            new_episodes.append(
                episode
            )

    if new_episodes:
        # Сначала создаём embeddings.
        # Пока API не завершился успешно,
        # episodes на диск не записываем.
        embedding_records = (
            await create_embeddings(
                new_episodes
            )
        )

        for episode in new_episodes:
            append_episode(
                episode,
                episodes_filename
            )

        append_embedding_records(
            embedding_records,
            live_embeddings_filename
        )

    return (
        tracker,
        new_episodes
    )


async def process_live_episode_message(
    tracker,
    message,
    episodes_filename=
        DEFAULT_EPISODES_FILE,
    live_embeddings_filename=
        DEFAULT_LIVE_EMBEDDINGS_FILE
):
    episode = tracker.process_message(
        message
    )

    if not episode:
        return None

    # Сначала embedding.
    # Если API недоступен, сообщение всё
    # равно уже находится в chat history,
    # поэтому episode восстановится
    # при следующем запуске.
    embedding_records = (
        await create_embeddings(
            [episode]
        )
    )

    append_episode(
        episode,
        episodes_filename
    )

    append_embedding_records(
        embedding_records,
        live_embeddings_filename
    )

    return episode