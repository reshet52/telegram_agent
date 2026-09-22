import asyncio

from mybot.memory.incremental import (
    update_incremental_memory
)
from mybot.storage.history import (
    print_messages
)


class TerminalInterface:
    def __init__(
        self,
        state,
        recent_messages,
        reply_service,
        workspace
    ):
        self.state = state
        self.recent_messages = (
            recent_messages
        )
        self.reply_service = (
            reply_service
        )
        self.workspace = workspace


    def print_help(self):
        print(
            "\nREAL-TIME РЕЖИМ ЗАПУЩЕН."
        )

        print(
            "\nТеперь программа постоянно "
            "слушает этот чат."
        )

        print(
            "\nКаждое новое сообщение "
            "автоматически сохраняется "
            "в chat_history.jsonl."
        )

        print(
            "\nЧтобы получить ответ:"
        )

        print(
            "- просто нажми Enter "
            "для обычного ответа"
        )

        print(
            "- или напиши своё указание "
            "и нажми Enter"
        )

        print(
            "- /show — показать "
            "текущий контекст"
        )

        print(
            "- /quit — выйти"
        )

        print(
            "- /mood <описание> — "
            "задать настроение"
        )

        print(
            "- /mood — показать "
            "текущее настроение"
        )

        print(
            "- /clear_mood — "
            "сбросить настроение"
        )

        print(
            "- /update_memory — "
            "обновить долговременную память"
        )


    async def update_memory(self):
        print(
            "\nОбновляю долговременную "
            "память..."
        )

        try:
            result = (
                await update_incremental_memory(
                    history_filename=
                        self.workspace.chat_history,
                    deleted_filename=
                        self.workspace.deleted_message_ids,
                    base_episode_embeddings_filename=
                        self.workspace.episode_embeddings,
                    episodes_filename=
                        self.workspace.episodes,
                    live_memory_filename=
                        self.workspace.agent_memory_live,
                    live_memory_embeddings_filename=
                        self.workspace.memory_embeddings_live,
                    state_filename=
                        self.workspace.memory_update_state
                )
            )

        except Exception as error:
            print(
                f"\nОшибка обновления памяти: "
                f"{error}"
            )

            return

        print(
            f"\nОбработано новых сообщений: "
            f'{result["messages"]}'
        )

        print(
            f"Добавлено новых memories: "
            f'{result["memories"]}'
        )


    async def generate(
        self,
        instruction=None
    ):
        print(
            "\nОтправляю запрос ИИ..."
        )

        try:
            answers = (
                await self.reply_service.generate(
                    instruction=instruction,
                    current_mood=
                        self.state.current_mood
                )
            )

        except Exception as error:
            print(
                f"\nОшибка: {error}"
            )

            return

        print(
            "\nВАРИАНТЫ ОТВЕТА:\n"
        )

        print(
            answers
        )


    async def run(self):
        self.print_help()

        while True:
            command = await asyncio.to_thread(
                input,
                "\n> "
            )

            command = command.strip()

            if command == "/quit":
                print(
                    "\nReal-time режим "
                    "остановлен."
                )

                return

            if command == "/show":
                print(
                    "\nТекущий контекст:\n"
                )

                print_messages(
                    self.recent_messages
                )

                continue

            if command == "/update_memory":
                await self.update_memory()
                continue

            if command == "/mood":
                if self.state.current_mood:
                    print(
                        "\nТекущее настроение:"
                    )

                    print(
                        self.state.current_mood
                    )

                else:
                    print(
                        "\nНастроение не задано."
                    )

                continue

            if command.startswith(
                "/mood "
            ):
                self.state.current_mood = (
                    command[
                        len("/mood "):
                    ].strip()
                )

                print(
                    "\nНастроение установлено:"
                )

                print(
                    self.state.current_mood
                )

                continue

            if command == "/clear_mood":
                self.state.current_mood = None

                print(
                    "\nНастроение сброшено."
                )

                continue

            instruction = (
                command
                if command
                else None
            )

            await self.generate(
                instruction
            )