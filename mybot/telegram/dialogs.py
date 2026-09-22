async def get_dialogs(client):
    dialogs = []

    async for dialog in client.iter_dialogs():
        dialogs.append(dialog)

    return dialogs


def format_dialogs(dialogs):
    lines = []

    for number, dialog in enumerate(
        dialogs,
        start=1
    ):
        name = (
            dialog.name
            or "Без имени"
        )

        lines.append(
            f"{number}. "
            f"{name} | "
            f"ID: {dialog.id}"
        )

    return "\n".join(lines)


async def choose_dialog(client):
    dialogs = await get_dialogs(
        client
    )

    print(
        "Загружаю список "
        "диалогов...\n"
    )

    print(
        format_dialogs(
            dialogs
        )
    )

    while True:
        try:
            choice = int(
                input(
                    "\nВведите номер "
                    "нужного диалога: "
                )
            )

        except ValueError:
            print(
                "Ошибка: нужно "
                "ввести число."
            )

            continue

        if 1 <= choice <= len(dialogs):
            return dialogs[
                choice - 1
            ]

        print(
            "Ошибка: диалога "
            "с таким номером нет."
        )