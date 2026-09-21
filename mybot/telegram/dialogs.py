async def choose_dialog(client):
    dialogs = []

    print("Загружаю список диалогов...\n")

    async for dialog in client.iter_dialogs():
        dialogs.append(dialog)

    for number, dialog in enumerate(dialogs, start=1):
        print(f"{number}. {dialog.name} | ID: {dialog.id}")

    while True:
        try:
            choice = int(input("\nВведите номер нужного диалога: "))
        except ValueError:
            print("Ошибка: нужно ввести число.")
            continue

        if 1 <= choice <= len(dialogs):
            return dialogs[choice - 1]

        print("Ошибка: диалога с таким номером нет.")