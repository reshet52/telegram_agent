from dataclasses import dataclass

from mybot.episodes.incremental import (
    initialize_episode_tracker
)
from mybot.memory.manager import (
    load_agent_memory
)
from mybot.services.reply_service import (
    ReplyService
)
from mybot.storage.deletions import (
    check_recent_deletions,
    mark_messages_deleted
)
from mybot.storage.history import (
    get_last_saved_message_id,
    load_all_messages,
    load_last_messages,
    print_messages
)
from mybot.storage.workspace import (
    create_workspace
)
from mybot.telegram.bot_interface import (
    BotInterface
)
from mybot.telegram.exporter import (
    append_messages_data,
    fetch_messages_after
)


@dataclass
class DialogRuntime:
    dialog_name: str
    workspace: object
    raw_history: list
    recent_messages: list
    known_message_ids: set
    reply_service: ReplyService
    bot_interface: BotInterface
    episode_tracker: object


async def confirm_new_workspace(
    client,
    selected_dialog,
    workspace
):
    history_exists = (
        workspace.chat_history.exists()
        and workspace.chat_history.stat().st_size > 0
    )

    if history_exists:
        return True

    print(
        "\nЭто новый workspace."
    )

    print(
        "Проверяю размер истории "
        "диалога в Telegram..."
    )

    history_info = (
        await client.get_messages(
            selected_dialog.id,
            limit=0
        )
    )

    total_messages = (
        history_info.total
        or 0
    )

    print(
        f"\nСообщений в диалоге: "
        f"{total_messages}"
    )

    print(
        "\nПри первом запуске:"
    )

    print(
        "- будет загружена история "
        "этого диалога;"
    )

    print(
        "- будут построены episodes;"
    )

    print(
        "- будут созданы embeddings "
        "для новых episodes."
    )

    confirmation = input(
        "\nИнициализировать диалог? "
        "[y/N]: "
    )

    confirmation = (
        confirmation
        .strip()
        .lower()
    )

    if confirmation not in {
        "y",
        "yes",
        "д",
        "да"
    }:
        print(
            "\nИнициализация отменена."
        )

        return False

    return True


async def prepare_dialog_runtime(
    client,
    selected_dialog,
    me,
    state,
    recent_messages_limit=15
):
    dialog_name = (
        selected_dialog.name
        or f"Dialog {selected_dialog.id}"
    )

    workspace = create_workspace(
        account_id=me.id,
        dialog_id=selected_dialog.id,
        dialog_name=dialog_name
    )

    print(
        "\nWorkspace:"
    )

    print(
        workspace.root
    )

    print(
        f"\nВыбран диалог: "
        f"{dialog_name}"
    )

    should_continue = (
        await confirm_new_workspace(
            client=client,
            selected_dialog=
                selected_dialog,
            workspace=workspace
        )
    )

    if not should_continue:
        return None

    last_saved_message_id = (
        get_last_saved_message_id(
            workspace.chat_history
        )
    )

    print(
        "\nПоследнее сообщение "
        "в локальной истории:"
    )

    print(
        f"ID {last_saved_message_id}"
    )

    print(
        "\nПроверяю сообщения, "
        "которые появились, "
        "пока агент был выключен..."
    )

    new_messages = (
        await fetch_messages_after(
            client=client,
            dialog=selected_dialog,
            min_message_id=
                last_saved_message_id,
            me_id=me.id
        )
    )

    if new_messages:
        append_messages_data(
            new_messages,
            filename=
                workspace.chat_history
        )

        print(
            f"Догружено новых сообщений: "
            f"{len(new_messages)}"
        )

    else:
        print(
            "Новых сообщений нет."
        )

    print(
        "\nПроверяю недавние "
        "удалённые сообщения..."
    )

    raw_history = load_all_messages(
        filename=
            workspace.chat_history,
        include_deleted=True,
        deleted_filename=
            workspace.deleted_message_ids
    )

    deleted_while_offline = (
        await check_recent_deletions(
            client=client,
            dialog_id=
                selected_dialog.id,
            messages=raw_history,
            deleted_filename=
                workspace.deleted_message_ids
        )
    )

    newly_deleted = (
        mark_messages_deleted(
            deleted_while_offline,
            filename=
                workspace.deleted_message_ids
        )
    )

    if newly_deleted:
        print(
            f"Найдено удалённых сообщений: "
            f"{len(newly_deleted)}"
        )

    else:
        print(
            "Новых удалённых "
            "сообщений не найдено."
        )

    recent_messages = (
        load_last_messages(
            filename=
                workspace.chat_history,
            count=
                recent_messages_limit,
            deleted_filename=
                workspace.deleted_message_ids
        )
    )

    known_message_ids = {
        message.get("message_id")
        for message in recent_messages
        if message.get("message_id")
        is not None
    }

    print(
        "\nТекущий контекст:\n"
    )

    print_messages(
        recent_messages
    )

    memory = load_agent_memory(
        user_profile_filename=
            workspace.user_profile,
        person_profile_filename=
            workspace.person_profile
    )

    reply_service = ReplyService(
        recent_messages=
            recent_messages,
        memory=memory,
        workspace=workspace
    )

    bot_interface = BotInterface(
        owner_id=me.id,
        recent_messages=
            recent_messages,
        reply_service=
            reply_service,
        state=state,
        workspace=workspace,
        telegram_client=client
    )

    print(
        "\nПроверяю новые эпизоды..."
    )

    (
        episode_tracker,
        new_episodes
    ) = await initialize_episode_tracker(
        history_filename=
            workspace.chat_history,
        episodes_filename=
            workspace.episodes,
        live_embeddings_filename=
            workspace.episode_embeddings_live,
        deleted_filename=
            workspace.deleted_message_ids
    )

    if new_episodes:
        print(
            f"Добавлено новых эпизодов: "
            f"{len(new_episodes)}"
        )

        print(
            "Для них созданы только "
            "новые embeddings."
        )

    else:
        print(
            "Новых завершённых "
            "эпизодов нет."
        )

    await bot_interface.start()

    return DialogRuntime(
        dialog_name=dialog_name,
        workspace=workspace,
        raw_history=raw_history,
        recent_messages=
            recent_messages,
        known_message_ids=
            known_message_ids,
        reply_service=
            reply_service,
        bot_interface=
            bot_interface,
        episode_tracker=
            episode_tracker
    )