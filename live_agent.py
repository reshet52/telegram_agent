from mybot.app.session_state import (
    SessionState
)
from mybot.app.terminal_interface import (
    TerminalInterface
)
from mybot.services.dialog_runtime import (
    prepare_dialog_runtime
)
from mybot.telegram.client import client
from mybot.telegram.dialogs import (
    choose_dialog
)
from mybot.telegram.live_events import (
    LiveEvents
)


RECENT_MESSAGES_LIMIT = 15


async def main():
    state = SessionState()

    print(
        "\nВыбери диалог для "
        "real-time режима:\n"
    )

    selected_dialog = (
        await choose_dialog(
            client
        )
    )

    me = await client.get_me()

    runtime = (
        await prepare_dialog_runtime(
            client=client,
            selected_dialog=
                selected_dialog,
            me=me,
            state=state,
            recent_messages_limit=
                RECENT_MESSAGES_LIMIT
        )
    )

    if runtime is None:
        return

    live_events = LiveEvents(
        client=client,
        selected_dialog=
            selected_dialog,
        me=me,
        dialog_name=
            runtime.dialog_name,
        workspace=
            runtime.workspace,
        raw_history=
            runtime.raw_history,
        recent_messages=
            runtime.recent_messages,
        known_message_ids=
            runtime.known_message_ids,
        bot_interface=
            runtime.bot_interface,
        episode_tracker=
            runtime.episode_tracker,
        recent_messages_limit=
            RECENT_MESSAGES_LIMIT
    )

    live_events.register()

    terminal_interface = (
        TerminalInterface(
            state=state,
            recent_messages=
                runtime.recent_messages,
            reply_service=
                runtime.reply_service,
            workspace=
                runtime.workspace
        )
    )

    try:
        await terminal_interface.run()

    finally:
        await runtime.bot_interface.stop()


with client:
    client.loop.run_until_complete(
        main()
    )