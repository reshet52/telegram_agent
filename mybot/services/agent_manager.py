from mybot.telegram.dialogs import (
    get_dialogs
)


class AgentManager:
    def __init__(
        self,
        telegram_client
    ):
        self.telegram_client = (
            telegram_client
        )

        self.selected_dialog = None


    async def get_dialogs(self):
        return await get_dialogs(
            self.telegram_client
        )


    async def find_dialog(
        self,
        dialog_id
    ):
        dialogs = await self.get_dialogs()

        for dialog in dialogs:
            if dialog.id == dialog_id:
                return dialog

        return None


    async def select_dialog(
        self,
        dialog_id
    ):
        dialog = await self.find_dialog(
            dialog_id
        )

        if dialog is None:
            return None

        self.selected_dialog = dialog

        return dialog


    @property
    def selected_dialog_id(self):
        if self.selected_dialog is None:
            return None

        return self.selected_dialog.id


    @property
    def selected_dialog_name(self):
        if self.selected_dialog is None:
            return None

        return (
            self.selected_dialog.name
            or f"Dialog {self.selected_dialog.id}"
        )