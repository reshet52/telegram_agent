"""Local dataset preparation; no downloads, model training or paid calls."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from mybot.services.learning_dataset import prepare_dataset
from mybot.storage.atomic import durable_call


async def handle_learning(bot, action, update):
    if not bot.is_owner(update):
        return
    try:
        ws = bot.goal.workspace()
        async with bot.runtime_controller.lock:
            if action == 'ui:learning':
                await bot.callbacks.show(None,
                    'Данные для обучения\n\n'
                    'Соберу пары «предыдущая переписка → ваш реальный ответ». '
                    'Поздняя история останется для проверки; отрывки ручной тренировки '
                    'не войдут в обучение. Используются только текстовые сообщения этого диалога.\n\n'
                    'Подготовка выполняется локально без запросов модели. '
                    'Дообучение и переключение рабочей модели этим не запускаются.',
                    InlineKeyboardMarkup([
                        [InlineKeyboardButton('Подготовить набор',
                            callback_data=f'ui:ld:{ws.account_id}:{ws.dialog_id}')],
                        [InlineKeyboardButton('⬅️ Тренировка', callback_data='ui:practice')]]))
                return
            if action.split(':') != ['ui', 'ld', str(ws.account_id), str(ws.dialog_id)]:
                raise ValueError('Диалог изменился. Откройте подготовку заново.')
            await bot.callbacks.show(None, 'Собираю локальный набор для обучения…')
            manifest = await durable_call(prepare_dataset, ws)
            await bot.callbacks.show(None,
                f"Набор готов\nОбучение: {manifest['counts']['train']} примеров\n"
                f"Проверка: {manifest['counts']['valid']} примеров\n\n"
                f"ID набора: {manifest['id']}\n\n"
                'Сохранён в каталоге этого диалога. Обучение весов ещё не запускалось.',
                InlineKeyboardMarkup([[InlineKeyboardButton('⬅️ Тренировка', callback_data='ui:practice')]]))
    except ValueError as error:
        await bot.callbacks.show(None, str(error))
    except Exception:
        logging.exception('Не удалось подготовить обучающий набор')
        await bot.callbacks.show(None, 'Набор не подготовлен. Исходная переписка не изменена.')
