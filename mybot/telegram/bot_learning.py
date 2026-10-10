"""Dataset export and explicitly requested local training of reviewed replies."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from mybot.services.learning_dataset import prepare_dataset
from mybot.services.reviewed_dataset import prepare_reviewed, reviewed_examples
from mybot.services.local_training import available, run_local_training
from mybot.storage.atomic import durable_call


async def handle_learning(bot, action, update):
    if not bot.is_owner(update):
        return
    try:
        ws = bot.goal.workspace()
        if action.startswith('ui:lt:'):
            async with bot.runtime_controller.lock:
                if action.split(':') != ['ui', 'lt', str(ws.account_id), str(ws.dialog_id)]:
                    raise ValueError('Диалог изменился. Откройте обучение заново.')
                if not available():
                    raise ValueError('Локальная модель или окружение MLX не найдены.')
                manifest = await durable_call(prepare_reviewed, ws)
            await bot.callbacks.show(None,
                f"Локальное обучение · {ws.dialog_name}\n"
                f"Примеров: {manifest['counts']['train']}; проверка: {manifest['counts']['valid']}.\n"
                'Запускаю 50 обновлений весов адаптера на Mac. Это может занять несколько минут. '
                'Закройте отдельный чат MLX, чтобы освободить память.')
            run = await run_local_training(ws, manifest['id'])
            await bot.callbacks.show(None,
                f'Локальное обучение завершено · {ws.dialog_name}\n\n'
                f"Обновлений весов: {run['steps']}\n"
                f"Ошибка на проверке: {run['loss_before']:.3f} → {run['loss_after']:.3f} "
                '(меньше — лучше).\n'
                'Веса адаптера действительно изменились. Адаптер сохранён отдельно для этого диалога. '
                'Рабочая модель пока не переключена; снижение ошибки не гарантирует качество ответов.')
            return
        async with bot.runtime_controller.lock:
            if action == 'ui:learning':
                count = len(await durable_call(reviewed_examples, ws))
                await bot.callbacks.show(None,
                    f'Локальное обучение стилю\n\nВыбрано ответов: {count}.\n'
                    'Оценивайте случайные отрывки: «Отправил бы», «Мой вариант» или '
                    '«Учиться на этом ответе». Для первого пробного запуска нужно минимум '
                    '10 непересекающихся примеров; обычно потребуется больше отрывков.\n\n'
                    'Кнопка обучения запускает 50 обновлений весов на этом Mac. '
                    'Выбирайте примеры обычного общения. Проверочная часть не участвует в обучении.\n\n'
                    '«Подготовить набор» — отдельный экспорт истории, без запуска обучения. '
                    'Обученный адаптер сохраняется отдельно; рабочая модель автоматически не меняется.',
                    InlineKeyboardMarkup([
                        [InlineKeyboardButton('Обучить локально · 50 шагов',
                            callback_data=f'ui:lt:{ws.account_id}:{ws.dialog_id}')],
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
        logging.exception('Не удалось выполнить действие обучения')
        await bot.callbacks.show(None, 'Действие обучения не завершено. Исходная переписка не изменена.')
