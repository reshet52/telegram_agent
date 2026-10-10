"""Progressive mixed reply cards; no contact sending."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import asyncio
import logging

from mybot.services.mixed_replies import MixedReplies
from mybot.services.reply_service import DEFAULT_INSTRUCTION


def card(record):
    if record.get('stale'):
        lines = ['Переписка изменилась. Эти варианты относятся к предыдущей ситуации. Нажмите «Ответ» для нового сообщения.']
        for item in record.get('variants', []):
            source = 'API' if item['source'] == 'api' else 'Локальная'
            lines.append(f"Вариант {item['number']} · {source}\n{item['text']}")
        return '\n\n'.join(lines), None
    lines = []
    rows = []
    buttons = []
    prep = record.get('preparation', {})
    if prep.get('result', {}).get('response_action') == 'wait':
        lines.append('Можно не отвечать сейчас: ' + prep['result'].get('action_reason', 'разговор завершён.')
                     + '\nВарианты ниже — если всё же хотите ответить.')
    if prep.get('status') == 'needs_clarification':
        return ('Для точного ответа нужно уточнение:\n' + prep['result']['question']
                + '\n\nОтветьте следующим сообщением. Уточнение действует только для этой ситуации.'), InlineKeyboardMarkup(
                    [[InlineKeyboardButton('Главное меню', callback_data='ui:home')]])
    if prep.get('status') == 'pending':
        lines.append('Разбираю последнюю реплику и ищу нужные сведения…')
    elif prep.get('status') == 'failed':
        lines.append('Смысловая подготовка недоступна. Использую переписку и найденные сведения.')
        rows.append([InlineKeyboardButton('Повторить подготовку', callback_data=f"ui:mr:{record['id']}:understanding")])
    for item in record['variants']:
        source = 'API' if item['source'] == 'api' else 'Локальная · ' + (item.get('version') or 'base')[:8]
        lines.append(f"Вариант {item['number']} · {source}\n{item['text']}")
        buttons.append(InlineKeyboardButton(str(item['number']), callback_data=f"ui:rv:{record['id']}:{item['number']}"))
    if buttons:
        rows.append(buttons)
    for side, state in record['sides'].items():
        title = 'API' if side == 'api' else 'Локальная модель'
        if state['status'] == 'pending':
            lines.append(title + ': готовлю варианты…')
        elif state['status'] == 'failed':
            lines.append(title + ': ' + state['error'])
            rows.append([InlineKeyboardButton('Повторить · ' + title,
                callback_data=f"ui:mr:{record['id']}:{side}")])
        if any(c.get('status') == 'unavailable' for c in state.get('checks', [])):
            lines.append(title + ': проверка смысла недоступна — проверьте текст самостоятельно.')
    if buttons:
        rows.append([InlineKeyboardButton('🚫 Все не подходят', callback_data=f"ui:reject:{record['id']}")])
    rows.append([InlineKeyboardButton('🏠 Главное меню', callback_data='ui:home')])
    return '\n\n'.join(lines) or 'Генерирую четыре варианта…', InlineKeyboardMarkup(rows)


async def generate(bot, update, instruction=None):
    """Release Telegram's sequential update handler while the providers run."""
    await launch(bot, _generate(bot, update, instruction, bot.reply_service), update.effective_message)


async def launch(bot, operation, source):
    previous = getattr(bot, 'mixed_task', None)
    if previous and not previous.done():
        previous.cancel()
        await asyncio.gather(previous, return_exceptions=True)
    async def run():
        try:
            await operation
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception('Не удалось завершить карточку смешанных ответов')
            await source.reply_text('Не удалось завершить генерацию. Готовые варианты сохранены в журнале.')
    bot.mixed_task = asyncio.create_task(run())


async def _generate(bot, update, instruction, service):
    from mybot.services.online_learning import request_started, request_finished
    request_started(service.workspace)
    # Separate small cards avoid Telegram's text limit and preserve partial results.
    sent = None
    async def publish(record):
        nonlocal sent
        from mybot.storage.reply_journal import generation_has_draft
        if generation_has_draft(service.workspace, record['id']):
            return  # Late provider results must not reopen an already chosen card.
        if bot.workspace != service.workspace:
            record['stale'] = True
        bot.reply_clarification = ({'key': record['id'], 'workspace': str(service.workspace.root)}
            if record.get('preparation', {}).get('status') == 'needs_clarification' and not record.get('stale') else None)
        text, markup = card(record)
        if len(text) > 4000:
            # Display complete replies in separate messages; buttons keep stable IDs.
            text = 'Четыре варианта · готовые ответы ниже.\n' + '\n'.join(
                f"{s}: {r['status']}" for s, r in record['sides'].items())
            for item in record['variants']:
                marker = item['number']
                if marker not in shown:
                    await bot.send_long_text(sent or update.effective_message, f"Вариант {marker} · {item['source']}\n{item['text']}")
                    shown.add(marker)
        if sent is None:
            sent = await bot.application.bot.send_message(chat_id=bot.owner_id, text=text, reply_markup=markup)
        else:
            await sent.edit_text(text, reply_markup=markup)
    shown = set()
    try:
        await MixedReplies(service).start(service.apply_mood(instruction or DEFAULT_INSTRUCTION, bot.state.current_mood), publish)
    finally:
        request_finished(service.workspace)


async def retry(bot, action, source):
    await launch(bot, _retry(bot, action, source, bot.reply_service), source)


async def _retry(bot, action, source, service):
    _, _, key, side = action.split(':')
    from mybot.services.online_learning import request_started, request_finished
    request_started(service.workspace)
    async def publish(record):
        from mybot.storage.reply_journal import generation_has_draft
        if generation_has_draft(service.workspace, record['id']):
            return
        if bot.workspace != service.workspace:
            record['stale'] = True
        bot.reply_clarification = ({'key': record['id'], 'workspace': str(service.workspace.root)}
            if record.get('preparation', {}).get('status') == 'needs_clarification' and not record.get('stale') else None)
        text, markup = card(record)
        if len(text) <= 4000:
            await source.edit_text(text, reply_markup=markup)
        else:
            await bot.send_long_text(source, text, reply_markup=markup)
    try:
        await MixedReplies(service).retry(key, side, publish)
    except ValueError as error:
        await source.edit_text(str(error))
    finally:
        request_finished(service.workspace)


def cancel_clarification(bot):
    pending = getattr(bot, 'reply_clarification', None)
    bot.reply_clarification = None
    if pending and bot.workspace and str(bot.workspace.root) == pending['workspace']:
        from mybot.storage.atomic import read_json
        generator = MixedReplies(bot.reply_service)
        record = read_json(generator.path(pending['key']))
        if record:
            record['cancelled'] = True
            record['stale'] = True
            generator.save(record)


async def clarify_text(bot, update, text):
    pending = getattr(bot, 'reply_clarification', None)
    if not pending:
        return False
    if not bot.workspace or pending['workspace'] != str(bot.workspace.root):
        bot.reply_clarification = None
        await update.effective_message.reply_text('Диалог изменился. Запросите новые варианты.')
        return True
    if not text.strip() or len(text) > 2000:
        await update.effective_message.reply_text('Уточнение должно содержать от 1 до 2000 символов.')
        return True
    bot.reply_clarification = None
    service = bot.reply_service
    async def operation():
        from mybot.services.online_learning import request_started, request_finished
        request_started(service.workspace)
        sent = None
        async def publish(record):
            nonlocal sent
            from mybot.storage.reply_journal import generation_has_draft
            if generation_has_draft(service.workspace, record['id']):
                return
            bot.reply_clarification = None
            text, markup = card(record)
            if bot.workspace != service.workspace:
                return
            if len(text) > 4000:
                await bot.send_long_text(update.effective_message, text, reply_markup=markup)
            elif sent is None:
                sent = await bot.application.bot.send_message(chat_id=bot.owner_id, text=text, reply_markup=markup)
            else:
                await sent.edit_text(text, reply_markup=markup)
        try:
            await MixedReplies(service).clarify(pending['key'], text, publish)
        except ValueError as error:
            await update.effective_message.reply_text(str(error))
        finally:
            request_finished(service.workspace)
    await launch(bot, operation(), update.effective_message)
    return True
