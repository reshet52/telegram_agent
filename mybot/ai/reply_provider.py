"""Reply-only backend selection; analysis and embeddings retain their own routing."""

from urllib.parse import urlparse

from openai import AsyncOpenAI

from mybot.ai.reply_contract import REPLY_INSTRUCTIONS, VARIANT_INSTRUCTIONS
from mybot.config import Config


def reply_model():
    return Config.REPLY_MODEL or Config.OPENAI_MODEL


def request_snapshot(prompt, *, allow_decision=False, count=3):
    if count not in (1, 2, 3, 4):
        raise ValueError("Количество вариантов должно быть от 1 до 4.")
    if allow_decision and count != 3:
        raise ValueError("Решения поддерживают прежний формат из трёх вариантов.")
    instructions = REPLY_INSTRUCTIONS if allow_decision else VARIANT_INSTRUCTIONS
    if count != 3:
        instructions = instructions.replace("три", str(count)).replace('"...","...","..."', ",".join(['"..."'] * count))
    return {'provider': Config.REPLY_PROVIDER, 'model': reply_model(),
            'instructions': instructions, 'input': prompt,
            'max_output_tokens': 1600, 'count': count}


def report_usage(usage, compatible=False):
    if usage is None:
        return
    print('\nИспользование финальной модели:')
    print(f"Input tokens: {getattr(usage, 'prompt_tokens' if compatible else 'input_tokens', None)}")
    print(f"Output tokens: {getattr(usage, 'completion_tokens' if compatible else 'output_tokens', None)}")
    print(f"Total tokens: {getattr(usage, 'total_tokens', None)}")


async def generate_reply(prompt, default_client, *, allow_decision=False, count=3):
    request = request_snapshot(prompt, allow_decision=allow_decision, count=count)
    return await execute_request(request, default_client)


def structured_request(instructions, payload, *, max_output_tokens=2200):
    """Analysis/checking use the same configured reply provider and inspectable request."""
    import json
    return {'provider': Config.REPLY_PROVIDER, 'model': reply_model(),
            'instructions': instructions, 'input': json.dumps(payload, ensure_ascii=False),
            'max_output_tokens': max_output_tokens}


async def execute_request(request, default_client):
    if request['provider'] == 'openai':
        response = await default_client.responses.create(
            model=request['model'], instructions=request['instructions'],
            input=request['input'], max_output_tokens=request['max_output_tokens'], store=False)
        report_usage(response.usage)
        return response.output_text
    if request['provider'] != 'compatible':
        raise ValueError('REPLY_PROVIDER должен быть openai или compatible.')
    url = urlparse(Config.REPLY_BASE_URL)
    if (url.scheme not in {'http', 'https'} or not url.hostname
            or url.username or url.password or url.query or url.fragment
            or not Config.REPLY_MODEL):
        raise ValueError('Для compatible задайте REPLY_BASE_URL и REPLY_MODEL; '
                         'ключ укажите отдельно в REPLY_API_KEY.')
    # Never forward the OpenAI key to a different endpoint or silently fall back.
    async with AsyncOpenAI(base_url=Config.REPLY_BASE_URL,
                           api_key=Config.REPLY_API_KEY or 'local-no-key',
                           max_retries=0, timeout=90.0) as client:
        response = await client.chat.completions.create(
            model=request['model'], messages=[
                {'role': 'system', 'content': request['instructions']},
                {'role': 'user', 'content': request['input']}],
            max_tokens=request['max_output_tokens'])
        if not response.choices or not response.choices[0].message.content:
            raise ValueError('Модель ответов не вернула текст.')
        report_usage(getattr(response, 'usage', None), compatible=True)
        return response.choices[0].message.content
