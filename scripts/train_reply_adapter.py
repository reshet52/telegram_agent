"""Validate a dialog dataset, then explicitly train a local LoRA adapter."""

import argparse
from pathlib import Path
from uuid import uuid4

from mybot.services.learning_dataset import load_dataset
from mybot.storage.atomic import write_json
from mybot.storage.reply_practice import now
from mybot.storage.workspace import load_workspace


def resolve_workspace(account, dialog):
    workspace = load_workspace(account, dialog)
    if workspace is None or (workspace.account_id, workspace.dialog_id) != (account, dialog):
        raise ValueError('Workspace отсутствует или принадлежит другому аккаунту/диалогу.')
    return workspace


def train(workspace, dataset_id, model_path, *, steps=100, max_length=2048):
    if type(steps) is not int or not 1 <= steps <= 10000 or not 256 <= max_length <= 8192:
        raise ValueError('Некорректные пределы обучения.')
    model_path = Path(model_path).resolve()
    if not model_path.is_dir():
        raise ValueError('Укажите каталог уже скачанной базовой модели.')
    manifest, rows = load_dataset(workspace, dataset_id)
    # Optional training dependencies never enter the running Telegram process.
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                              trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError('Базовая модель должна иметь шаблон чата.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    filtered = {}
    for split, items in rows.items():
        # Reject whole overlength examples instead of truncating away the target.
        filtered[split] = [row for row in items if len(tokenizer.apply_chat_template(
            row['prompt'] + row['completion'], tokenize=True)) <= max_length]
    if not all(filtered.values()):
        raise ValueError('После ограничения длины одна из выборок пуста. Обучение не запущено.')
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                                                trust_remote_code=False)
    model.config.use_cache = False
    root = workspace.root / 'learning' / 'adapters' / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    run = {'version': 1, 'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
           'dataset_id': dataset_id, 'dataset_hashes': manifest['hashes'],
           'base_model': str(model_path), 'created_at': now(), 'status': 'training',
           'steps': steps, 'max_length': max_length,
           'counts': {key: len(value) for key, value in filtered.items()}}
    write_json(root / 'run.json', run)
    try:
        trainer = SFTTrainer(model=model, processing_class=tokenizer,
            train_dataset=Dataset.from_list(filtered['train']),
            eval_dataset=Dataset.from_list(filtered['valid']),
            peft_config=LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32,
                                   lora_dropout=0.05, target_modules='all-linear'),
            args=SFTConfig(output_dir=str(root / 'checkpoints'), max_steps=steps,
                max_length=max_length, completion_only_loss=True, packing=False,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                gradient_accumulation_steps=8, learning_rate=1e-4,
                gradient_checkpointing=True, bf16=False, fp16=False,
                eval_strategy='no', save_strategy='steps', save_steps=50,
                save_total_limit=2, report_to='none', push_to_hub=False, seed=42))
        # Evaluate the same adapter at initialization and after training.
        before = trainer.evaluate()
        trainer.train()
        after = trainer.evaluate()
        trainer.save_model(str(root / 'adapter'))
        tokenizer.save_pretrained(str(root / 'adapter'))
        run.update(status='trained_unapproved', before=before, after=after, finished_at=now())
    except BaseException:
        run.update(status='interrupted_or_failed', finished_at=now())
        write_json(root / 'run.json', run)
        raise
    write_json(root / 'run.json', run)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', type=int, required=True)
    parser.add_argument('--dialog', type=int, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model-path')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--train', action='store_true', help='Запустить вычисления; без флага только проверка.')
    args = parser.parse_args()
    workspace = resolve_workspace(args.account, args.dialog)
    manifest, _ = load_dataset(workspace, args.dataset)
    print('Проверено примеров:', manifest['counts'])
    if args.train:
        if not args.model_path:
            parser.error('--train требует --model-path')
        print('Адаптер сохранён:', train(workspace, args.dataset, args.model_path,
                                        steps=args.steps, max_length=args.max_length))


if __name__ == '__main__':
    main()
