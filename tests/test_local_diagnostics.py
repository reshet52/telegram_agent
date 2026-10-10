import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.local_diagnostics import prepare, render, sha
from mybot.storage.atomic import write_json
from mybot.storage.workspace import Workspace


class DiagnosticsTests(unittest.TestCase):
    def test_candidate_is_copied_without_activation_or_historical_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            key = 'a' * 32
            source = ws.root / 'learning/mlx_runs' / key
            checkpoint = source / 'checkpoints/final'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'adapters.safetensors').write_bytes(b'rejected weights')
            write_json(checkpoint / 'state.json', {'files': {'adapters.safetensors': sha(checkpoint / 'adapters.safetensors')}})
            write_json(source / 'checkpoint.json', {'folder': 'final', 'state_sha256': sha(checkpoint / 'state.json')})
            write_json(source / 'adapter_config.json', {})
            write_json(source / 'run.json', dict(account_id=1, dialog_id=2, status='trained_unapproved',completed_steps=5,steps=5))
            current = ws.root / 'current'
            current.mkdir()
            (current / 'adapters.safetensors').write_bytes(b'active weights')
            case = {'id': 'case', 'categories': ['question'], 'instruction': 'answer', 'results': {
                'baseline': {'sides': {'local': {'request': {'messages': [{'role': 'user', 'content': 'question'}]}}}}}}
            context = [dict(message_id=1, sender='Other',text='question',date=None)]
            with patch('scripts.local_diagnostics.selected',return_value='b'*32), \
                 patch('scripts.local_diagnostics.adapter',return_value=current), \
                 patch('scripts.local_diagnostics.reply_evaluation.load',return_value={'cases':[case]}), \
                 patch('scripts.local_diagnostics.reply_evaluation.validate'), \
                 patch('scripts.local_diagnostics.PracticeStore') as store:
                store.return_value.source.return_value = (context,[{'text':'HIDDEN_OWNER_TARGET'}])
                root,data = prepare(ws,key)
            self.assertEqual(len(data['cases']),4)
            self.assertNotIn('HIDDEN_OWNER_TARGET',(root/'diagnostics.json').read_text())
            self.assertEqual((current/'adapters.safetensors').read_bytes(),b'active weights')
            self.assertEqual((checkpoint/'adapters.safetensors').read_bytes(),b'rejected weights')
            self.assertFalse((ws.root/'learning/selected_model.json').exists())
            self.assertFalse((source/'adapters.safetensors').exists())

    def test_review_escapes_generated_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data={'cases':[{'context':[], 'instruction':'<script>bad</script>',
                           'results':{'candidate':{'variants':['<img src=x onerror=bad()>']}}}]}
            render(root,data)
            page=(root/'review.html').read_text()
            self.assertNotIn('<script>',page)
            self.assertNotIn('<img',page)
            self.assertIn('&lt;img',page)


if __name__=='__main__':
    unittest.main()
