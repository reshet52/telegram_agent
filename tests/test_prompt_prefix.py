import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from mybot.ai.prompt_prefix import PromptPrefix


class PrefixTests(unittest.TestCase):
    def cache_api(self):
        def make(model):
            return [NS(offset=0)]
        def trim(caches, count):
            for cache in caches:
                cache.offset -= count
            return count
        return NS(make_prompt_cache=make, can_trim_prompt_cache=lambda c: True, trim_prompt_cache=trim)

    def test_reuses_only_exact_prefix_and_reprocesses_final_input_token(self):
        with patch.dict(sys.modules, {'mlx_lm.models.cache': self.cache_api()}):
            prefix = PromptPrefix(object())
            suffix, kwargs, reused = prefix.prepare([1, 2, 3, 4])
            self.assertEqual((suffix, reused), ([1, 2, 3, 4], 0))
            kwargs['prompt_cache'][0].offset = 12  # Includes generated reply tokens.
            suffix, kwargs, reused = prefix.prepare([1, 2, 3, 9])
            self.assertEqual((suffix, reused), ([9], 3))
            self.assertEqual(kwargs['prompt_cache'][0].offset, 3)
            kwargs['prompt_cache'][0].offset = 8
            self.assertEqual(prefix.prepare([1, 2, 3, 9])[0], [9])
            self.assertEqual(prefix.prepare([5, 6])[0], [5, 6])

    def test_new_sampling_call_has_no_previous_context(self):
        with patch.dict(sys.modules, {'mlx_lm.models.cache': self.cache_api()}):
            first, second = PromptPrefix(object()), PromptPrefix(object())
            first.prepare([1, 2, 3])
            first.cache[0].offset = 7
            suffix, _, reused = second.prepare([1, 2, 3])
            self.assertEqual((suffix, reused), ([1, 2, 3], 0))
            self.assertIsNot(first.cache, second.cache)

    def test_incompatible_offsets_fall_back_to_complete_input(self):
        with patch.dict(sys.modules, {'mlx_lm.models.cache': self.cache_api()}):
            prefix = PromptPrefix(object())
            prefix.prepare([1, 2, 3])
            prefix.cache = [NS(offset=6), NS(offset=7)]
            suffix, _, reused = prefix.prepare([1, 2, 9])
            self.assertEqual((suffix, reused), ([1, 2, 9], 0))
