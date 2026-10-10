"""Reusable prefix inside one sampling call; never shared between owner requests."""


class PromptPrefix:
    def __init__(self, model):
        from mlx_lm.models.cache import make_prompt_cache
        self.model = model
        self.cache = make_prompt_cache(model)
        self.previous = []

    def prepare(self, tokens):
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
        common = 0
        for old, new in zip(self.previous, tokens):
            if old != new:
                break
            common += 1
        # The final input token must be processed to obtain fresh next-token logits.
        common = min(common, max(0, len(tokens) - 1))
        offsets = [getattr(c, 'offset', None) for c in self.cache]
        if (not offsets or any(type(n) is not int for n in offsets)
                or len(set(offsets)) != 1 or offsets[0] < common
                or not can_trim_prompt_cache(self.cache)):
            self.cache = make_prompt_cache(self.model)
            common = 0
        else:
            remove = offsets[0] - common
            if trim_prompt_cache(self.cache, remove) != remove:
                self.cache = make_prompt_cache(self.model)
                common = 0
        self.previous = list(tokens)
        return tokens[common:], {'prompt_cache': self.cache}, common
