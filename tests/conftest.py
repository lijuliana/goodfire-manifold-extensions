import numpy as np
import pytest
import torch
from transformers import BatchEncoding


class CharTokenizer:
    """Deterministic character tokenizer for tiny random models (no downloads)."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, batch, return_tensors="pt", padding=True, truncation=True, max_length=64):
        ids = [[1] + [ord(c) % 200 + 2 for c in s][: max_length - 1] for s in batch]
        n = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), n, dtype=torch.long)
        mask = torch.zeros(len(ids), n, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": mask})

    def decode(self, ids):
        return "".join(chr(int(i) - 2 + ord("a")) if int(i) >= 2 else "?" for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 200 + 2 for c in text]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
