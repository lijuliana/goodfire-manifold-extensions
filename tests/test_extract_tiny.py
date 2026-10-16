"""Hook + truncation + padding correctness on a random tiny Llama (CPU, no downloads)."""

import numpy as np
import torch
from conftest import CharTokenizer
from transformers import LlamaConfig, LlamaModel

from manifolds.extract import ActivationExtractor, ExtractConfig


def _tiny(tmp_path):
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    model = LlamaModel(cfg).eval()
    model.save_pretrained(tmp_path / "tiny")
    return model


def test_truncated_hook_equals_full_model_hidden_state(tmp_path):
    full = _tiny(tmp_path)
    tok = CharTokenizer()
    ext = ActivationExtractor(
        ExtractConfig(model_id=str(tmp_path / "tiny"), layer=1, dtype="float32", device="cpu",
                      batch_size=2, truncate_to_layer=True),
        tokenizer=tok,
    )
    assert ext.model.config.num_hidden_layers == 2 and ext.n_layers_full == 4
    prompts = ["short", "a much longer prompt here"]
    acts = ext.extract(prompts, progress=False)
    assert acts.shape == (2, 32)
    for i, p in enumerate(prompts):
        enc = tok([p])
        with torch.no_grad():
            hs = full(**enc, output_hidden_states=True).hidden_states
        ref = hs[2][0, -1].numpy()  # output of block 1 == hidden_states[2]
        np.testing.assert_allclose(acts[i], ref, rtol=1e-4, atol=1e-5)


def test_padding_does_not_change_last_token(tmp_path):
    _tiny(tmp_path)
    ext = ActivationExtractor(
        ExtractConfig(model_id=str(tmp_path / "tiny"), layer=3, dtype="float32", device="cpu",
                      batch_size=8, truncate_to_layer=False),
        tokenizer=CharTokenizer(),
    )
    prompts = ["x", "xy", "a longer one", "and the longest prompt of all"]
    batched = ext.extract(prompts, progress=False)
    single = np.stack([ext.extract([p], progress=False)[0] for p in prompts])
    np.testing.assert_allclose(batched, single, rtol=1e-4, atol=1e-5)


def test_span_token_and_layer_selection(tmp_path):
    full = _tiny(tmp_path)
    tok = CharTokenizer()
    ext = ActivationExtractor(
        ExtractConfig(model_id=str(tmp_path / "tiny"), layer=3, dtype="float32", device="cpu",
                      batch_size=4, truncate_to_layer=False),
        tokenizer=tok,
    )
    prompts = ["abcdef", "xy"]
    spans = [(0, 3), (0, 1)]           # read the 3rd and 1st characters
    acts = ext.extract(prompts, progress=False, spans=spans, layer=1)
    for i, (p, (_a, b)) in enumerate(zip(prompts, spans, strict=True)):
        enc = tok([p])
        with torch.no_grad():
            hs = full(**enc, output_hidden_states=True).hidden_states
        ti = len(tok(p[:b])["input_ids"]) - 1
        np.testing.assert_allclose(acts[i], hs[2][0, ti].numpy(), rtol=1e-4, atol=1e-5)
    assert ext.n_span_mismatch == 0
