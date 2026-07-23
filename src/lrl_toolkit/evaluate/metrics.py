"""Low-level metric primitives. Perplexity lives here; the higher-level,
coverage-aware benchmarks (native_cloze, belebele, global_mmlu, flores chrF,
judge) are in the ``benchmarks/`` package and call into helpers there."""

from __future__ import annotations

import math

from ..pretrain.train import _pack_blocks


def perplexity(
    model, tokenizer, texts, *, seq_len: int = 512, max_blocks: int = 50
) -> float | None:
    """Token-level perplexity over packed blocks of the given texts.

    If the corpus is too small to fill a block at ``seq_len``, the block size is
    halved (down to 16) so small LRL corpora still get a score.
    """
    import torch

    texts = list(texts)
    blocks: list[list[int]] = []
    sl = seq_len
    while sl >= 16 and not blocks:
        blocks = _pack_blocks(iter(texts), tokenizer, sl, max_blocks=max_blocks)
        sl //= 2
    if not blocks:
        return None
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for block in blocks:
            ids = torch.tensor([block], dtype=torch.long)
            out = model(ids, labels=ids)
            n = ids.numel() - 1  # HF shifts labels internally
            total_loss += float(out.loss) * n
            total_tokens += n
    if total_tokens == 0:
        return None
    return math.exp(total_loss / total_tokens)
