"""Smart initialization for the embeddings of newly added tokenizer tokens.

When the tokenizer stage extends the vocabulary, ``resize_token_embeddings``
gives every new token a fresh (normally-distributed) embedding with no relation
to its meaning. For low-resource languages that is a real handicap: the new
tokens are exactly the ones the model must learn fastest, and they start from
noise.

This module re-initializes each new token from the base model's *own* embedding
space, following the intuition behind FOCUS/WECHSEL: a new token's embedding is
the mean of the embeddings of the base-vocabulary pieces it decomposes into.
Concretely, we take the new token's surface string, re-tokenize it with the
*original* tokenizer, and average those pieces' embeddings. Tokens that cannot be
decomposed into known pieces keep the default init, so this is always safe.

No external data or dependencies are needed — only the base tokenizer, the
extended tokenizer, and the model's existing embedding table.
"""

from __future__ import annotations

from ..utils import get_logger

log = get_logger("lrl.pretrain")


def _constituent_ids(
    surface: str, base_tokenizer, max_base_id: int, skip_ids: set[int]
) -> list[int]:
    """Base-vocabulary token ids that ``surface`` decomposes into, filtered to
    real (pre-existing, non-special) embeddings."""
    if not surface:
        return []
    ids = base_tokenizer(surface, add_special_tokens=False).get("input_ids", [])
    return [i for i in ids if i < max_base_id and i not in skip_ids]


def smart_init_new_embeddings(
    model,
    base_tokenizer,
    extended_tokenizer,
    old_num_embeddings: int,
) -> int:
    """Overwrite the embeddings of tokens added after ``old_num_embeddings`` with
    the mean of their base-tokenizer constituent embeddings.

    Must be called *after* ``model.resize_token_embeddings(...)`` so the new rows
    already exist (and remain as the fallback for undecomposable tokens).

    Returns the number of tokens that were re-initialized.
    """
    import torch

    new_len = len(extended_tokenizer)
    if new_len <= old_num_embeddings:
        return 0

    input_emb = model.get_input_embeddings().weight
    output_emb_mod = model.get_output_embeddings()
    # Only touch the output matrix separately when it is not tied to the input.
    output_emb = None
    if output_emb_mod is not None and output_emb_mod.weight.data_ptr() != input_emb.data_ptr():
        output_emb = output_emb_mod.weight

    # Special-token ids are poor averaging material; exclude them.
    skip_ids = {i for i in base_tokenizer.all_special_ids if i is not None}

    n_init = 0
    with torch.no_grad():
        for new_id in range(old_num_embeddings, new_len):
            tok = extended_tokenizer.convert_ids_to_tokens(new_id)
            if tok is None:
                continue
            surface = extended_tokenizer.convert_tokens_to_string([tok])
            pieces = _constituent_ids(surface, base_tokenizer, old_num_embeddings, skip_ids)
            if not pieces:
                continue  # keep default init
            idx = torch.tensor(pieces, device=input_emb.device)
            input_emb[new_id] = input_emb[idx].mean(dim=0)
            if output_emb is not None:
                output_emb[new_id] = output_emb[idx].mean(dim=0)
            n_init += 1

    log.info(
        "[pretrain] smart embed init: re-initialized %d/%d new tokens from base subwords.",
        n_init,
        new_len - old_num_embeddings,
    )
    return n_init
