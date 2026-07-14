"""Tokenizer extension: adapt a base tokenizer to a target language's script.

The core LRL win: base tokenizers fragment under-represented scripts into many
subword pieces ("high fertility"), which wastes context and slows training. We
train a tokenizer on the cleaned target corpus and merge its novel pieces into
the base vocabulary, then report fertility before/after.

Embedding resizing to match the new vocabulary happens later, at model load time
in the pretrain stage (``model.resize_token_embeddings(len(tokenizer))``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..corpus import iter_documents
from ..utils import get_logger

log = get_logger("lrl.tokenizer")


def _corpus_texts(corpus_dir: Path, limit: int | None = None) -> Iterator[str]:
    n = 0
    for doc in iter_documents(corpus_dir):
        if doc.text:
            yield doc.text
            n += 1
            if limit is not None and n >= limit:
                return


def fertility(tokenizer, texts: list[str]) -> float:
    """Average subword tokens per whitespace word (lower is better)."""
    total_tokens = 0
    total_words = 0
    for t in texts:
        words = t.split()
        if not words:
            continue
        total_words += len(words)
        total_tokens += len(tokenizer.tokenize(t))
    return total_tokens / max(total_words, 1)


def build_extended_tokenizer(
    base_id: str,
    corpus_dir: Path,
    *,
    strategy: str = "extend",
    added_tokens: int = 8000,
    out_dir: Path,
    token: str | None = None,
    sample_docs: int = 2000,
) -> dict:
    """Build and save the (possibly extended) tokenizer. Returns a report dict."""
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(base_id, token=token)
    sample = list(_corpus_texts(corpus_dir, limit=sample_docs))
    base_fertility = fertility(base, sample) if sample else None

    n_added = 0
    if strategy == "none" or not sample:
        tokenizer = base
    elif strategy == "train":
        if not base.is_fast:
            raise RuntimeError(f"{base_id} lacks a fast tokenizer; cannot train a new one.")
        vocab_size = added_tokens or base.vocab_size
        tokenizer = base.train_new_from_iterator(iter(sample), vocab_size=vocab_size)
    else:  # extend (default)
        if not base.is_fast:
            raise RuntimeError(f"{base_id} lacks a fast tokenizer; cannot extend it.")
        target_size = base.vocab_size + added_tokens
        trained = base.train_new_from_iterator(iter(sample), vocab_size=target_size)
        base_vocab = set(base.get_vocab())
        novel = [tok for tok in trained.get_vocab() if tok not in base_vocab]
        novel = novel[:added_tokens]
        n_added = base.add_tokens(novel)
        tokenizer = base

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    ext_fertility = fertility(tokenizer, sample) if sample else None

    report = {
        "base_tokenizer": base_id,
        "strategy": strategy,
        "requested_added_tokens": added_tokens,
        "tokens_added": n_added,
        "base_vocab_size": base.vocab_size,
        "final_vocab_size": len(tokenizer),
        "fertility": {"base": base_fertility, "extended": ext_fertility},
        "sample_docs": len(sample),
        "saved_to": str(out_dir),
    }
    log.info(
        "[tokenizer] %s: +%d tokens, fertility %.3f -> %.3f",
        strategy,
        n_added,
        base_fertility or 0.0,
        ext_fertility or 0.0,
    )
    return report
