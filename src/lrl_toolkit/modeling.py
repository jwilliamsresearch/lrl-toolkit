"""Shared helpers to resolve and load a project's trained model + tokenizer.

Used by the evaluate and export stages. Prefers the SFT (finetune) adapter, then
the continued-pretraining adapter, falling back to the untouched base model.
"""

from __future__ import annotations

from pathlib import Path

from .config import ResolvedProject


def resolve_artifacts(project: ResolvedProject) -> tuple[Path | None, Path | None, str]:
    """Return (tokenizer_dir, adapter_dir, kind) for the best available model."""
    tok_dir = project.stage_dir("tokenizer") / "tokenizer"
    finetune_adapter = project.stage_dir("finetune") / "adapter"
    pretrain_adapter = project.stage_dir("pretrain") / "adapter"

    if finetune_adapter.exists():
        adapter, kind = finetune_adapter, "sft"
    elif pretrain_adapter.exists():
        adapter, kind = pretrain_adapter, "pretrain"
    else:
        adapter, kind = None, "base"
    return (tok_dir if tok_dir.exists() else None), adapter, kind


def load_model_and_tokenizer(
    base_id: str,
    tokenizer_dir: Path | None,
    adapter_dir: Path | None,
    *,
    token: str | None = None,
    merge: bool = False,
):
    """Load the base model + tokenizer, apply an adapter, optionally merge it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = str(tokenizer_dir) if tokenizer_dir else base_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float32, token=token)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if adapter_dir and adapter_dir.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        if merge:
            model = model.merge_and_unload()
    return model, tokenizer
