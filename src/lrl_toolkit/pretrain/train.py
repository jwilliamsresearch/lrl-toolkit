"""Continued pretraining via LoRA/QLoRA on the cleaned corpus.

Loads the base causal LM, resizes its embeddings to the (possibly extended)
tokenizer, packs the corpus into fixed-length blocks, and trains a LoRA adapter.
QLoRA (4-bit) is used only when a CUDA GPU and bitsandbytes are both available;
otherwise it transparently falls back to plain LoRA so the same config runs on a
laptop/CPU (just slower).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..corpus import iter_documents
from ..utils import get_logger

log = get_logger("lrl.pretrain")


def _can_use_4bit() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _unsloth_available() -> bool:
    """Unsloth needs a CUDA GPU; import is deferred so CPU/CI runs never require it."""
    try:
        import importlib.util

        import torch

        return importlib.util.find_spec("unsloth") is not None and torch.cuda.is_available()
    except Exception:
        return False


def _resize_and_init(model, base_id, extended_tokenizer, token, embed_init: str) -> int:
    """Resize embeddings to the (possibly extended) tokenizer and, when requested,
    smart-initialize the new tokens. Returns how many tokens were smart-initialized."""
    from transformers import AutoTokenizer

    old_num = model.get_input_embeddings().weight.shape[0]
    if len(extended_tokenizer) == old_num:
        return 0
    model.resize_token_embeddings(len(extended_tokenizer))
    if embed_init != "subword_mean":
        return 0
    from .embed_init import smart_init_new_embeddings

    base_tok = AutoTokenizer.from_pretrained(base_id, token=token)
    return smart_init_new_embeddings(model, base_tok, extended_tokenizer, old_num)


def _build_unsloth(
    *, base_id, tokenizer, token, seq_len, use_4bit, lora_r, lora_alpha, seed, embed_init: str
):
    """Load the base model through Unsloth, adapt it to the extended tokenizer, and
    wrap it with a LoRA adapter. Returns (peft_model, n_embed_init).

    Raises on any Unsloth error so the caller can fall back to the HF path.
    """
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name=base_id,
        max_seq_length=seq_len,
        dtype=None,  # let Unsloth pick (bf16/fp16) for the GPU
        load_in_4bit=use_4bit,
        token=token,
    )
    # Adapt embeddings to our (possibly extended) tokenizer, then smart-init.
    n_embed_init = _resize_and_init(model, base_id, tokenizer, token, embed_init)
    model.config.use_cache = False
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,  # Unsloth is optimized for dropout-free LoRA
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    return model, n_embed_init


def _corpus_texts(corpus_dir: Path) -> Iterator[str]:
    for doc in iter_documents(corpus_dir):
        if doc.text:
            yield doc.text


def _pack_blocks(
    texts: Iterator[str], tokenizer, seq_len: int, max_blocks: int | None
) -> list[list[int]]:
    eos = tokenizer.eos_token_id
    # We deliberately tokenize whole documents and repack them into seq_len
    # blocks below, so the tokenizer's "sequence longer than model_max_length"
    # warning is expected and misleading here — pre-arm its once-only flag so it
    # stays quiet. The full sequence is never fed to the model.
    if hasattr(tokenizer, "deprecation_warnings"):
        _key = "sequence-length-is-longer-than-the-specified-maximum"
        tokenizer.deprecation_warnings[_key] = True
    buf: list[int] = []
    blocks: list[list[int]] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        if eos is not None:
            buf.append(eos)
        while len(buf) >= seq_len:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
            if max_blocks is not None and len(blocks) >= max_blocks:
                return blocks
    return blocks


class _BlockDataset:
    """Minimal map-style dataset of packed token blocks for the HF Trainer."""

    def __init__(self, blocks: list[list[int]]):
        import torch

        self._t = torch
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict:
        ids = self._t.tensor(self.blocks[idx], dtype=self._t.long)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": self._t.ones_like(ids)}


def run_pretraining(
    *,
    base_id: str,
    tokenizer_dir: Path | None,
    corpus_dir: Path,
    out_dir: Path,
    method: str,
    compute,
    seq_len: int,
    epochs: float,
    max_steps: int | None,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    seed: int,
    token: str | None = None,
    embed_init: str = "subword_mean",
) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    tok_src = str(tokenizer_dir) if tokenizer_dir and tokenizer_dir.exists() else base_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve precision / quantization against the actual hardware.
    want_4bit = compute.quantization.value == "4bit" and method == "qlora"
    use_4bit = want_4bit and _can_use_4bit()
    method_used = "qlora" if use_4bit else ("lora" if method in ("lora", "qlora") else "full")
    on_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (on_cuda and compute.precision == "bf16") else torch.float32

    # Optional Unsloth fast-path (single consumer GPU: ~2x faster, less VRAM).
    backend = "hf"
    n_embed_init = 0
    model = None
    if getattr(compute, "use_unsloth", False) and _unsloth_available():
        try:
            model, n_embed_init = _build_unsloth(
                base_id=base_id, tokenizer=tokenizer, token=token, seq_len=seq_len,
                use_4bit=use_4bit, lora_r=lora_r, lora_alpha=lora_alpha, seed=seed,
                embed_init=embed_init,
            )
            backend = "unsloth"
            method_used = "qlora" if use_4bit else "lora"
        except Exception as exc:  # never let the fast-path break the run
            log.warning("[pretrain] Unsloth path failed (%s); falling back to HF.", exc)
            model = None

    if model is None:  # standard HuggingFace path
        model_kwargs: dict = {"torch_dtype": dtype, "token": token}
        if use_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if want_4bit and not use_4bit:
            log.warning(
                "[pretrain] QLoRA requested but no CUDA+bitsandbytes; falling back to LoRA."
            )

        model = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)
        # Match embeddings to the (possibly extended) tokenizer + smart-init new tokens.
        n_embed_init = _resize_and_init(model, base_id, tokenizer, token, embed_init)
        model.config.use_cache = False

        if method_used in ("lora", "qlora"):
            if use_4bit:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model)
            lora = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            )
            model = get_peft_model(model, lora)

    # Only pack as many blocks as the run will actually consume. Packing is a
    # single-threaded tokenization pass over the whole corpus, which can take hours
    # on a large corpus — yet a `max_steps`-capped run needs only a small slice. When
    # a step cap is set, stop after steps x effective-batch blocks (plus one batch of
    # headroom); otherwise (epoch-based runs) pack the full corpus as before.
    max_blocks = None
    if max_steps:
        blocks_per_step = compute.per_device_batch_size * compute.gradient_accumulation_steps
        max_blocks = max_steps * blocks_per_step + blocks_per_step
    blocks = _pack_blocks(_corpus_texts(corpus_dir), tokenizer, seq_len, max_blocks=max_blocks)
    if not blocks:
        raise RuntimeError("No training blocks produced; corpus too small for the seq_len.")
    dataset = _BlockDataset(blocks)

    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=compute.per_device_batch_size,
        gradient_accumulation_steps=compute.gradient_accumulation_steps,
        num_train_epochs=epochs if not max_steps else 1,
        max_steps=max_steps if max_steps else -1,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
        # Unsloth manages its own gradient checkpointing; don't double-enable.
        gradient_checkpointing=(
            compute.gradient_checkpointing and on_cuda and backend != "unsloth"
        ),
        bf16=on_cuda and compute.precision == "bf16",
        fp16=on_cuda and compute.precision == "fp16",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    result = trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    return {
        "base_model": base_id,
        "method_requested": method,
        "method_used": method_used,
        "backend": backend,
        "device": "cuda" if on_cuda else "cpu",
        "n_blocks": len(blocks),
        "seq_len": seq_len,
        "steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "vocab_size": len(tokenizer),
        "embed_init": embed_init,
        "embed_init_tokens": n_embed_init,
        "saved_to": str(adapter_dir),
    }
