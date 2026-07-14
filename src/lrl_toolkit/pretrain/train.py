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


def _corpus_texts(corpus_dir: Path) -> Iterator[str]:
    for doc in iter_documents(corpus_dir):
        if doc.text:
            yield doc.text


def _pack_blocks(
    texts: Iterator[str], tokenizer, seq_len: int, max_blocks: int | None
) -> list[list[int]]:
    eos = tokenizer.eos_token_id
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

    model_kwargs: dict = {"torch_dtype": dtype, "token": token}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if want_4bit and not use_4bit:
        log.warning("[pretrain] QLoRA requested but no CUDA+bitsandbytes; falling back to LoRA.")

    model = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs)
    # Match embeddings to the (possibly extended) tokenizer.
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
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

    blocks = _pack_blocks(_corpus_texts(corpus_dir), tokenizer, seq_len, max_blocks=None)
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
        gradient_checkpointing=compute.gradient_checkpointing and on_cuda,
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
        "device": "cuda" if on_cuda else "cpu",
        "n_blocks": len(blocks),
        "seq_len": seq_len,
        "steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "vocab_size": len(tokenizer),
        "saved_to": str(adapter_dir),
    }
