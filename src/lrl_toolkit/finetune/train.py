"""Supervised fine-tuning (SFT) on accepted conversational pairs, via TRL.

Loads the base model, applies the continued-pretraining adapter (if present) so
SFT builds on the language adaptation, ensures a chat template, and trains a LoRA
adapter on the ``messages``-format data. Like pretraining, QLoRA degrades to LoRA
when there is no CUDA GPU.
"""

from __future__ import annotations

from pathlib import Path

from ..convdata.schema import read_jsonl
from ..pretrain.train import _can_use_4bit
from ..utils import get_logger

log = get_logger("lrl.finetune")

# Minimal ChatML template, used when the base tokenizer has none (e.g. base LMs).
_CHATML = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)


def run_sft(
    *,
    base_id: str,
    tokenizer_dir: Path | None,
    pretrain_adapter_dir: Path | None,
    pairs_path: Path,
    out_dir: Path,
    method: str,
    compute,
    max_seq_len: int,
    epochs: float,
    max_steps: int | None,
    learning_rate: float,
    seed: int,
    token: str | None = None,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = read_jsonl(pairs_path)
    if not rows:
        raise RuntimeError(f"No accepted conversational pairs at {pairs_path}.")
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    tok_src = str(tokenizer_dir) if tokenizer_dir and tokenizer_dir.exists() else base_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = _CHATML

    want_4bit = compute.quantization.value == "4bit" and method == "qlora"
    use_4bit = want_4bit and _can_use_4bit()
    method_used = "qlora" if use_4bit else ("lora" if method in ("lora", "qlora") else "full")
    on_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (on_cuda and compute.precision == "bf16") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, token=token)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Fold in the continued-pretraining adapter so SFT starts from the adapted LM.
    if pretrain_adapter_dir and pretrain_adapter_dir.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(pretrain_adapter_dir))
        model = model.merge_and_unload()
        log.info("[finetune] merged continued-pretraining adapter.")

    peft_config = None
    if method_used in ("lora", "qlora"):
        peft_config = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        )
    if want_4bit and not use_4bit:
        log.warning("[finetune] QLoRA requested but no CUDA+bitsandbytes; falling back to LoRA.")

    args = SFTConfig(
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
        max_length=max_seq_len,
        packing=False,
        bf16=on_cuda and compute.precision == "bf16",
        fp16=on_cuda and compute.precision == "fp16",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    return {
        "base_model": base_id,
        "method_requested": method,
        "method_used": method_used,
        "device": "cuda" if on_cuda else "cpu",
        "n_examples": len(rows),
        "steps": int(result.global_step),
        "train_loss": float(result.training_loss),
        "saved_to": str(adapter_dir),
    }
