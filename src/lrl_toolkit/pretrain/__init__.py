"""Pretrain stage: continued causal-LM pretraining on the clean corpus.

M0 status: scaffolding — records the training plan resolved against the compute
profile. M2 implements QLoRA/LoRA continued pretraining via Transformers + PEFT
(+ optional Unsloth), with sequence packing and gradient checkpointing.
"""

from __future__ import annotations

from ..pipeline.base import Stage, StageContext, StageResult
from ..utils import get_logger, write_json

log = get_logger("lrl.pretrain")


class PretrainStage(Stage):
    name = "pretrain"

    def run(self, ctx: StageContext) -> StageResult:
        cfg = ctx.project.config.pretrain
        compute = ctx.project.compute_profile
        model = ctx.project.model_profile
        out_dir = ctx.stage_dir(self.name)

        seq_len = cfg.seq_len
        if compute.max_seq_len_cap:
            seq_len = min(seq_len, compute.max_seq_len_cap)

        plan = {
            "base_model": model.hf_id,
            "method": cfg.method.value,
            "seq_len": seq_len,
            "epochs": cfg.epochs,
            "max_steps": cfg.max_steps,
            "learning_rate": cfg.learning_rate,
            "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha},
            "compute": {
                "device": compute.device,
                "precision": compute.precision,
                "quantization": compute.quantization.value,
                "per_device_batch_size": compute.per_device_batch_size,
                "gradient_accumulation_steps": compute.gradient_accumulation_steps,
                "gradient_checkpointing": compute.gradient_checkpointing,
                "use_unsloth": compute.use_unsloth,
                "distributed": compute.distributed.value,
            },
            "status": "placeholder",
        }
        card_path = write_json(out_dir / "pretrain_card.json", plan)
        log.info("[pretrain] %s via %s (seq_len=%s)", model.hf_id, cfg.method.value, seq_len)

        return StageResult(outputs=[ctx.relpath(card_path)], metrics={"method": cfg.method.value})
