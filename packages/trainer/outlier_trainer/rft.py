"""Stage 1: rejection-sampling fine-tuning on tier 2+ trajectories (TRL SFTTrainer + LoRA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from outlier_trainer.data import Row, rft_examples


def run_rft(
    rows: list[Row],
    *,
    base_model: str,
    out_dir: Path,
    lora: dict[str, Any],
    epochs: float,
    lr: float,
    smoke: bool,
) -> dict[str, Any]:
    examples = rft_examples(rows)
    if not examples:
        return {"stage": "rft", "examples": 0, "skipped": "no tier 2+ examples with prompts"}
    if smoke:
        examples = examples[:8]
    from datasets import Dataset  # heavy imports stay inside the stage
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base_model)
    ds = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": c}]}
            for p, c in examples
        ]
    )
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs if not smoke else 1,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 8,
        logging_steps=1,
        save_strategy="no",
        max_steps=4 if smoke else -1,
        report_to=[],
        bf16=False,
    )
    trainer = SFTTrainer(
        model=base_model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            task_type="CAUSAL_LM",
        ),
    )
    result = trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    return {
        "stage": "rft",
        "examples": len(examples),
        "train_loss": float(result.training_loss),
        "steps": int(result.global_step),
        "adapter": str(out_dir / "adapter"),
    }
