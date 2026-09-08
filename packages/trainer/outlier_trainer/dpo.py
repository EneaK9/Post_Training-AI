"""Stage 2: preference pairs, tier 2+ chosen vs tier 0 rejected on the same brief (TRL DPO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from outlier_trainer.data import Row, dpo_pairs


def run_dpo(
    rows: list[Row],
    *,
    base_model: str,
    adapter: str | None,
    out_dir: Path,
    lora: dict[str, Any],
    beta: float,
    lr: float,
    smoke: bool,
) -> dict[str, Any]:
    pairs = dpo_pairs(rows)
    if not pairs:
        return {
            "stage": "dpo",
            "pairs": 0,
            "skipped": "no tier 2+ vs tier 0 pairs on a shared brief",
        }
    if smoke:
        pairs = pairs[:8]
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    ds = Dataset.from_list([{"prompt": p, "chosen": c, "rejected": r} for p, c, r in pairs])
    cfg = DPOConfig(
        output_dir=str(out_dir),
        beta=beta,
        learning_rate=lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 8,
        max_steps=4 if smoke else -1,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=None
        if adapter
        else LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            task_type="CAUSAL_LM",
        ),
    )
    result = trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    return {
        "stage": "dpo",
        "pairs": len(pairs),
        "train_loss": float(result.training_loss),
        "steps": int(result.global_step),
        "adapter": str(out_dir / "adapter"),
    }
