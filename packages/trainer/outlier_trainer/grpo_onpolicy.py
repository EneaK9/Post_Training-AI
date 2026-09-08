"""Stage 4: the literal section 7.3 loop with TRL's GRPOTrainer.

Fresh rollouts can never carry a real outcome, so the reward here is reward-model shaping plus
penalties: exactly the failure mode section 6 warns about. It is therefore gated: run it against
the simulator or pass --allow-rm-reward on purpose. Advantages are overridden after TRL computes
its own by subclassing `_generate_and_score_completions`, since TRL 1.12 has no advantage hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from outlier_trainer.advantages import advantages, tau_schedule


def run_grpo_onpolicy(
    prompts: list[str],
    *,
    base_model: str,
    adapter: str | None,
    out_dir: Path,
    lora: dict[str, Any],
    rl: dict[str, Any],
    reward_fn,
    combo_fn,
    vllm_url: str | None,
    smoke: bool,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    eps_mean = float(rl.get("eps_mean", 0.1))
    beta = float(rl.get("beta", 0.5))

    class OutlierGRPOTrainer(GRPOTrainer):
        _step_counter = 0

        def _generate_and_score_completions(self, inputs):  # type: ignore[override]
            out = super()._generate_and_score_completions(inputs)
            rewards = getattr(self, "_last_rewards", None)
            completions = getattr(self, "_last_completions", None)
            if rewards is not None and completions is not None and "advantages" in out:
                tau = tau_schedule(
                    OutlierGRPOTrainer._step_counter,
                    float(rl.get("tau_start", 0.1)),
                    float(rl.get("tau_target", 1.0)),
                    int(rl.get("tau_anneal_steps", 200)),
                )
                keys = combo_fn(completions)
                adv = advantages(
                    [float(r) for r in rewards], keys, tau=tau, eps_mean=eps_mean, beta=beta
                )
                out["advantages"] = torch.tensor(
                    adv, dtype=out["advantages"].dtype, device=out["advantages"].device
                )
            OutlierGRPOTrainer._step_counter += 1
            return out

    def wrapped_reward(prompts, completions, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        rewards = reward_fn(prompts, texts, **kwargs)
        trainer._last_rewards = list(rewards)  # type: ignore[attr-defined]
        trainer._last_completions = texts  # type: ignore[attr-defined]
        return rewards

    ds = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": p}]} for p in (prompts[:2] if smoke else prompts)]
    )
    cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_generations=int(rl.get("k", 32)) if not smoke else 2,
        max_completion_length=1024 if not smoke else 64,
        learning_rate=float(rl.get("learning_rate", 1e-5)),
        beta=float(rl.get("kl_coef", 0.05)),
        epsilon=float(rl.get("clip_eps", 0.2)),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=2 if smoke else int(rl.get("max_steps", 200)),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_vllm=bool(vllm_url) and not smoke,
        vllm_mode="server" if vllm_url else "colocate",
        bf16=False,
    )
    trainer = OutlierGRPOTrainer(
        model=base_model,
        reward_funcs=wrapped_reward,
        args=cfg,
        train_dataset=ds,
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
        "stage": "grpo_onpolicy",
        "prompts": len(ds),
        "train_loss": float(result.training_loss),
        "steps": int(result.global_step),
        "adapter": str(out_dir / "adapter"),
    }
