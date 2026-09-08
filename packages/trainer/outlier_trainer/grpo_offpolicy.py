"""Stage 3: off-policy GRPO over the archive (custom loop; TRL has no replay buffer).

For each brief, the shipped trajectories form the group. Rewards follow section 7.2 with real
tiers. The policy's sequence log-prob on each archived completion is compared with the
reference model's (the archive came from other generators, so the reference is the behavior
proxy); the ratio is clipped; advantages come from `advantages.py` (risk transform, mean mix,
uniqueness scaling); a KL penalty to the reference is added; the reference refreshes only at
run boundaries.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from outlier_trainer.advantages import RewardConfig, advantages, tau_schedule
from outlier_trainer.data import Row, count_positives, offpolicy_groups
from outlier_trainer.guards import GuardConfig, check_guards, combination_entropy, swing_rate
from outlier_trainer.prompts import render_chat


def _seq_logprob(model, tok, prompt: str, completion: str, device) -> Any:
    import torch

    full = render_chat(tok, prompt, completion)
    prefix = render_chat(tok, prompt, None)
    ids = tok(full, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
    n_prefix = tok(prefix, return_tensors="pt", truncation=True, max_length=4096).input_ids.shape[1]
    logits = model(ids).logits[:, :-1]
    targets = ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return logp[:, max(n_prefix - 1, 0) :].sum()


def run_grpo_offpolicy(
    rows: list[Row],
    *,
    base_model: str,
    adapter: str | None,
    out_dir: Path,
    lora: dict[str, Any],
    rl: dict[str, Any],
    reward_cfg: RewardConfig,
    guard_cfg: GuardConfig,
    gold_gap: float | None,
    smoke: bool,
) -> dict[str, Any]:
    n_pos = count_positives(rows)
    groups = offpolicy_groups(rows, cfg=reward_cfg, n_positives=n_pos)
    if not groups:
        return {
            "stage": "grpo_offpolicy",
            "groups": 0,
            "skipped": "no brief has 2+ trajectories with prompts",
        }
    if smoke:
        groups = groups[:2]
        groups = [g[:4] for g in groups]
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base_model)
    ref = AutoModelForCausalLM.from_pretrained(base_model).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    if adapter:
        policy = PeftModel.from_pretrained(policy, adapter, is_trainable=True)
    else:
        policy = get_peft_model(
            policy,
            LoraConfig(
                r=int(lora.get("r", 16)),
                lora_alpha=int(lora.get("alpha", 32)),
                lora_dropout=float(lora.get("dropout", 0.05)),
                task_type="CAUSAL_LM",
            ),
        )
    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=float(rl.get("learning_rate", 1e-5))
    )
    clip_eps = float(rl.get("clip_eps", 0.2))
    kl_coef = float(rl.get("kl_coef", 0.05))
    eps_mean = float(rl.get("eps_mean", 0.1))
    beta = float(rl.get("beta", 0.5))
    steps_total = len(groups) if smoke else len(groups) * int(rl.get("epochs", 1))
    history: list[dict[str, float]] = []
    stop_reason: str | None = None
    step = 0
    steps_since_outcome = 0
    for _epoch in range(1 if smoke else int(rl.get("epochs", 1))):
        for group in groups:
            tau = tau_schedule(
                step,
                float(rl.get("tau_start", 0.1)),
                float(rl.get("tau_target", 1.0)),
                int(rl.get("tau_anneal_steps", 200)),
            )
            rewards = np.array([r for _, r in group])
            keys = [row.combination_key for row, _ in group]
            adv = advantages(rewards.tolist(), keys, tau=tau, eps_mean=eps_mean, beta=beta)
            policy.train()
            losses = []
            kls = []
            for (row, _), a in zip(group, adv, strict=True):
                logp = _seq_logprob(policy, tok, row.prompt or "", row.completion, device)
                with torch.no_grad():
                    logp_ref = _seq_logprob(ref, tok, row.prompt or "", row.completion, device)
                ratio = torch.exp(logp - logp_ref)
                a_t = torch.tensor(float(a), device=device)
                surrogate = torch.min(
                    ratio * a_t, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a_t
                )
                kl = logp - logp_ref
                loss = -surrogate + kl_coef * kl
                losses.append(loss)
                kls.append(kl.detach())
            total = torch.stack(losses).mean()
            opt.zero_grad()
            total.backward()
            opt.step()
            step += 1
            steps_since_outcome = (
                0
                if any(row.outlier_tier is not None for row, _ in group)
                else steps_since_outcome + 1
            )
            swing = swing_rate([row.typicality for row, _ in group])
            entropy = combination_entropy(keys)
            history.append(
                {
                    "step": step,
                    "loss": float(total.item()),
                    "kl": float(torch.stack(kls).mean().item()),
                    "tau": tau,
                    "swing": swing,
                    "entropy": entropy,
                }
            )
            guard = check_guards(
                swing=swing,
                entropy=entropy,
                gold_gap=gold_gap,
                steps_since_outcome=steps_since_outcome,
                cfg=guard_cfg,
            )
            if guard.stop and not smoke:
                stop_reason = "; ".join(guard.reasons)
                break
        if stop_reason:
            break
    policy.save_pretrained(str(out_dir / "adapter"))
    return {
        "stage": "grpo_offpolicy",
        "groups": len(groups),
        "steps": step,
        "planned_steps": steps_total,
        "stop_reason": stop_reason,
        "final_loss": history[-1]["loss"] if history else math.nan,
        "history": history[-50:],
        "adapter": str(out_dir / "adapter"),
    }
