"""Trainer entrypoint: `python -m outlier_trainer.run --stage rft --snapshot archive.parquet ...`

Reads the Parquet snapshot, runs one stage, writes `<out>/run.json` with metrics and the adapter
path. The backend's `train` job launches this as a subprocess and copies run.json into the
`training_runs` row. GPU dependencies are imported inside the stages so `--help` and `--dry`
work anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from outlier_trainer.advantages import RewardConfig
from outlier_trainer.data import count_positives, load_snapshot
from outlier_trainer.guards import GuardConfig

STAGES = ("rft", "dpo", "grpo_offpolicy", "grpo_onpolicy")


def load_rl_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "rl": cfg.get("rl", {}),
        "reward_model": cfg.get("reward_model", {}),
        "generation": cfg.get("generation", {}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Outlier AI Loop B trainer")
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--snapshot", required=True, help="Parquet archive snapshot")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter to continue from")
    ap.add_argument("--config", default=None, help="config/config.yaml for rl:/reward_model:")
    ap.add_argument("--smoke", action="store_true", help="tiny run for CI")
    ap.add_argument("--dry", action="store_true", help="load data, report counts, do not train")
    ap.add_argument(
        "--allow-rm-reward",
        action="store_true",
        help="permit the on-policy loop outside the simulator",
    )
    ap.add_argument(
        "--simulator", action="store_true", help="on-policy loop against the simulator reward"
    )
    ap.add_argument(
        "--gold-gap", type=float, default=None, help="current gold gap from the backend"
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_rl_config(args.config)
    rl = cfg.get("rl", {})
    rm = cfg.get("reward_model", {})
    reward_cfg = RewardConfig(
        shaping_weight=float(rl.get("shaping_weight", 0.3)),
        tag_penalty=float(rl.get("tag_penalty", 0.2)),
        format_penalty=float(rl.get("format_penalty", 0.5)),
        cold_until_positives=int(rm.get("cold_until_positives", 50)),
    )
    guard_cfg = GuardConfig(
        swing_rate_min=float(rl.get("guards", {}).get("swing_rate_min", 0.15)),
        entropy_floor=float(rl.get("guards", {}).get("entropy_floor", 0.6)),
        max_steps_between_outcomes=int(rl.get("guards", {}).get("max_steps_between_outcomes", 50)),
        gold_gap_threshold=float(rm.get("gold_gap_threshold", 0.15)),
    )
    rows = load_snapshot(args.snapshot)
    started = time.time()
    report: dict[str, Any] = {
        "stage": args.stage,
        "snapshot": args.snapshot,
        "rows": len(rows),
        "rows_with_prompt": sum(1 for r in rows if r.prompt),
        "tier2_positives": count_positives(rows),
        "base_model": args.base_model,
        "adapter_in": args.adapter,
        "smoke": args.smoke,
    }
    if args.dry:
        report["dry"] = True
        (out / "run.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0
    lora = rl.get("lora", {"r": 16, "alpha": 32, "dropout": 0.05})
    try:
        if args.stage == "rft":
            from outlier_trainer.rft import run_rft

            result = run_rft(
                rows,
                base_model=args.base_model,
                out_dir=out,
                lora=lora,
                epochs=float(rl.get("epochs", 2)),
                lr=float(rl.get("learning_rate", 1e-5)),
                smoke=args.smoke,
            )
        elif args.stage == "dpo":
            from outlier_trainer.dpo import run_dpo

            result = run_dpo(
                rows,
                base_model=args.base_model,
                adapter=args.adapter,
                out_dir=out,
                lora=lora,
                beta=float(rl.get("dpo_beta", 0.1)),
                lr=float(rl.get("learning_rate", 1e-5)),
                smoke=args.smoke,
            )
        elif args.stage == "grpo_offpolicy":
            from outlier_trainer.grpo_offpolicy import run_grpo_offpolicy

            result = run_grpo_offpolicy(
                rows,
                base_model=args.base_model,
                adapter=args.adapter,
                out_dir=out,
                lora=lora,
                rl=rl,
                reward_cfg=reward_cfg,
                guard_cfg=guard_cfg,
                gold_gap=args.gold_gap,
                smoke=args.smoke,
            )
        else:
            if not (args.simulator or args.allow_rm_reward):
                report["error"] = (
                    "grpo_onpolicy trains against the reward model, not real outcomes; "
                    "pass --simulator or --allow-rm-reward"
                )
                (out / "run.json").write_text(json.dumps(report, indent=2))
                print(report["error"], file=sys.stderr)
                return 2
            from outlier_trainer.grpo_onpolicy import run_grpo_onpolicy

            prompts = sorted({r.prompt for r in rows if r.prompt})

            def reward_fn(prompts_, completions, **_):
                # simulator stand-in: longer, better-formed completions score higher; the backend
                # supplies the real RM shaping through the API when not in simulator mode
                return [
                    min(1.0, len(c) / 2000.0) * (1.0 if "<idea>" in c else 0.2) for c in completions
                ]

            def combo_fn(completions):
                import re

                keys = []
                for c in completions:
                    m = re.search(r"<cards>(.*?)</cards>", c, re.S)
                    keys.append(
                        "+".join(sorted(x.strip() for x in m.group(1).split(","))) if m else "none"
                    )
                return keys

            result = run_grpo_onpolicy(
                prompts,
                base_model=args.base_model,
                adapter=args.adapter,
                out_dir=out,
                lora=lora,
                rl=rl,
                reward_fn=reward_fn,
                combo_fn=combo_fn,
                vllm_url=cfg.get("generation", {}).get("vllm_url"),
                smoke=args.smoke,
            )
        report.update(result)
        report["status"] = "completed" if not result.get("skipped") else "skipped"
    except ImportError as e:
        report.update(
            {
                "status": "failed",
                "error": f"training dependencies missing: {e}. "
                "Install with `uv sync --all-packages --extra train`.",
            }
        )
    except Exception as e:
        report.update({"status": "failed", "error": repr(e)})
    report["seconds"] = round(time.time() - started, 1)
    (out / "run.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2, default=str))
    return 0 if report.get("status") in ("completed", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
