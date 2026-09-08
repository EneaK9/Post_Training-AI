"""Arms of the online eval. Each system is a (backend, fake policy, approval policy) triple."""

from __future__ import annotations

from dataclasses import dataclass

from outlier_schemas.enums import BackendKind


@dataclass(frozen=True)
class System:
    name: str
    backend: BackendKind
    fake_policy: str  # only used by the fake backend
    approval: str  # top_rm | random
    adapter: str | None = None
    description: str = ""


SYSTEMS: dict[str, System] = {
    "loop_a_api": System(
        "loop_a_api",
        BackendKind.anthropic,
        "archive",
        "top_rm",
        description="API model in Loop A with the archive in the prompt",
    ),
    "loop_a_local": System(
        "loop_a_local",
        BackendKind.local,
        "archive",
        "top_rm",
        description="open model in Loop A, same prompt",
    ),
    "loop_b": System(
        "loop_b",
        BackendKind.local,
        "archive",
        "top_rm",
        adapter="loop_b",
        description="trained checkpoint, same prompt",
    ),
    "mean_rl_baseline": System(
        "mean_rl_baseline",
        BackendKind.local,
        "random",
        "random",
        adapter="mean_rl",
        description="mean-objective RL baseline",
    ),
    # simulator stand-ins so the harness runs end to end without money
    "loop_a_fake": System(
        "loop_a_fake",
        BackendKind.fake,
        "archive",
        "top_rm",
        description="archive-aware fake generator",
    ),
    "random_fake": System(
        "random_fake",
        BackendKind.fake,
        "random",
        "random",
        description="history-blind fake generator",
    ),
}
