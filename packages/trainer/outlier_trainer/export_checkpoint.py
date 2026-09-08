"""Copy a LoRA adapter directory into object storage so vLLM (and the backend) can load it."""

from __future__ import annotations

from pathlib import Path


def export_adapter(adapter_dir: str | Path, put) -> list[str]:
    adapter_dir = Path(adapter_dir)
    uris: list[str] = []
    for f in sorted(adapter_dir.rglob("*")):
        if f.is_file():
            key = f"checkpoints/{adapter_dir.name}/{f.relative_to(adapter_dir)}"
            uris.append(put(key, f.read_bytes(), "application/octet-stream"))
    return uris
