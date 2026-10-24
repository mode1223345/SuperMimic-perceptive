"""Reject incompatible terrain semantics before loading network/optimizer state."""

import hashlib
import json
import os
from pathlib import Path


def map_metadata(terrain: dict, map_dim: int, encoder: dict | None = None) -> dict:
    payload = {"schema": 1, "terrain": terrain, "map_dim": map_dim,
               "local_shape": [2, 15, 23], "local_extent": [1.1, .7],
               "height": "terrain_z-base_z", "alignment": "gravity+yaw",
               "interpolation": "bilinear", "align_corners": True}
    if encoder is not None:
        if encoder['map_dim'] != map_dim:
            raise ValueError('AE map dimension differs from map configuration')
        payload['encoder'] = encoder
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "fingerprint": hashlib.sha256(canonical.encode()).hexdigest()}


def check_checkpoint(checkpoint: dict, expected: dict, resume: bool = False):
    """Small snapshots are valid for evaluation; optimizer resume requires last.pt."""
    saved = checkpoint.get("heightmap")
    if saved is None:
        raise ValueError("Checkpoint has no heightmap metadata; upstream checkpoints need explicit conversion.")
    if saved != expected:
        raise ValueError("Checkpoint heightmap configuration differs from this run. Use its original terrain/configuration.")
    if 'encoder' in saved:
        from .Ae import validate_shared_checkpoint
        validate_shared_checkpoint(checkpoint, saved['encoder'])
    if resume:
        required = {"pi", "wm", "critic", "opt_pi", "opt_wm", "opt_v", "args",
                    "iteration", "wall_time", "sim_steps"}
        if missing := required - checkpoint.keys():
            raise ValueError(f"Resume requires full last.pt; missing {sorted(missing)}")


def write_run_metadata(log_dir, arguments: dict, metadata: dict, resume: bool):
    """Only call after checkpoint/model loading succeeds; preserve original args."""
    directory = Path(log_dir)
    path = directory / "Heightmap.json"
    existing = path.exists()
    if existing:
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing run heightmap configuration differs; choose a new log_dir")
        if not resume:
            raise FileExistsError("Run already exists; use --resume or a new --log_dir")
    elif any((directory / name).exists() for name in ("args.json", "train_log.csv", "last.pt")):
        raise ValueError("Existing run has no valid heightmap configuration; choose a new log_dir")
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {"Resume.json" if existing else "args.json": arguments}
    if not existing:
        outputs["Heightmap.json"] = metadata
    for name, value in outputs.items():
        destination = directory / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
