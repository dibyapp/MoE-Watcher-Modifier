"""
Checkpoint inspection and tensor I/O.

Supports any safetensors checkpoint with a model.safetensors.index.json.
No assumptions about model family, architecture, or framework.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


class Checkpoint:
    """
    Represents a safetensors checkpoint directory.

    Works with any model that stores weights as safetensors shards with
    a model.safetensors.index.json weight map. Single-file checkpoints
    (model.safetensors without an index) are also supported.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.root = Path(model_dir).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.root}")

        config_path = self.root / "config.json"
        self.config: dict[str, Any] = read_json(config_path) if config_path.exists() else {}

        index_path = self.root / "model.safetensors.index.json"
        single_path = self.root / "model.safetensors"

        if index_path.exists():
            self._index = read_json(index_path)
            self.weight_map: dict[str, str] = self._index["weight_map"]
            self.sharded = True
        elif single_path.exists():
            # Single-file checkpoint — synthesize a weight map
            from safetensors import safe_open
            with safe_open(str(single_path), framework="pt", device="cpu") as f:
                keys = list(f.keys())
            self.weight_map = {k: "model.safetensors" for k in keys}
            self._index = {"weight_map": self.weight_map}
            self.sharded = False
        else:
            raise FileNotFoundError(
                f"No safetensors checkpoint found in {self.root}. "
                "Expected model.safetensors.index.json or model.safetensors."
            )

    @property
    def shard_names(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    @property
    def all_keys(self) -> list[str]:
        return list(self.weight_map.keys())

    def keys_in_shard(self, shard_name: str) -> list[str]:
        return [k for k, v in self.weight_map.items() if v == shard_name]

    def load_shard(self, shard_name: str) -> dict[str, Any]:
        """Load a single shard. Returns a dict of key -> tensor."""
        import torch
        from safetensors.torch import load_file
        return load_file(str(self.root / shard_name))

    def iter_shards(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (shard_name, tensors) for each shard."""
        for shard_name in self.shard_names:
            yield shard_name, self.load_shard(shard_name)

    def load_keys(self, keys: list[str]) -> dict[str, Any]:
        """Load only the specified keys, reading the minimum shards needed."""
        import torch
        from safetensors.torch import load_file

        shard_to_keys: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            if key not in self.weight_map:
                raise KeyError(f"Key not found in checkpoint: {key}")
            shard_to_keys[self.weight_map[key]].append(key)

        result: dict[str, Any] = {}
        for shard_name, shard_keys in shard_to_keys.items():
            tensors = load_file(str(self.root / shard_name))
            for k in shard_keys:
                result[k] = tensors[k]
        return result

    def find_keys(self, pattern: str) -> list[str]:
        """Find all weight keys matching a regex pattern."""
        rx = re.compile(pattern)
        return [k for k in self.weight_map if rx.search(k)]

    def load_embed_tokens(self) -> Any:
        """
        Load the token embedding table (embed_tokens.weight) from the checkpoint.

        Used by the daemon to convert token IDs to hidden states so that real
        prompt text drives the router probing — no GPU needed, runs on CPU.
        """
        import torch

        candidates = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "gpt_neox.embed_in.weight",
            "model.embed_tokens_mtp.weight",
        ]
        for key in candidates:
            if key in self.weight_map:
                return self.load_keys([key])[key].float()

        # Fallback: find any key that ends with embed_tokens.weight
        for key in self.all_keys:
            if key.endswith("embed_tokens.weight"):
                return self.load_keys([key])[key].float()

        raise KeyError(
            "Could not find embed_tokens weight in checkpoint. "
            "Checked: " + ", ".join(candidates)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "config": {
                k: self.config.get(k)
                for k in (
                    "model_type", "architectures", "num_hidden_layers",
                    "hidden_size", "num_experts", "num_experts_per_tok",
                    "moe_intermediate_size",
                )
                if k in self.config
            },
            "shards": len(self.shard_names),
            "total_keys": len(self.weight_map),
        }
