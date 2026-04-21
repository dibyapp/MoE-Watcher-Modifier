"""
Expert usage statistics collection.

Two modes:
  router-only  Load only router weight tensors (~MB), probe with random hidden
               states. No full model load, no GPU required.

  full-model   Load the full model via transformers, install forward hooks on
               every MoE gate, run real prompts. Requires enough RAM/VRAM.

Both modes produce the same Stats object, compatible with the planner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Stats:
    """
    Expert usage statistics across all layers.

    counts[layer][expert]    — number of times this expert was selected
    prob_sums[layer][expert] — sum of routing probabilities assigned to this expert
    num_layers: int
    num_experts: int
    topk: int
    mode: str                — 'router-only' or 'full-model'
    num_samples: int         — random samples (router-only) or prompt count (full-model)
    elapsed_seconds: float
    """

    counts: torch.Tensor       # [num_layers, num_experts] float64
    prob_sums: torch.Tensor    # [num_layers, num_experts] float64
    num_layers: int
    num_experts: int
    topk: int
    mode: str
    num_samples: int
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "topk": self.topk,
            "mode": self.mode,
            "num_samples": self.num_samples,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "counts": self.counts.tolist(),
            "prob_sums": self.prob_sums.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Stats":
        return cls(
            counts=torch.tensor(d["counts"], dtype=torch.float64),
            prob_sums=torch.tensor(d["prob_sums"], dtype=torch.float64),
            num_layers=d["num_layers"],
            num_experts=d["num_experts"],
            topk=d["topk"],
            mode=d["mode"],
            num_samples=d["num_samples"],
            elapsed_seconds=d.get("elapsed_seconds", 0.0),
        )

    @property
    def total_selections(self) -> int:
        return int(self.counts.sum().item())

    def accumulate(self, other: "Stats") -> None:
        """Add another Stats object into this one in-place."""
        self.counts += other.counts
        self.prob_sums += other.prob_sums
        self.num_samples += other.num_samples

    @property
    def coverage_fraction(self) -> float:
        """Fraction of (layer, expert) slots selected at least once."""
        return float((self.counts > 0).float().mean().item())

    def global_rank(self) -> list[int]:
        """Expert ids sorted by total selection count across all layers, descending."""
        return torch.argsort(self.counts.sum(dim=0), descending=True).tolist()


# ---------------------------------------------------------------------------
# Router-only collection
# ---------------------------------------------------------------------------

def collect_router_only(
    checkpoint,
    schema,
    num_experts: int,
    num_layers: int,
    topk: int,
    hidden_size: int,
    num_samples: int = 512,
    verbose: bool = True,
    progress_callback=None,
) -> Stats:
    """
    Probe each layer's router with random unit-norm hidden states.

    Loads only the router (gate) weight tensors from the checkpoint — typically
    a few MB total. No GPU required; all computation is on CPU.

    Args:
        checkpoint: Checkpoint instance.
        schema: MoESchema describing router key patterns.
        num_experts: Total experts per layer.
        num_layers: Number of transformer layers.
        topk: Top-k routing value from the original config.
        hidden_size: Model hidden dimension.
        num_samples: Number of random hidden states per router. Higher = more
                     stable ranking. 512 is sufficient for most cases.
        verbose: Print progress to stdout.
        progress_callback: Optional callable(layer_idx, total_layers) for UIs.
    """
    t0 = time.time()

    # Find all router keys
    router_keys = [k for k in checkpoint.all_keys if schema.match_router(k)]
    if not router_keys:
        raise ValueError(
            f"No router keys found in checkpoint using schema '{schema.name}'. "
            "Check --schema or verify the checkpoint structure with 'inspect'."
        )

    if verbose:
        print(f"[stats] Loading {len(router_keys)} router tensors...")

    router_tensors = checkpoint.load_keys(router_keys)

    # Parse layer index from each key
    routers: dict[int, torch.Tensor] = {}
    for key, tensor in router_tensors.items():
        m = schema.match_router(key)
        if m:
            routers[int(m.group("layer"))] = tensor.float()

    if verbose:
        print(f"[stats] Loaded {len(routers)} routers. Probing with {num_samples} samples each...")

    counts = torch.zeros(num_layers, num_experts, dtype=torch.float64)
    prob_sums = torch.zeros(num_layers, num_experts, dtype=torch.float64)

    for i, layer_idx in enumerate(sorted(routers.keys())):
        router_weight = routers[layer_idx]  # [num_experts, hidden_size]

        hidden = torch.randn(num_samples, hidden_size)
        hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            logits = hidden @ router_weight.T
            probs = torch.softmax(logits, dim=-1)
            topk_r = torch.topk(probs, k=min(topk, num_experts), dim=-1)

        flat_ids = topk_r.indices.reshape(-1)
        flat_probs = topk_r.values.reshape(-1).to(torch.float64)
        ones = torch.ones(flat_ids.shape[0], dtype=torch.float64)

        counts[layer_idx].scatter_add_(0, flat_ids, ones)
        prob_sums[layer_idx].scatter_add_(0, flat_ids, flat_probs)

        if progress_callback:
            progress_callback(i + 1, len(routers))
        elif verbose and (i + 1) % 8 == 0:
            print(f"[stats]   layer {i+1}/{len(routers)}")

    elapsed = time.time() - t0
    if verbose:
        print(f"[stats] Done in {elapsed:.1f}s. Total selections: {int(counts.sum()):,}")

    return Stats(
        counts=counts,
        prob_sums=prob_sums,
        num_layers=num_layers,
        num_experts=num_experts,
        topk=topk,
        mode="router-only",
        num_samples=num_samples,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Prompt-driven router probing (daemon mode)
# ---------------------------------------------------------------------------

def probe_prompt(
    tokens: list[int],
    embed_weight: "torch.Tensor",
    routers: "dict[int, torch.Tensor]",
    topk: int,
    num_experts: int,
    num_layers: int,
) -> Stats:
    """
    Simulate routing for a single prompt using real token embeddings.

    This is the core of daemon mode: given actual token IDs from a real user
    request, look up their embeddings from embed_tokens.weight, mean-pool to
    a single hidden state, forward through each layer's router weight, and
    record which experts would be selected.

    No model process needed — just the checkpoint weights loaded at startup.

    Args:
        tokens: Token IDs for the prompt.
        embed_weight: embed_tokens.weight tensor [vocab, hidden].
        routers: {layer_idx: router_weight [num_experts, hidden]}.
        topk: Number of experts selected per token.
        num_experts: Total experts per layer.
        num_layers: Number of MoE layers.

    Returns:
        Stats with counts/prob_sums incremented for this prompt.
    """
    if not tokens:
        return Stats(
            counts=torch.zeros(num_layers, num_experts, dtype=torch.float64),
            prob_sums=torch.zeros(num_layers, num_experts, dtype=torch.float64),
            num_layers=num_layers,
            num_experts=num_experts,
            topk=topk,
            mode="daemon",
            num_samples=0,
        )

    counts = torch.zeros(num_layers, num_experts, dtype=torch.float64)
    prob_sums = torch.zeros(num_layers, num_experts, dtype=torch.float64)

    token_ids = torch.tensor(tokens, dtype=torch.long)
    # Clamp to valid vocab range
    token_ids = token_ids.clamp(0, embed_weight.shape[0] - 1)

    with torch.no_grad():
        # [seq_len, hidden]
        hidden_seq = embed_weight[token_ids].float()
        # Use per-token routing (more accurate than mean-pooled)
        # Each token produces its own routing decision
        for layer_idx, router_weight in routers.items():
            if layer_idx >= num_layers:
                continue
            logits = hidden_seq @ router_weight.T  # [seq_len, num_experts]
            probs = torch.softmax(logits, dim=-1)
            topk_r = torch.topk(probs, k=min(topk, num_experts), dim=-1)
            flat_ids = topk_r.indices.reshape(-1)
            flat_probs = topk_r.values.reshape(-1).to(torch.float64)
            ones = torch.ones(flat_ids.shape[0], dtype=torch.float64)
            counts[layer_idx].scatter_add_(0, flat_ids, ones)
            prob_sums[layer_idx].scatter_add_(0, flat_ids, flat_probs)

    return Stats(
        counts=counts,
        prob_sums=prob_sums,
        num_layers=num_layers,
        num_experts=num_experts,
        topk=topk,
        mode="daemon",
        num_samples=1,
    )


# ---------------------------------------------------------------------------
# Full-model collection (transformers hooks)
# ---------------------------------------------------------------------------

def collect_full_model(
    checkpoint,
    schema,
    prompts: list[str],
    num_experts: int,
    num_layers: int,
    topk: int,
    device_map: str = "auto",
    dtype_str: str = "bfloat16",
    max_length: int = 2048,
    verbose: bool = True,
) -> Stats:
    """
    Load the full model via transformers and collect real routing statistics
    by installing forward hooks on every MoE gate layer.

    Requires enough RAM/VRAM to load the model. Will fail on memory-constrained
    machines for large checkpoints — use router-only mode in that case.
    """
    import re as _re
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dtype = getattr(torch, dtype_str)

    if verbose:
        print(f"[stats] Loading model from {checkpoint.root} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint.root), trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint.root),
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=dtype,
    )
    model.eval()

    counts = torch.zeros(num_layers, num_experts, dtype=torch.float64)
    prob_sums = torch.zeros(num_layers, num_experts, dtype=torch.float64)

    # Build a pattern to find gate modules by name
    # Works for any schema: extract the part before .weight in the router_pattern
    gate_module_pattern = schema.router_pattern.rstrip("$").removesuffix(r"\.weight")

    hooks = []
    for name, module in model.named_modules():
        m = _re.match(gate_module_pattern, name)
        if not m:
            continue
        layer_idx = int(m.group("layer"))

        def make_hook(lidx):
            def hook(_module, _input, output):
                logits = output[0] if isinstance(output, tuple) else output
                probs = torch.softmax(logits.float().cpu(), dim=-1)
                topk_r = torch.topk(probs, k=min(topk, probs.shape[-1]), dim=-1)
                flat_ids = topk_r.indices.reshape(-1)
                flat_probs = topk_r.values.reshape(-1).to(torch.float64)
                counts[lidx].scatter_add_(0, flat_ids, torch.ones(flat_ids.shape[0], dtype=torch.float64))
                prob_sums[lidx].scatter_add_(0, flat_ids, flat_probs)
            return hook

        hooks.append(module.register_forward_hook(make_hook(layer_idx)))

    if verbose:
        print(f"[stats] Installed hooks on {len(hooks)} gate modules. Running {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc, use_cache=False)
        if verbose:
            print(f"[stats]   prompt {i+1}/{len(prompts)}")

    for h in hooks:
        h.remove()

    elapsed = time.time() - t0
    if verbose:
        print(f"[stats] Done in {elapsed:.1f}s.")

    return Stats(
        counts=counts,
        prob_sums=prob_sums,
        num_layers=num_layers,
        num_experts=num_experts,
        topk=topk,
        mode="full-model",
        num_samples=len(prompts),
        elapsed_seconds=elapsed,
    )
