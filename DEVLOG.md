# MoE-Watcher-Modifier — Development Log

Complete record of every design decision, bug fix, and test result from
building and validating the toolkit.

---

## Origin

Started as a one-off pruning attempt for `Qwen/Qwen3-Coder-Next-FP8`
(80 GB, 512 experts, 48 layers) targeting a single NVIDIA L4 (22 GB VRAM).
Every vLLM / CUDA approach hit a hard wall: `fused_moe/layer.py` allocates
ALL 512 expert tensors on the GPU in `__init__` before any CPU offload kicks
in. No flag, no workaround exists in vLLM 0.19.0.

That failure prompted a broader question: can we build a tool that works with
*any* MoE checkpoint, *any* inference runtime, and *any* hardware — so
anyone wanting to prune their model doesn't have to repeat this pain?

The answer is MoE-Watcher-Modifier.

---

## Architecture Decisions

### Why safetensors-only, no model load for pruning?

The rewriter (`pruner.py`) processes checkpoints shard-by-shard directly.
It never calls `from_pretrained`. This means:
- No GPU required
- Memory bounded by the largest single shard, not the full model
- Works on a laptop for an 80 B model
- No framework version dependencies

### Why router-only stats mode?

Loading only the `gate.weight` tensors (~50 MB even for 512-expert models)
lets the tool run on CPU in minutes. The tradeoff is that random hidden states
don't reflect real usage — but it gives a structural baseline good enough
for the first prune iteration.

### Why a daemon proxy instead of hooking into the inference process?

We can't inject code into a running ollama/vLLM/llama.cpp process.
But we don't need to. Every request contains the prompt text. We:
1. Load `embed_tokens.weight` once at startup (~hundreds of MB)
2. Load all router weights once (~tens of MB)
3. Per request: tokenize → embed → forward through routers → record selections

This gives real routing signal driven by actual user prompts, with zero
intrusion into the inference server. Works with any runtime on any OS.

---

## Files Built

```
MoE-Watcher-Modifier/
├── moe_monitor.py              CLI: inspect / router-only / full-model / report / daemon
├── moe_prune.py                CLI: inspect / dry-run / prune-checkpoint / uniform-plan
├── moe_pruner/
│   ├── checkpoint.py           Safetensors I/O, load_embed_tokens()
│   ├── schema.py               Per-family tensor naming (6 schemas)
│   ├── stats.py                Stats dataclass, collect_router_only(), probe_prompt()
│   ├── planner.py              Plan dataclass, build_plan(), build_uniform_plan()
│   ├── pruner.py               Shard-by-shard checkpoint rewriter
│   ├── report.py               Rich + plain-text report rendering
│   ├── daemon.py               Transparent HTTP proxy + background stats accumulation
│   └── discovery.py            System detection, port scanning, setup wizard
├── requirements.txt
└── README.md
```

---

## Schema Fixes Discovered During Testing

### OLMoE router key

**Expected** (from HuggingFace model card / transformers source):
```
model.layers.{n}.mlp.router.weight
```

**Actual** in the `allenai/OLMoE-1B-7B-0924` checkpoint:
```
model.layers.{n}.mlp.gate.weight
```

**Fix** — updated `schema.py`:
```python
# Before
router_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.router\.weight$"

# After
router_pattern=r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$"
```

### Missing `format: pt` metadata in pruned shards

**Symptom:**
```
OSError: The safetensors archive ... does not contain the valid metadata.
Make sure you save your model with the `save_pretrained` method.
```

**Root cause:** `safetensors.torch.save_file()` accepts arbitrary metadata but
transformers' loader checks for a `"format"` key and raises if absent.
Original shards always carry `{"format": "pt"}`. The pruner was writing
`{"moe_pruner": "true", "source_shard": "..."}` without the format key.

**Fix** — `pruner.py`:
```python
metadata = {
    "format": "pt",          # required by transformers loader
    "moe_pruner": "true",
    "source_shard": str(src.root / shard_name),
}
save_file(rewritten, str(output_dir / shard_name), metadata=metadata)
```

---

## End-to-End Test: OLMoE-1B-7B

**Model:** `allenai/OLMoE-1B-7B-0924`  
**Hardware:** NVIDIA L4 (22 GB VRAM), 16 CPU cores, 62.8 GB RAM  
**Target:** 64 → 32 experts, top-k 8 → 4

### Step 1 — Download

```bash
huggingface-cli download allenai/OLMoE-1B-7B-0924 \
  --local-dir /home/dibyaprakash/models/OLMoE-1B-7B \
  --include "*.safetensors" "*.json" "*.txt" "tokenizer*"
```

Result: 3 shards, 13 GB total.

### Step 2 — Inspect

```bash
python3 moe_monitor.py inspect --model-dir /home/dibyaprakash/models/OLMoE-1B-7B
```

```json
{
  "detected_schema": "olmoe",
  "router_keys_found": 16,
  "expert_keys_found": 3072
}
```

Model: 16 layers × 64 experts = 1024 expert slots, 3 tensors each = 3072 keys.
16 router keys (one per layer). Schema auto-detected correctly after the fix.

### Step 3 — Router-only stats

```bash
python3 moe_monitor.py router-only \
  --model-dir /home/dibyaprakash/models/OLMoE-1B-7B \
  --output ./stats/olmoe-report.json \
  --keep-experts 32 --new-topk 4 --samples 512 --strategy per-layer
```

- 512 random unit-norm hidden states per router
- Total selections: 65,536
- Coverage: 100% (all 64 experts selected at least once per layer with random probes)
- Elapsed: ~5s on CPU

Top global experts by random-probe counts:

| Rank | Expert | Selections |
|---|---|---|
| 1 | 12 | 1,160 |
| 2 | 37 | 1,154 |
| 3 | 17 | 1,132 |

### Step 4 — Dry-run

```bash
python3 moe_prune.py dry-run \
  --model-dir /home/dibyaprakash/models/OLMoE-1B-7B \
  --plan ./stats/olmoe-report-plan.json
```

```
[dry-run] Plan validation passed.
[prune] model-00001-of-00003.safetensors: 1147 → 605 tensors [dry-run]
[prune] model-00002-of-00003.safetensors: 1197 → 626 tensors [dry-run]
[prune] model-00003-of-00003.safetensors: 875  → 452 tensors [dry-run]
[dry-run] Complete. 1683 output tensors.
```

### Step 5 — Prune

```bash
python3 moe_prune.py prune-checkpoint \
  --model-dir /home/dibyaprakash/models/OLMoE-1B-7B \
  --plan ./stats/olmoe-report-plan.json \
  --output-dir /home/dibyaprakash/models/OLMoE-1B-7B-pruned-32 \
  --copy-support-files
```

```
[prune] 64 → 32 experts  topk 8 → 4  strategy=per-layer
[prune] Done in 17.3s. Output: 1683 tensors, 7.39 GB
```

| Metric | Before | After |
|---|---|---|
| Experts per layer | 64 | 32 |
| Top-k | 8 | 4 |
| Total tensors | 3,219 | 1,683 |
| Checkpoint size | 13.0 GB | 7.39 GB |
| Rewrite time | — | 17.3s |
| GPU required | No | No |

### Step 6 — Verify

```bash
python3 moe_monitor.py inspect --model-dir /home/dibyaprakash/models/OLMoE-1B-7B-pruned-32
```

```json
{
  "detected_schema": "olmoe",
  "router_keys_found": 16,
  "expert_keys_found": 1536
}
```

Config updated correctly: `num_experts: 32`, `num_experts_per_tok: 4`.

### Step 7 — Load and run inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    '/home/dibyaprakash/models/OLMoE-1B-7B-pruned-32',
    dtype=torch.bfloat16, device_map='auto'
)
```

**Result: PASS.** Model loads on `cuda:0`, generates tokens without errors.

**Generation quality:** Degraded — repetitive / incoherent output (expected).
OLMoE was trained assuming all 64 experts exist. Removing 32 without finetuning
breaks the routing distribution. Router-only stats with random probes also don't
reflect real usage patterns.

This is the documented limitation in README.md. The fix is the iterative daemon
workflow described below.

---

## Daemon — Design and Implementation

### Components

**`moe_pruner/discovery.py`**
- `detect_system()` — reads OS, CPU count, RAM from `/proc/meminfo`, GPU info
  from `torch.cuda` (NVIDIA), `rocm-smi` (AMD), or `torch.backends.mps` (Apple)
- `discover_servers()` — probes 10 well-known ports (11434 ollama, 8000 vLLM,
  8080 llama.cpp, 1234 LM Studio, etc.) with a 0.5s TCP timeout per port
- `run_setup_wizard()` — interactive 3-step setup: system info → checkpoint
  validation → backend selection. Re-prompts on bad paths instead of crashing.

**`moe_pruner/daemon.py`**
- `DaemonState` — holds loaded weights and accumulated `Stats`. Lock protects
  stats writes from the background thread vs reads for reports.
- `make_proxy_handler()` — stdlib `BaseHTTPRequestHandler` subclass. Forwards
  ALL HTTP methods to backend immediately, then enqueues prompt for analysis.
  Zero added latency on the response path.
- `_extract_prompts()` — handles OpenAI chat (`messages[].content`), OpenAI
  completions (`prompt`), ollama native (`prompt`), and vision content arrays.
- Background worker thread drains the queue and calls `probe_prompt()`.
- `SIGUSR1` / `SIGHUP` force an immediate report (works on Linux/macOS).

**`moe_pruner/stats.py` — `probe_prompt()`**
- Given real token IDs, looks up embeddings from `embed_tokens.weight`
- Forwards each token's hidden state through every router weight
- Records per-token expert selections per layer
- Returns a `Stats` object that gets accumulated into the running total

### Startup behavior

```
[1/3] Detecting system...         ← detect_system() + print_system_info()
[2/3] Checkpoint directory...     ← validate path, warn if no .safetensors found
[3/3] Inference backend...        ← discover_servers() → pick from table
                                     or enter manually if nothing found
──────────── Setup complete ────────────
[daemon] Proxy listening on 0.0.0.0:8080
[daemon] Loading checkpoint: /path/to/checkpoint   ← in background thread
[daemon] Ready. Monitoring active.
```

The proxy starts accepting connections immediately while the loader thread
runs in the background. Requests before loading is complete are still
forwarded — they just don't contribute stats yet.

### Live test on this machine

```bash
python3 moe_monitor.py daemon \
  --model-dir /path/to/checkpoint \
  --keep-experts 128 --new-topk 4
```

Auto-discovered:
```
ollama  http://localhost:11434  [aecode-qwen3:latest, qwen3-coder:latest]
```
Only one server found → selected automatically, proxy started.

---

## Known Limitations

1. **Quality after pruning** — always degraded without finetuning. 50% expert
   reduction (64→32) with random-probe stats produces repetitive output.
   Recommendation: use daemon to collect real stats, then prune, then finetune.

2. **Router-only stats are approximate** — random hidden states don't match
   real input distributions. Use `full-model` mode or daemon for production
   quality rankings.

3. **Daemon tokenizer fallback** — if `transformers` is not installed, the
   daemon falls back to `ord(char) % vocab_size` per character. Stats will
   be very approximate. Install transformers for accurate routing analysis.

4. **Single-shard index** — the pruner writes the same shard filenames as the
   source. For very large models the per-shard tensor count is halved but
   shard file sizes shrink proportionally. No rebalancing is done.

5. **FP8 checkpoints** — pruned FP8 tensors are written as-is. If the inference
   runtime re-quantizes on load, results may differ from expectation.

---

## Recommended Workflow for Production Quality

```
1.  router-only on original checkpoint  →  initial plan (fast, no GPU)
2.  prune-checkpoint                    →  smaller checkpoint
3.  Start inference server with pruned checkpoint
4.  Start daemon in front of it
5.  Use the model normally for hours/days
6.  Daemon auto-reports every N prompts  →  real routing stats
7.  Run printed prune command on original checkpoint
8.  Finetune / distill on domain data   →  quality recovery
9.  Repeat from step 3 if needed
```

---

## Supported Model Families

| Schema | Models | Router key pattern |
|---|---|---|
| `qwen3_next` | Qwen3-Coder-Next, Qwen3-Next | `mlp.gate.weight` |
| `qwen_moe` | Qwen1.5-MoE, Qwen2-MoE | `mlp.gate.weight` |
| `mixtral` | Mixtral-8x7B, 8x22B | `block_sparse_moe.gate.weight` |
| `deepseek` | DeepSeek-V2/V3/R1 | `mlp.gate.weight` |
| `phi3_moe` | Phi-3.5-MoE | `block_sparse_moe.gate` |
| `olmoe` | OLMoE (all variants) | `mlp.gate.weight` |
