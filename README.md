# MoE-Watcher-Modifier

Model-agnostic Mixture-of-Experts analysis and pruning toolkit.

Analyses which experts a MoE model actually uses, ranks them by importance,
and rewrites the checkpoint with fewer experts — reducing model size without
modifying training or inference code.

Works with any safetensors MoE checkpoint. No specific model family, serving
framework, or hardware required.

---

## Supported Model Families

| Schema | Models |
|---|---|
| `qwen3_next` | Qwen3-Coder-Next, Qwen3-Next |
| `qwen_moe` | Qwen1.5-MoE, Qwen2-MoE |
| `mixtral` | Mixtral-8x7B, Mixtral-8x22B |
| `deepseek` | DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 |
| `phi3_moe` | Phi-3.5-MoE |
| `olmoe` | OLMoE |

Schema is auto-detected from `config.json`. Pass `--schema` to override.

---

## Requirements

```
torch>=2.0.0
safetensors>=0.4.0
rich>=13.0.0          # optional, nicer output
transformers>=4.40.0  # optional, full-model mode only
```

Install:
```bash
pip install torch safetensors rich
pip install transformers  # only if using full-model mode
```

---

## Two Tools

| Tool | Purpose |
|---|---|
| `moe_monitor.py` | Analyse expert usage, rank experts, generate pruning plan |
| `moe_prune.py` | Rewrite checkpoint using a pruning plan |

`moe_monitor.py` subcommands:

| Subcommand | When to use |
|---|---|
| `inspect` | See checkpoint layout, auto-detect schema |
| `router-only` | Quick stats from random probes — no GPU, no full model load |
| `full-model` | Best stats from real prompts — needs RAM/VRAM to load model |
| `report` | Re-rank a saved report for a different keep count (instant) |
| `daemon` | **Live monitoring** — transparent proxy in front of any server |

---

## Workflow

### Step 1 — Inspect the checkpoint

```bash
python3 moe_monitor.py inspect --model-dir /path/to/checkpoint
```

Output: JSON summary of layers, experts, detected schema, sample tensor keys.

---

### Step 2 — Collect expert usage statistics

**Router-only mode** (recommended first pass):

Loads only the router weight tensors (~MB). Probes each router with random
unit-norm hidden states. No GPU required. Completes in minutes even for large
checkpoints.

```bash
python3 moe_monitor.py router-only \
  --model-dir /path/to/checkpoint \
  --output ./stats/report.json \
  --keep-experts 128 \
  --new-topk 4 \
  --samples 1024 \
  --strategy per-layer
```

**Full-model mode** (best quality stats):

Loads the full model, installs forward hooks on every MoE gate, runs your
real prompts through. Captures true routing decisions on your actual workload.
Requires enough RAM/VRAM to load the model.

```bash
python3 moe_monitor.py full-model \
  --model-dir /path/to/checkpoint \
  --prompts ./prompts.jsonl \
  --output ./stats/report.json \
  --keep-experts 128 \
  --new-topk 4
```

Prompts file format — JSONL with a `prompt` field, or plain text (one prompt per line):

```jsonl
{"prompt": "Write a Python function to merge two sorted lists."}
{"prompt": "Debug this Go code: ..."}
```

---

### Step 3 — Review the report

The monitor prints a ranked table of expert usage and the exact prune command
to run. It also writes:

- `./stats/report.json` — full report with per-layer stats and counts
- `./stats/report-plan.json` — pruning plan ready for `moe_prune.py`

To re-rank with a different keep count without re-probing:

```bash
python3 moe_monitor.py report \
  --input ./stats/report.json \
  --keep-experts 64
```

---

### Step 4 — Dry-run the rewrite

Validates the plan against the checkpoint structure without writing any files.

```bash
python3 moe_prune.py dry-run \
  --model-dir /path/to/checkpoint \
  --plan ./stats/report-plan.json
```

---

### Step 5 — Rewrite the checkpoint

```bash
python3 moe_prune.py prune-checkpoint \
  --model-dir /path/to/checkpoint \
  --plan ./stats/report-plan.json \
  --output-dir ./pruned-128experts \
  --copy-support-files
```

`--copy-support-files` copies the tokenizer, generation config, and other
auxiliary files into the output directory so it is a complete, loadable checkpoint.

The rewriter:
- Drops expert tensors not in the keep list
- Renames retained experts to compact IDs `0..target-1`
- Slices each router weight matrix to match the retained expert IDs
- Updates `config.json` with the new expert count and top-k
- Writes `pruning_manifest.json` for traceability

No GPU required. Memory usage is bounded by the largest shard size.

---

### Step 6 — Verify the output

```bash
python3 moe_monitor.py inspect --model-dir ./pruned-128experts
```

---

## Daemon — Live Traffic Monitoring

The best expert rankings come from your real workload. The daemon is a
transparent HTTP proxy that sits in front of any model server and silently
accumulates routing statistics from live user traffic.

**Supports any runtime:** ollama, vLLM, llama.cpp server, LM Studio,
text-generation-webui, or anything with an OpenAI-compatible API.

**Works on any hardware:** Linux, macOS, Windows, AMD, NVIDIA, CPU-only.
The daemon loads only the router weights (~MB) and the embedding table —
not the full model.

### Start the daemon

```bash
python3 moe_monitor.py daemon \
  --model-dir /path/to/checkpoint \
  --backend http://localhost:11434 \
  --listen-port 8080 \
  --keep-experts 128 \
  --new-topk 4 \
  --output ./stats/live-report.json \
  --report-every 50
```

Then point your app at `http://localhost:8080` instead of the backend.
All requests are forwarded transparently — no latency added to the response path.

### How it works

```
Your app  →  :8080 (daemon)  →  :11434 (ollama / vLLM / llama.cpp / ...)
                  │
                  └─ background thread:
                       1. Extract prompt text from request body
                       2. Tokenize with checkpoint's own tokenizer
                       3. Look up token embeddings from embed_tokens.weight
                       4. Forward each token's hidden state through router weights
                       5. Record selected experts per layer
                       6. Every N requests: write report + print prune command
```

### Force an immediate report

```bash
# Print ranked expert table and ready-to-run prune command right now:
kill -USR1 $(pgrep -f "moe_monitor.py daemon")
```

### Daemon workflow (recommended for production quality)

```
1. Start daemon in front of your model server
2. Use the model normally for a few hours / days
3. After N prompts, daemon auto-prints ranked report + prune command
4. Run the printed moe_prune.py command
5. Load pruned checkpoint in your server
6. Repeat if needed — each iteration uses real routing signal
7. Finetune / distill on domain data  →  full quality recovery
```

---

### Step 6 — Verify the output

```bash
python3 moe_monitor.py inspect --model-dir ./pruned-128experts
```

---

## Iterative Workflow

For best results, iterate:

```
1. router-only on original checkpoint  →  initial plan (structural preferences)
2. prune-checkpoint                    →  smaller checkpoint
3. Load pruned checkpoint (fits in memory now)
4. full-model on pruned checkpoint     →  real routing stats
5. report --keep-experts N             →  refined plan
6. prune-checkpoint again              →  final checkpoint
7. Finetune / distill on domain data   →  quality recovery
```

---

## Output Files

### Report JSON (`report.json`)

```json
{
  "schema": "qwen3_next",
  "mode": "router-only",
  "source_num_experts": 512,
  "target_num_experts": 128,
  "total_router_selections": 491520,
  "expert_coverage_fraction": 0.993,
  "global_expert_ranking": [316, 290, 197, ...],
  "layer_stats": [
    {
      "layer": 0,
      "active_experts": 508,
      "top5_experts": [262, 81, 116, 52, 132],
      "keep_ids": [5, 8, 12, ...]
    }
  ],
  "plan": { ... }
}
```

### Plan JSON (`report-plan.json`)

```json
{
  "source_num_experts": 512,
  "target_num_experts": 128,
  "source_topk": 10,
  "target_topk": 4,
  "strategy": "per-layer",
  "layer_keep_ids": [[5, 8, 12, ...], ...]
}
```

### Pruning Manifest (`pruning_manifest.json`)

Written into the output checkpoint directory:

```json
{
  "source_dir": "/path/to/original",
  "output_dir": "./pruned-128experts",
  "source_num_experts": 512,
  "target_num_experts": 128,
  "strategy": "per-layer",
  "elapsed_seconds": 312.4,
  "total_size_bytes": 22300000000
}
```

---

## Compatibility

The pruned checkpoint is standard safetensors format and can be loaded by:

| Runtime | Compatible | Notes |
|---|---|---|
| `transformers` | ✅ | Load with `AutoModelForCausalLM.from_pretrained` |
| `vLLM` | ✅ | `vllm serve ./pruned-128experts` |
| `llama.cpp` | ⚠️ | Needs conversion to GGUF first (see below) |
| `ollama` | ⚠️ | Needs GGUF conversion first |

### Convert to GGUF for llama.cpp / ollama

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp --depth=1
cd llama.cpp && pip install -r requirements.txt

# Convert the pruned safetensors checkpoint to GGUF
python3 convert_hf_to_gguf.py /path/to/pruned-128experts \
  --outfile ./pruned-128experts-q4.gguf \
  --outtype q4_k_m

# Load with llama-server
./llama-server --model ./pruned-128experts-q4.gguf --port 8000
```

---

## Important Limitations

**Pruning alone degrades quality.** The model was trained assuming all N experts
exist. Removing experts without finetuning produces incoherent output. Always
finetune or distill after pruning.

**Router-only stats are approximate.** Random hidden states do not reflect your
real input distribution. Use full-model mode after an initial prune for better stats.

**Aggressive pruning (>75% reduction) risks severe quality loss.** Recommended
targets:
- `512 → 256` — minimal risk
- `512 → 128` — noticeable degradation, recoverable with finetuning
- `512 → 64`  — significant degradation, heavy finetuning needed

---

## Project Structure

```
MoE-Watcher-Modifier/
├── moe_monitor.py          # CLI: expert usage analysis, plan generation, daemon
├── moe_prune.py            # CLI: checkpoint rewriting
├── moe_pruner/
│   ├── __init__.py
│   ├── checkpoint.py       # Safetensors checkpoint I/O + embed_tokens loader
│   ├── schema.py           # MoE tensor naming conventions per model family
│   ├── stats.py            # Expert usage statistics + prompt-driven probing
│   ├── planner.py          # Pruning plan generation from stats
│   ├── pruner.py           # Checkpoint rewriter
│   ├── report.py           # Terminal and JSON report generation
│   └── daemon.py           # Transparent proxy daemon for live traffic monitoring
├── requirements.txt
└── README.md
```
# MoE-Watcher-Modifier
# MoE-Watcher-Modifier
