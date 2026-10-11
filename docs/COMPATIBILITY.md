# MoE checkpoint compatibility and evaluation

## Inspect before pruning

All built-in schemas live in [schema.py](../moe_pruner/schema.py). These are naming patterns, not
tests of every named model. Supply `config.json` and either `model.safetensors` or a safetensors shard index.

| Schema | Router suffix after layer | Expert namespace | Expert count config |
| --- | --- | --- | --- |
| mixtral | `block_sparse_moe.gate.weight` | `block_sparse_moe.experts.<id>.` | `num_local_experts` |
| qwen3_next / qwen_moe / olmoe | `mlp.gate.weight` | `mlp.experts.<id>.` | `num_experts` |
| deepseek | `mlp.gate.weight` | `mlp.experts.<id>.` | `n_routed_experts` |
| phi3_moe | `block_sparse_moe.gate.` prefix | `block_sparse_moe.experts.<id>.` | `num_local_experts` |

Layer keys start with `model.layers.<layer>.`. Router tensors must be 2D, with rows indexed by expert.
Each routed layer needs separately named expert IDs covering the configured expert count. The plan
must match the source layer count, expert count, and top-k. Packed experts are rejected when there is
no matching per-expert namespace. Sparse/dense layer placement and quantization metadata need inspection.

Everything outside matched routers and experts is copied unchanged. This is not sufficient for arbitrary
quantized checkpoints, router biases, expert correction tensors, nested text configs, or fused kernels.
Do not infer runtime compatibility from a successful rewrite or schema detection.

## What the monitor measures

Random-router probing uses synthetic hidden states. The daemon reuses token embeddings at every layer;
actual inference uses transformed hidden states. Neither is actual backend routing telemetry.
Full-model hooks derive top-k counts from matching gate outputs using generic softmax scoring. Compare
this with the model's implementation: some architectures use additional routing rules. Check that every
expected gate is hooked and that reports contain samples before using an importance plan.

## A reproducible evaluation to contribute

1. Record the public model identifier/revision, license, tool commit, dependency versions, and matched layout.
2. Keep the original checkpoint and use a separate empty output directory.
3. Use representative calibration prompts and a separate held-out evaluation set.
4. Compare task scores and generated outputs before/after with identical decoding settings and seeds.
5. Measure peak memory, disk size, and latency in the intended runtime; report failures too.
6. Disclose pruning strategy, retained expert count, top-k, and any recovery training.

Use public or synthetic data in issues. Do not share private prompts, filesystem paths, credentials, or weights.
There is no tested universal compression ratio or guaranteed quality-preserving expert count.

## Primary references

- [PyTorch platform installation](https://pytorch.org/get-started/locally/)
- [Safetensors format](https://github.com/huggingface/safetensors)
- [Transformers Mixtral documentation](https://huggingface.co/docs/transformers/model_doc/mixtral)
- [Custom model code and trust_remote_code](https://huggingface.co/docs/transformers/custom_models)
