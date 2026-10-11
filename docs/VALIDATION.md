# Validation

Run `python -m unittest discover -s tests -v` in the virtual environment.

The tests create a tiny two-layer, four-expert Mixtral-style safetensors fixture in a temporary directory.
They check retained tensor values, compact expert IDs, sliced router rows, preserved embeddings, source
checksums, rewritten config/index, read-only dry runs, invalid plans, rejected overlapping/nonempty
outputs, and the documented inspect -> router-only -> dry-run -> prune CLI sequence.

Local validation on Windows with Python 3.13, CPU PyTorch, NumPy, and safetensors passed. The workflow
runs equivalent tests with Python 3.11 on Ubuntu, Windows, and macOS. Check the current Actions result;
the presence of a workflow file does not itself prove a pass.

This does not validate real-model inference, full-model hooks, daemon forwarding/streaming, FP8 or other
quantized formats, model quality, runtime speedups, or GPU behavior. No real models are downloaded in CI.
