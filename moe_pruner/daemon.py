"""
Transparent proxy daemon for live MoE expert usage monitoring.

Sits in front of any OpenAI-compatible model server (ollama, vLLM, llama.cpp,
LM Studio, text-generation-webui, etc.) and silently accumulates expert usage
statistics as users interact with the model normally.

How it works:
  1. At startup, loads ONLY the lightweight parts of the checkpoint:
       - embed_tokens.weight  (maps token IDs → hidden states)
       - All router (gate) weight tensors (~MB total even for 80B models)
     This is fast (<30s) and uses minimal RAM. The full model is NOT loaded
     here — it runs separately in your existing inference server.

  2. For every incoming request, the daemon:
       a. Forwards the full request to the backend server immediately
       b. In a background thread, extracts the prompt text
       c. Tokenizes it with the checkpoint's tokenizer
       d. Looks up token embeddings from embed_tokens.weight
       e. Runs each token's hidden state through every router weight
       f. Records which experts would be selected
     The background analysis never touches the main request path.

  3. Every N requests (configurable, default 100), or on SIGUSR1 / SIGHUP,
     the daemon:
       - Writes updated stats to stats/live-report.json
       - Prints a ranked expert table to stdout
       - Prints the exact moe_prune.py command to prune with current stats

Usage:
  python3 moe_monitor.py daemon \\
    --model-dir /path/to/checkpoint \\
    --backend http://localhost:11434 \\
    --listen-port 8080 \\
    --keep-experts 128 \\
    --new-topk 4 \\
    --report-every 100 \\
    --output ./stats/live-report.json

Then point your application at http://localhost:8080 instead of the backend.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError


class DaemonState:
    """
    Shared state for the daemon: loaded weights and accumulated stats.

    All stats writes happen in the background thread; stats reads (for report)
    happen under lock. The proxy handler itself never blocks on stats.
    """

    def __init__(
        self,
        model_dir: str,
        schema_name: str | None,
        keep_experts: int,
        new_topk: int | None,
        report_every: int,
        output_path: Path,
        backend_url: str,
        strategy: str = "per-layer",
    ) -> None:
        self.model_dir = model_dir
        self.schema_name = schema_name
        self.keep_experts = keep_experts
        self.new_topk = new_topk
        self.report_every = report_every
        self.output_path = output_path
        self.backend_url = backend_url.rstrip("/")
        self.strategy = strategy

        # Loaded at startup
        self.embed_weight = None       # [vocab, hidden] float32 on CPU
        self.routers: dict[int, Any] = {}  # {layer_idx: tensor [num_experts, hidden]}
        self.tokenizer = None
        self.schema = None
        self.num_experts = 0
        self.num_layers = 0
        self.topk = 0
        self.hidden_size = 0

        # Live stats — protected by lock
        self._lock = threading.Lock()
        self._stats = None  # Stats object, created after load
        self._request_count = 0
        self._analyzed_count = 0

        # Background work queue: list of prompt strings
        self._queue: list[str] = []
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()

        self._loaded = False
        self._load_error: str | None = None

    def load(self) -> None:
        """Load checkpoint weights. Called once at startup in a thread."""
        import torch
        from moe_pruner.checkpoint import Checkpoint
        from moe_pruner.schema import get_schema

        t0 = time.time()
        print(f"[daemon] Loading checkpoint: {self.model_dir}")

        ckpt = Checkpoint(self.model_dir)
        self.schema = get_schema(self.schema_name, ckpt)

        cfg = ckpt.config
        self.num_layers = cfg.get(self.schema.num_layers_config_key, 0)
        self.num_experts = cfg.get(self.schema.num_experts_config_key, 0)
        self.topk = cfg.get(self.schema.num_experts_per_tok_config_key, 0)
        self.hidden_size = cfg.get(self.schema.hidden_size_config_key, 0)

        for name, val in [
            ("num_layers", self.num_layers),
            ("num_experts", self.num_experts),
            ("topk", self.topk),
            ("hidden_size", self.hidden_size),
        ]:
            if not val:
                self._load_error = (
                    f"Could not read {name} from config.json. "
                    "Try --schema or check your checkpoint."
                )
                return

        print(f"[daemon] Schema: {self.schema.name}  "
              f"layers={self.num_layers}  experts={self.num_experts}  "
              f"topk={self.topk}  hidden={self.hidden_size}")

        print("[daemon] Loading embed_tokens.weight ...")
        self.embed_weight = ckpt.load_embed_tokens()
        print(f"[daemon] embed_tokens: {list(self.embed_weight.shape)}")

        print("[daemon] Loading router weights ...")
        router_keys = [k for k in ckpt.all_keys if self.schema.match_router(k)]
        router_tensors = ckpt.load_keys(router_keys)
        for key, tensor in router_tensors.items():
            m = self.schema.match_router(key)
            if m:
                self.routers[int(m.group("layer"))] = tensor.float()
        print(f"[daemon] Loaded {len(self.routers)} routers in {time.time()-t0:.1f}s")

        # Load tokenizer (optional — falls back to whitespace split)
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, trust_remote_code=True
            )
            print(f"[daemon] Tokenizer loaded: {type(self.tokenizer).__name__}")
        except Exception as e:
            print(f"[daemon] Warning: could not load tokenizer ({e}). "
                  "Using character-level fallback — stats will be approximate.")

        # Initialize stats
        from moe_pruner.stats import Stats
        self._stats = Stats(
            counts=torch.zeros(self.num_layers, self.num_experts, dtype=torch.float64),
            prob_sums=torch.zeros(self.num_layers, self.num_experts, dtype=torch.float64),
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            topk=self.topk,
            mode="daemon",
            num_samples=0,
        )

        self._loaded = True
        print("[daemon] Ready. Monitoring active.")

    def tokenize(self, text: str) -> list[int]:
        """Convert text to token IDs. Falls back to byte values if no tokenizer."""
        if self.tokenizer is not None:
            try:
                return self.tokenizer(text, add_special_tokens=False)["input_ids"]
            except Exception:
                pass
        # Fallback: ordinal values of each character (rough approximation)
        vocab_size = self.embed_weight.shape[0] if self.embed_weight is not None else 65536
        return [ord(c) % vocab_size for c in text[:2048]]

    def enqueue_prompt(self, prompt: str) -> None:
        """Add a prompt to the analysis queue (non-blocking, called from proxy thread)."""
        with self._queue_lock:
            self._queue.append(prompt)
        self._queue_event.set()

    def process_queue(self) -> None:
        """Background worker: drain the queue and accumulate stats."""
        while True:
            self._queue_event.wait()
            self._queue_event.clear()

            while True:
                with self._queue_lock:
                    if not self._queue:
                        break
                    prompt = self._queue.pop(0)

                if not self._loaded:
                    continue

                try:
                    self._analyze_prompt(prompt)
                except Exception as e:
                    print(f"[daemon] Analysis error: {e}", file=sys.stderr)

    def _analyze_prompt(self, prompt: str) -> None:
        from moe_pruner.stats import probe_prompt

        tokens = self.tokenize(prompt)
        if not tokens:
            return

        result = probe_prompt(
            tokens=tokens,
            embed_weight=self.embed_weight,
            routers=self.routers,
            topk=self.topk,
            num_experts=self.num_experts,
            num_layers=self.num_layers,
        )

        with self._lock:
            self._stats.accumulate(result)
            self._analyzed_count += 1
            analyzed = self._analyzed_count
            total = self._request_count

        if analyzed % self.report_every == 0:
            print(f"\n[daemon] === Auto-report after {analyzed} analyzed prompts "
                  f"({total} total requests) ===")
            self._emit_report()

    def emit_report_now(self) -> None:
        """Called on SIGUSR1 / SIGHUP to force a report."""
        print("\n[daemon] === Signal received — emitting report ===")
        self._emit_report()

    def _emit_report(self) -> None:
        """Write report JSON and print ranked table + prune command."""
        from moe_pruner.planner import build_plan
        from moe_pruner.report import build_report, print_report
        from moe_pruner.checkpoint import write_json

        with self._lock:
            if self._stats is None or self._stats.num_samples == 0:
                print("[daemon] No prompts analyzed yet — skipping report.")
                return
            # Snapshot
            import torch
            stats_snap = type(self._stats)(
                counts=self._stats.counts.clone(),
                prob_sums=self._stats.prob_sums.clone(),
                num_layers=self._stats.num_layers,
                num_experts=self._stats.num_experts,
                topk=self._stats.topk,
                mode=self._stats.mode,
                num_samples=self._stats.num_samples,
                elapsed_seconds=self._stats.elapsed_seconds,
            )

        plan = build_plan(
            stats_snap,
            keep_count=self.keep_experts,
            strategy=self.strategy,
            new_topk=self.new_topk,
        )

        report = build_report(stats_snap, plan, self.model_dir, self.schema.name)

        plan_path = self.output_path.parent / (self.output_path.stem + "-plan.json")
        write_json(self.output_path, report)
        write_json(plan_path, plan.to_dict())

        print_report(report, plan_path, self.model_dir)
        print(f"\n[daemon] Report: {self.output_path}")
        print(f"[daemon] Plan:   {plan_path}")


# ---------------------------------------------------------------------------
# HTTP proxy handler
# ---------------------------------------------------------------------------

def _extract_prompts(body: bytes) -> list[str]:
    """
    Extract prompt text from an OpenAI-compatible request body.

    Handles:
      - POST /v1/chat/completions  {"messages": [{"role": "user", "content": "..."}]}
      - POST /v1/completions       {"prompt": "..."}
      - POST /api/generate         {"prompt": "..."}  (ollama native)
      - POST /api/chat             {"messages": [...]}  (ollama chat)

    Returns a list of text strings (may be empty if unrecognized format).
    """
    try:
        data = json.loads(body)
    except Exception:
        return []

    texts = []

    # OpenAI chat completions / ollama chat
    if "messages" in data:
        for msg in data["messages"]:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    # Vision API: content is list of dicts with type/text
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))

    # OpenAI completions / ollama generate
    if "prompt" in data:
        p = data["prompt"]
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, list):
            texts.extend([x for x in p if isinstance(x, str)])

    return [t for t in texts if t.strip()]


def make_proxy_handler(state: DaemonState):
    """Return a handler class bound to the given DaemonState."""

    class ProxyHandler(BaseHTTPRequestHandler):

        def log_message(self, format, *args):
            pass  # Suppress default access log spam

        def do_GET(self):
            self._proxy()

        def do_POST(self):
            self._proxy()

        def do_DELETE(self):
            self._proxy()

        def do_PUT(self):
            self._proxy()

        def do_OPTIONS(self):
            self._proxy()

        def do_HEAD(self):
            self._proxy()

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length > 0 else b""

        def _proxy(self) -> None:
            body = self._read_body()

            with state._lock:
                state._request_count += 1

            # Enqueue prompt analysis (non-blocking)
            if state._loaded and body:
                prompts = _extract_prompts(body)
                for prompt in prompts:
                    state.enqueue_prompt(prompt)

            # Forward the request to backend
            target = f"{state.backend_url}{self.path}"
            headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length")
            }
            if body:
                headers["Content-Length"] = str(len(body))

            req = Request(target, data=body if body else None, headers=headers,
                          method=self.command)
            try:
                with urlopen(req, timeout=300) as resp:
                    resp_body = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() in ("content-length", "transfer-encoding",
                                         "connection"):
                            continue
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
            except URLError as e:
                msg = f"Backend unreachable: {e}".encode()
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except Exception as e:
                msg = f"Proxy error: {e}".encode()
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

    return ProxyHandler


# ---------------------------------------------------------------------------
# Daemon entry point
# ---------------------------------------------------------------------------

def run_daemon(
    model_dir: str,
    backend_url: str,
    listen_port: int,
    keep_experts: int,
    new_topk: int | None,
    output_path: Path,
    report_every: int = 100,
    schema_name: str | None = None,
    strategy: str = "per-layer",
    listen_host: str = "0.0.0.0",
) -> None:
    state = DaemonState(
        model_dir=model_dir,
        schema_name=schema_name,
        keep_experts=keep_experts,
        new_topk=new_topk,
        report_every=report_every,
        output_path=output_path,
        backend_url=backend_url,
        strategy=strategy,
    )

    # Load weights in background so we start proxying immediately
    load_thread = threading.Thread(target=state.load, daemon=True, name="loader")
    load_thread.start()

    # Background analysis worker
    worker_thread = threading.Thread(
        target=state.process_queue, daemon=True, name="analyzer"
    )
    worker_thread.start()

    # Signal handlers for on-demand reports
    def _signal_report(signum, frame):
        threading.Thread(target=state.emit_report_now, daemon=True).start()

    try:
        signal.signal(signal.SIGUSR1, _signal_report)
        signal.signal(signal.SIGHUP, _signal_report)
    except (AttributeError, OSError):
        pass  # Windows doesn't have SIGUSR1/SIGHUP

    handler_class = make_proxy_handler(state)
    server = HTTPServer((listen_host, listen_port), handler_class)

    print(f"[daemon] Proxy listening on {listen_host}:{listen_port}")
    print(f"[daemon] Forwarding to backend: {backend_url}")
    print(f"[daemon] Report every {report_every} analyzed requests → {output_path}")
    print(f"[daemon] Target: {keep_experts} experts  strategy={strategy}")
    print(f"[daemon] Send SIGUSR1 or SIGHUP to force an immediate report.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[daemon] Interrupted. Emitting final report...")
        state.emit_report_now()
        server.server_close()
