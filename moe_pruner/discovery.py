"""
Auto-discovery of running model servers and system environment.

Probes well-known ports for OpenAI-compatible and native API endpoints.
Detects OS, GPU hardware, and available inference runtimes.
No external dependencies — pure stdlib (plus optional torch for GPU info).
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

@dataclass
class SystemInfo:
    os_name: str          # "Linux", "macOS", "Windows"
    os_version: str
    arch: str             # "x86_64", "arm64", etc.
    gpus: list[str]       # e.g. ["NVIDIA L4 (22 GB)", "NVIDIA A100 (80 GB)"]
    gpu_backend: str      # "cuda", "rocm", "metal", "cpu"
    cpu_cores: int
    ram_gb: float


def detect_system() -> SystemInfo:
    os_name = platform.system()
    if os_name == "Darwin":
        os_name = "macOS"
    os_version = platform.version()
    arch = platform.machine()
    cpu_cores = os.cpu_count() or 1

    # RAM
    ram_gb = 0.0
    try:
        import resource
        # Not available on all platforms; use /proc/meminfo on Linux
        pass
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_gb = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass
    if ram_gb == 0:
        try:
            import subprocess
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
            )
            ram_gb = int(out.strip()) / 1024 ** 3
        except Exception:
            pass

    gpus: list[str] = []
    gpu_backend = "cpu"

    # NVIDIA via torch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_backend = "cuda"
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1024 ** 3
                gpus.append(f"{props.name} ({vram_gb:.0f} GB VRAM)")
    except Exception:
        pass

    # AMD ROCm
    if not gpus:
        try:
            import subprocess
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL
            ).decode()
            for line in out.splitlines():
                line = line.strip()
                if line and not line.startswith("=") and "GPU" in line.upper():
                    gpus.append(f"AMD {line.strip()}")
            if gpus:
                gpu_backend = "rocm"
        except Exception:
            pass

    # Apple Metal / MPS
    if not gpus:
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                import subprocess
                try:
                    out = subprocess.check_output(
                        ["system_profiler", "SPDisplaysDataType"], stderr=subprocess.DEVNULL
                    ).decode()
                    for line in out.splitlines():
                        if "Chipset Model" in line:
                            chip = line.split(":")[-1].strip()
                            gpus.append(f"Apple {chip} (Metal/MPS)")
                            break
                except Exception:
                    gpus.append("Apple Silicon (Metal/MPS)")
                gpu_backend = "metal"
        except Exception:
            pass

    if not gpus:
        gpus = ["None (CPU-only)"]

    return SystemInfo(
        os_name=os_name,
        os_version=os_version,
        arch=arch,
        gpus=gpus,
        gpu_backend=gpu_backend,
        cpu_cores=cpu_cores,
        ram_gb=round(ram_gb, 1),
    )


def print_system_info(info: SystemInfo) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        c = Console()
        gpu_str = "\n".join(f"    • {g}" for g in info.gpus)
        text = (
            f"[bold]OS:[/bold]      {info.os_name} {info.arch}\n"
            f"[bold]CPUs:[/bold]    {info.cpu_cores} cores"
            + (f"  |  RAM: {info.ram_gb} GB" if info.ram_gb else "") + "\n"
            f"[bold]GPU(s):[/bold]  {info.gpus[0]}"
            + ("".join(f"\n         {g}" for g in info.gpus[1:]) if len(info.gpus) > 1 else "") + "\n"
            f"[bold]Backend:[/bold] {info.gpu_backend.upper()}"
        )
        c.print(Panel(text, title="[bold cyan]System", border_style="cyan"))
    except ImportError:
        print("\n  System:")
        print(f"    OS:      {info.os_name} {info.arch}")
        print(f"    CPUs:    {info.cpu_cores} cores" + (f"  RAM: {info.ram_gb} GB" if info.ram_gb else ""))
        print(f"    GPU(s):  {', '.join(info.gpus)}")
        print(f"    Backend: {info.gpu_backend.upper()}")
        print()


# ---------------------------------------------------------------------------
# Server discovery
# ---------------------------------------------------------------------------

@dataclass
class ModelServer:
    name: str           # e.g. "ollama", "vLLM"
    url: str            # e.g. "http://localhost:11434"
    port: int
    models: list[str] = field(default_factory=list)


# (runtime_name, port, models_path, list_key, name_key)
_KNOWN_RUNTIMES = [
    ("ollama",               11434, "/api/tags",   "models", "name"),
    ("vLLM",                  8000, "/v1/models",  "data",   "id"),
    ("vLLM",                  8080, "/v1/models",  "data",   "id"),
    ("llama.cpp server",      8080, "/v1/models",  "data",   "id"),
    ("llama.cpp server",      8000, "/v1/models",  "data",   "id"),
    ("LM Studio",             1234, "/v1/models",  "data",   "id"),
    ("text-gen-webui",        5000, "/v1/models",  "data",   "id"),
    ("text-gen-webui",        7860, "/v1/models",  "data",   "id"),
    ("koboldcpp",             5001, "/v1/models",  "data",   "id"),
    ("OpenAI-compat proxy",   4000, "/v1/models",  "data",   "id"),
]


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _fetch_json(url: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            import json
            return json.loads(resp.read())
    except Exception:
        return None


def discover_servers(host: str = "localhost") -> list[ModelServer]:
    """Probe well-known ports and return a list of running model servers."""
    found: dict[int, ModelServer] = {}

    for (name, port, models_path, list_key, name_key) in _KNOWN_RUNTIMES:
        if port in found:
            continue
        if not _port_open(host, port):
            continue

        base = f"http://{host}:{port}"
        models: list[str] = []
        data = _fetch_json(base + models_path)
        if data:
            items = data.get(list_key, [])
            if isinstance(items, list):
                for item in items:
                    n = item.get(name_key, "") if isinstance(item, dict) else str(item)
                    if n:
                        models.append(n)

        found[port] = ModelServer(name=name, url=base, port=port, models=models)

    return list(found.values())


def print_discovered(servers: list[ModelServer]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        c = Console()
        t = Table(title="Discovered Model Servers", border_style="cyan")
        t.add_column("#", width=3, style="dim")
        t.add_column("Runtime", style="bold cyan")
        t.add_column("URL", style="bold white")
        t.add_column("Models", style="yellow")
        for i, s in enumerate(servers, 1):
            models_str = ", ".join(s.models[:3])
            if len(s.models) > 3:
                models_str += f" (+{len(s.models)-3} more)"
            t.add_row(str(i), s.name, s.url, models_str or "—")
        c.print(t)
    except ImportError:
        print("\n  Discovered model servers:")
        for i, s in enumerate(servers, 1):
            models_str = ", ".join(s.models[:3]) or "—"
            print(f"    {i}. {s.name:25s}  {s.url}  [{models_str}]")
        print()


# ---------------------------------------------------------------------------
# Interactive setup wizard
# ---------------------------------------------------------------------------

def _ask(prompt: str, choices: list[str], allow_custom: bool = False) -> str:
    """Print numbered choices and return the selected value."""
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    if allow_custom:
        print(f"  {len(choices)+1}. Enter manually")
    while True:
        try:
            raw = input(f"\n{prompt} [1-{len(choices) + (1 if allow_custom else 0)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            if allow_custom and idx == len(choices):
                try:
                    val = input("  Enter value: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    sys.exit(0)
                if val:
                    return val
        print(f"  Please enter a number between 1 and {len(choices) + (1 if allow_custom else 0)}.")


def run_setup_wizard(
    backend_url: Optional[str],
    model_dir: Optional[str],
) -> tuple[str, str]:
    """
    Interactive setup wizard.

    Detects system, finds running servers, asks the user to confirm/choose.
    Returns (backend_url, model_dir) ready to pass to run_daemon().
    """
    try:
        from rich.console import Console
        from rich.rule import Rule
        c = Console()
        c.print(Rule("[bold cyan]MoE-Watcher-Modifier Daemon Setup", style="cyan"))
    except ImportError:
        print("=" * 60)
        print("  MoE-Watcher-Modifier Daemon Setup")
        print("=" * 60)

    # 1. System info
    print("\n[1/3] Detecting system...")
    info = detect_system()
    print_system_info(info)

    # 2. Model directory validation
    print("[2/3] Checkpoint directory...")
    while True:
        if model_dir:
            p = Path(model_dir).expanduser().resolve()
            if not p.exists():
                print(f"\n  ERROR: Directory not found: {p}")
                print("  Please enter the correct path to your safetensors checkpoint.")
                try:
                    model_dir = input("  --model-dir: ").strip() or None
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    sys.exit(0)
                continue
            # Quick check for safetensors files
            st_files = list(p.glob("*.safetensors")) + list(p.glob("*.safetensors.index.json"))
            if not st_files:
                print(f"\n  WARNING: No safetensors files found in {p}")
                print("  Are you sure this is the right directory?")
                try:
                    confirm = input("  Continue anyway? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    sys.exit(0)
                if confirm != "y":
                    try:
                        model_dir = input("  --model-dir: ").strip() or None
                    except (EOFError, KeyboardInterrupt):
                        sys.exit(0)
                    continue
            print(f"\n  Checkpoint: {p}")
            model_dir = str(p)
            break
        else:
            print("\n  No --model-dir provided.")
            try:
                model_dir = input("  Enter path to checkpoint directory: ").strip() or None
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)

    # 3. Backend server
    print("\n[3/3] Inference backend...")
    if backend_url:
        print(f"\n  Using --backend: {backend_url}")
    else:
        print("\n  Scanning for running model servers...")
        servers = discover_servers()

        if servers:
            print_discovered(servers)
            if len(servers) == 1:
                print(f"  Only one server found — using {servers[0].url}")
                backend_url = servers[0].url
            else:
                choices = [f"{s.name}  {s.url}" for s in servers]
                choice = _ask("Select backend server", choices, allow_custom=True)
                # Extract URL from "name  url" format if it came from choices
                if "  " in choice and choice.startswith(tuple(s.name for s in servers)):
                    backend_url = choice.split("  ", 1)[1].strip()
                else:
                    backend_url = choice
        else:
            print("\n  No servers found on well-known ports.")
            print("  Make sure your model server is running, then choose:")
            runtime_choices = [
                "ollama   → http://localhost:11434",
                "vLLM     → http://localhost:8000",
                "llama.cpp→ http://localhost:8080",
                "LM Studio→ http://localhost:1234",
            ]
            choice = _ask("Which runtime are you using?", runtime_choices, allow_custom=True)
            if "→" in choice:
                backend_url = choice.split("→", 1)[1].strip()
            else:
                backend_url = choice

    print()
    try:
        from rich.console import Console
        from rich.rule import Rule
        Console().print(Rule("[bold green]Setup complete", style="green"))
    except ImportError:
        print("-" * 60)
        print("  Setup complete")
        print("-" * 60)
    print()

    return backend_url, model_dir
