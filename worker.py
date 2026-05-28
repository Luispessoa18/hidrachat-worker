from __future__ import annotations

import os
import subprocess
import time
import atexit
from dataclasses import dataclass

import urllib.error
import urllib.request
import json
from pathlib import Path


ROOT_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = ROOT_DIR / "models"
LLAMA_SERVER_PROCESS: subprocess.Popen | None = None


@dataclass
class Config:
    root_url:    str   = os.getenv("HIDRACHAT_ROOT_URL",    "https://hidrachat.cloud")
    name:        str   = os.getenv("HIDRACHAT_WORKER_NAME", "my-worker")
    owner_email: str   = os.getenv("HIDRACHAT_WORKER_EMAIL", "")
    llama_bin:   str   = os.getenv("HIDRACHAT_LLAMACPP_BIN", "llama-cli")
    model_path:  str   = os.getenv("HIDRACHAT_MODEL_PATH",  "")
    model_name:  str   = os.getenv("HIDRACHAT_MODEL_NAME",  "local-gguf")
    model_size:  str   = os.getenv("HIDRACHAT_MODEL_SIZE",  "any")
    region:      str   = os.getenv("HIDRACHAT_REGION",      "local")
    threads:     int   = int(os.getenv("HIDRACHAT_THREADS",       "4"))
    ram_gb:      float = float(os.getenv("HIDRACHAT_RAM_GB",      "4"))
    poll_seconds: float = float(os.getenv("HIDRACHAT_POLL_SECONDS", "2"))
    llama_server_bin: str = ""
    llama_port:  int   = int(os.getenv("HIDRACHAT_LLAMA_PORT",    "8091"))
    n_gpu_layers: int  = int(os.getenv("HIDRACHAT_N_GPU_LAYERS",  "0"))
    ctx_size:    int   = int(os.getenv("HIDRACHAT_CTX_SIZE",      "4096"))
    system_prompt: str = os.getenv(
        "HIDRACHAT_SYSTEM_PROMPT",
        "Voce e um assistente direto e util. Responda em portugues do Brasil. "
        "Para cumprimentos simples, responda em uma frase curta. "
        "Nao invente redes sociais, biografia, blog, links ou listas se o usuario nao pedir.",
    )


def find_llamacpp_binary() -> str:
    env_bin = os.getenv("HIDRACHAT_LLAMACPP_BIN")
    if env_bin:
        return env_bin
    base = ROOT_DIR / "llamacpp"
    candidates = [
        base / "build" / "bin" / "llama-cli",
        base / "build" / "bin" / "main",
        base / "build" / "bin" / "Release" / "llama-cli.exe",
        base / "build" / "bin" / "Release" / "main.exe",
        base / "build" / "bin" / "llama-cli.exe",
        base / "build" / "bin" / "main.exe",
        base / "llama-cli.exe",
        base / "main.exe",
        base / "llama-cli",
        base / "main",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "llama-cli"


def find_llama_server_binary() -> str:
    env_bin = os.getenv("HIDRACHAT_LLAMA_SERVER_BIN")
    if env_bin:
        return env_bin
    base = ROOT_DIR / "llamacpp"
    candidates = [
        base / "build" / "bin" / "llama-server",
        base / "build" / "bin" / "Release" / "llama-server.exe",
        base / "build" / "bin" / "llama-server.exe",
        base / "llama-server.exe",
        base / "llama-server",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def find_models() -> list[Path]:
    models_dir = Path(os.getenv("HIDRACHAT_MODEL_DIR", str(MODELS_DIR)))
    if not models_dir.exists():
        return []
    return sorted(models_dir.rglob("*.gguf"), key=lambda p: p.name.lower())


def infer_model_size(model_path: Path) -> str:
    name = model_path.name.lower()
    for marker in ("0.5b", "1b", "1.5b", "2b", "3b", "7b", "8b", "13b", "14b", "32b"):
        if marker in name:
            return marker
    return "any"


def choose_model() -> Path:
    env_model = os.getenv("HIDRACHAT_MODEL_PATH")
    if env_model:
        return Path(env_model)
    models = find_models()
    if not models:
        raise RuntimeError(
            "No .gguf model found in models/. Download one and place it there, "
            "or set HIDRACHAT_MODEL_PATH."
        )
    print("\nGGUF models found:")
    for i, m in enumerate(models, 1):
        mb = m.stat().st_size / (1024 * 1024)
        print(f"  {i}. {m.relative_to(ROOT_DIR)} ({mb:.0f} MB)")
    while True:
        choice = input(f"Choose model [1-{len(models)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid choice.")


def prepare_config() -> Config:
    cfg = Config()
    if not cfg.owner_email:
        cfg.owner_email = input("HidraChat account email: ").strip().lower()
    model_path = choose_model()
    cfg.model_path   = str(model_path)
    cfg.llama_bin    = find_llamacpp_binary()
    cfg.llama_server_bin = find_llama_server_binary()
    cfg.model_name   = os.getenv("HIDRACHAT_MODEL_NAME", model_path.stem)
    cfg.model_size   = os.getenv("HIDRACHAT_MODEL_SIZE", infer_model_size(model_path))
    return cfg


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def count_tokens_rough(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def effective_max_tokens(prompt: str, requested: int) -> int:
    normalized = prompt.strip().lower().replace("!", "").replace(".", "")
    greetings  = {"oi", "ola", "olá", "bom dia", "boa tarde", "boa noite", "hi", "hello"}
    if normalized in greetings:
        return min(requested, 32)
    if len(prompt.split()) <= 5:
        return min(requested, 96)
    return requested


def format_prompt(cfg: Config, user_prompt: str) -> str:
    return (
        f"Sistema: {cfg.system_prompt}\n\n"
        f"Usuario: {user_prompt.strip()}\n\n"
        "Assistente:"
    )


def llama_server_url(cfg: Config, path: str) -> str:
    return f"http://127.0.0.1:{cfg.llama_port}{path}"


def llama_server_ready(cfg: Config) -> bool:
    try:
        get_json(llama_server_url(cfg, "/health"), timeout=2)
        return True
    except Exception:
        return False


def stop_llama_server() -> None:
    global LLAMA_SERVER_PROCESS
    if LLAMA_SERVER_PROCESS and LLAMA_SERVER_PROCESS.poll() is None:
        LLAMA_SERVER_PROCESS.terminate()
    LLAMA_SERVER_PROCESS = None


def ensure_llama_server(cfg: Config) -> bool:
    global LLAMA_SERVER_PROCESS
    if not cfg.llama_server_bin:
        return False
    if llama_server_ready(cfg):
        print(f"llama-server already running at 127.0.0.1:{cfg.llama_port}")
        return True
    cmd = [cfg.llama_server_bin, "-m", cfg.model_path,
           "--host", "127.0.0.1", "--port", str(cfg.llama_port),
           "-t", str(cfg.threads), "-c", str(cfg.ctx_size)]
    if cfg.n_gpu_layers > 0:
        cmd.extend(["-ngl", str(cfg.n_gpu_layers)])
    print("Loading model into llama-server (first load may take a while)...")
    LLAMA_SERVER_PROCESS = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    atexit.register(stop_llama_server)
    deadline = time.time() + 180
    while time.time() < deadline:
        if LLAMA_SERVER_PROCESS.poll() is not None:
            raise RuntimeError("llama-server exited during model load")
        if llama_server_ready(cfg):
            print("Model loaded and ready.")
            _warmup(cfg)
            return True
        time.sleep(1)
    raise RuntimeError("Timeout waiting for llama-server")


def _warmup(cfg: Config) -> None:
    try:
        post_json(llama_server_url(cfg, "/v1/chat/completions"),
                  {"messages": [{"role": "user", "content": "ok"}], "max_tokens": 2, "temperature": 0},
                  timeout=120)
        print("Warmup done.")
    except Exception as e:
        print(f"Warmup failed (continuing anyway): {e}")


def run_server_mode(cfg: Config, prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
    max_tokens = effective_max_tokens(prompt, max_tokens)
    started    = time.perf_counter()
    try:
        resp = post_json(llama_server_url(cfg, "/v1/chat/completions"),
                         {"messages": [{"role": "system", "content": cfg.system_prompt},
                                       {"role": "user",   "content": prompt.strip()}],
                          "max_tokens": max_tokens, "temperature": 0.35, "top_p": 0.9,
                          "stop": ["\nUsuario:", "\nUsuário:", "\nSistema:"]},
                         timeout=max(180, max_tokens * 6))
    except urllib.error.HTTPError:
        return run_completion_mode(cfg, prompt, max_tokens)

    output = ""
    if resp.get("choices"):
        c = resp["choices"][0]
        output = (c.get("message") or {}).get("content") or c.get("text", "")
    output = output or resp.get("content") or resp.get("response") or ""
    output = str(output).strip()
    if not output:
        raise RuntimeError(f"Empty output from llama-server: {json.dumps(resp)[:300]}")
    elapsed  = time.perf_counter() - started
    out_toks = count_tokens_rough(output)
    return output, int(elapsed * 1000), out_toks, out_toks / elapsed if elapsed > 0 else 0


def run_completion_mode(cfg: Config, prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
    max_tokens = effective_max_tokens(prompt, max_tokens)
    started    = time.perf_counter()
    resp       = post_json(llama_server_url(cfg, "/completion"),
                           {"prompt": format_prompt(cfg, prompt), "n_predict": max_tokens,
                            "temperature": 0.35, "top_p": 0.9, "cache_prompt": True,
                            "stop": ["\nUsuario:", "\nUsuário:", "\nSistema:", "</s>"]},
                           timeout=max(180, max_tokens * 6))
    output  = str(resp.get("content") or resp.get("response") or "").strip()
    if not output:
        raise RuntimeError(f"Empty output: {json.dumps(resp)[:300]}")
    elapsed  = time.perf_counter() - started
    out_toks = count_tokens_rough(output)
    return output, int(elapsed * 1000), out_toks, out_toks / elapsed if elapsed > 0 else 0


def run_cli_mode(cfg: Config, prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
    max_tokens = effective_max_tokens(prompt, max_tokens)
    cmd        = [cfg.llama_bin, "-m", cfg.model_path, "-p", format_prompt(cfg, prompt),
                  "-n", str(max_tokens), "-t", str(cfg.threads),
                  "--no-display-prompt", "--temp", "0.35"]
    started    = time.perf_counter()
    proc       = subprocess.run(cmd, text=True, capture_output=True, timeout=max(120, max_tokens * 4))
    elapsed    = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"llama-cli exited with {proc.returncode}")
    output = proc.stdout.strip()
    if not output:
        raise RuntimeError("llama-cli returned empty output")
    out_toks = count_tokens_rough(output)
    return output, int(elapsed * 1000), out_toks, out_toks / elapsed if elapsed > 0 else 0


def detect_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024 ** 2), 1)
    except Exception:
        pass
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            return round(int(r.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        r = subprocess.run(["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                           capture_output=True, text=True, timeout=3)
        for line in r.stdout.splitlines():
            if line.strip().isdigit():
                return round(int(line.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    return float(os.getenv("HIDRACHAT_RAM_GB", "4"))


def detect_gpu(n_gpu_layers: int) -> str:
    if n_gpu_layers <= 0:
        return "CPU"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return "AMD ROCm"
    except Exception:
        pass
    return "Vulkan/GPU"


def register(cfg: Config) -> str:
    ram = detect_ram_gb()
    gpu = detect_gpu(cfg.n_gpu_layers)
    cfg.ram_gb = ram
    print(f"RAM: {ram} GB  |  Backend: {gpu}")
    res = post_json(f"{cfg.root_url}/worker/register",
                    {"name": cfg.name, "owner_email": cfg.owner_email,
                     "worker_type": "desktop", "region": cfg.region,
                     "model_name": cfg.model_name, "model_size": cfg.model_size,
                     "ram_gb": ram, "cpu_threads": cfg.threads,
                     "gpu": gpu, "tokens_per_second": 1})
    return res["worker_id"]


def heartbeat(cfg: Config, worker_id: str, tps: float = 0) -> None:
    post_json(f"{cfg.root_url}/worker/heartbeat",
              {"worker_id": worker_id, "ram_available_gb": cfg.ram_gb, "tokens_per_second": tps})


def main() -> None:
    cfg         = prepare_config()
    server_mode = ensure_llama_server(cfg)
    print(f"\nllama-server: {cfg.llama_server_bin or '-'}")
    print(f"llama-cli:    {cfg.llama_bin}")
    print(f"Model:        {cfg.model_path}")
    worker_id = register(cfg)
    print(f"Worker registered: {worker_id}")
    print(f"Connected to: {cfg.root_url}\n")
    last_tps = 0.0
    while True:
        try:
            heartbeat(cfg, worker_id, last_tps)
            task = get_json(f"{cfg.root_url}/pull-task?worker_id={worker_id}")
            if not task.get("job_id"):
                time.sleep(cfg.poll_seconds)
                continue
            print(f"[JOB] {task['job_id']} ({task.get('type','gen')}, {task.get('complexity','?')})")
            try:
                if server_mode:
                    out, ms, toks, last_tps = run_server_mode(cfg, task["prompt"], int(task.get("max_tokens") or 256))
                else:
                    out, ms, toks, last_tps = run_cli_mode(cfg, task["prompt"], int(task.get("max_tokens") or 256))
                print(f"[DONE] {toks} tokens, {last_tps:.1f} tok/s")
                post_json(f"{cfg.root_url}/job/submit",
                          {"job_id": task["job_id"], "worker_id": worker_id, "output": out,
                           "input_tokens": count_tokens_rough(task["prompt"]),
                           "output_tokens": toks, "worker_time_ms": ms, "success": True})
            except Exception as exc:
                print(f"[FAIL] {exc}")
                post_json(f"{cfg.root_url}/job/submit",
                          {"job_id": task["job_id"], "worker_id": worker_id,
                           "output": "", "worker_time_ms": 0, "success": False, "error": str(exc)})
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[OFFLINE] {exc}")
            time.sleep(max(5, cfg.poll_seconds))


if __name__ == "__main__":
    main()
