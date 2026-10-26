"""Local inference engine: load/unload specialists, generate with timing and resource stats.

Works fully offline once exports are on disk (set HF_HUB_OFFLINE=1 for HF-tokenizer models).
Models are registered in configs/models.yaml (name -> export directory).
"""

from __future__ import annotations

import gc
import os
import threading
import time

import torch
import yaml

from ..datasets.tokenizer import Tokenizer, prompt_ids
from ..training.config import resolve_path
from .export import load_export
from .router import route


def system_stats() -> dict:
    out = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["ram_used_gib"] = round((vm.total - vm.available) / 1024 ** 3, 2)
        out["ram_total_gib"] = round(vm.total / 1024 ** 3, 2)
        out["process_rss_gib"] = round(psutil.Process().memory_info().rss / 1024 ** 3, 3)
    except ImportError:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_used_gib"] = round((total - free) / 1024 ** 3, 2)
        out["gpu_total_gib"] = round(total / 1024 ** 3, 2)
        out["torch_allocated_gib"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 3)
    return out


def load_tokenizer_from_export(path):
    tj = os.path.join(path, "tokenizer.json")
    if os.path.exists(tj) and not os.path.exists(os.path.join(path, "tokenizer_config.json")):
        from tokenizers import Tokenizer as HFTok
        return Tokenizer(HFTok.from_file(tj), "bpe", "tokenizer.json")
    from transformers import AutoTokenizer
    return Tokenizer(AutoTokenizer.from_pretrained(path), "hf", os.path.basename(path))


class Engine:
    def __init__(self, registry_path="configs/models.yaml", device=None, backend="reference"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backend = backend
        self.registry = {}
        rp = resolve_path(registry_path)
        if os.path.exists(rp):
            with open(rp) as f:
                self.registry = (yaml.safe_load(f) or {}).get("models", {})
        self.loaded: dict = {}
        self.lock = threading.Lock()

    def available(self):
        out = {}
        for name, spec in self.registry.items():
            p = resolve_path(spec["path"])
            out[name] = {"path": p, "installed": os.path.exists(os.path.join(p, "triforge.json")),
                         "loaded": name in self.loaded, "specialist": spec.get("specialist", name)}
        return out

    def load(self, name):
        with self.lock:
            if name in self.loaded:
                return self.loaded[name]["meta"]
            if name not in self.registry:
                raise KeyError(f"unknown model {name!r}; registered: {sorted(self.registry)}")
            path = resolve_path(self.registry[name]["path"])
            if not os.path.exists(os.path.join(path, "triforge.json")):
                raise FileNotFoundError(f"{name}: no export at {path}. Train and export it first "
                                        f"(see README), or download it with scripts/download_hf.py.")
            t0 = time.time()
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            model = load_export(path, device=self.device, backend=self.backend, dtype=dtype)
            tok = load_tokenizer_from_export(path)
            meta = {**model.export_meta, "load_seconds": round(time.time() - t0, 2)}
            self.loaded[name] = {"model": model, "tok": tok, "meta": meta}
            return meta

    def unload(self, name):
        with self.lock:
            self.loaded.pop(name, None)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def pick(self, prompt, model=None):
        """Explicit model name wins; otherwise the router recommends a specialist."""
        if model and model != "auto":
            return model, None
        r = route(prompt)
        name = next((n for n, s in self.registry.items() if s.get("specialist", n) == r.specialist), r.specialist)
        return name, r

    @torch.no_grad()
    def generate(self, prompt, model=None, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.95,
                 stream=None):
        name, r = self.pick(prompt, model)
        self.load(name)
        entry = self.loaded[name]
        m, tok, meta = entry["model"], entry["tok"], entry["meta"]
        ids = prompt_ids(tok, prompt, meta.get("template", "triforge"))
        ids = ids[-(m.cfg.max_seq_len - max_new_tokens):]
        x = torch.tensor([ids], device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        first = {}

        def on_token(t):
            first.setdefault("t", time.perf_counter())
            if stream:
                stream(tok.decode([t]))

        t0 = time.perf_counter()
        out = m.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                         eos_id=tok.eos_id, stream=on_token)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        new = out[0, len(ids):].tolist()
        ttft = first.get("t", t1) - t0
        decode_s = max(1e-9, t1 - first.get("t", t0))
        stats = {"model": name, "status": meta.get("status"), "prompt_tokens": len(ids), "new_tokens": len(new),
                 "time_to_first_token_s": round(ttft, 4),
                 "prompt_tokens_per_s": round(len(ids) / max(ttft, 1e-9), 1),
                 "decode_tokens_per_s": round(max(0, len(new) - 1) / decode_s, 1), **system_stats()}
        if self.device.type == "cuda":
            stats["peak_gpu_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        if r is not None:
            stats["router"] = {"specialist": r.specialist, "scores": r.scores, "reason": r.reason}
        return tok.decode(new), stats
