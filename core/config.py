"""Config loader — YAML + CLI overrides. No secrets, pure offline."""

from __future__ import annotations

import copy
import os
from typing import Any, Optional

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "demo.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    cfg_path = path or os.environ.get("ASSISTANT_CONFIG", DEFAULT_PATH)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    # resolve relative paths against project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(cfg_path)))
    _resolve_paths(cfg, root)
    return cfg


def _resolve_paths(obj: Any, root: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and not os.path.isabs(v):
                # path-like keys or values containing models/data or file extensions
                looks_path = any(
                    v.startswith(p) for p in ("models/", "data/", "config/", "./")
                ) or v.endswith((".onnx", ".xml", ".bin", ".wav", ".db", ".npz", ".txt"))
                if looks_path:
                    cand = os.path.normpath(os.path.join(root, v))
                    # only resolve if exists or it's under known dirs (may be created later)
                    obj[k] = cand
            elif isinstance(v, (dict, list)):
                _resolve_paths(v, root)
    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, (dict, list)):
                _resolve_paths(x, root)
