"""
Lightweight LLM adapter with JSON caching for deterministic backtests.

The adapter makes a single chat completion style request (OpenAI-compatible)
per prompt, caches the structured JSON response, and returns parsed content.
If the API key is missing or the call fails, it falls back to a neutral stub
so the backtest can proceed without breaking.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class LLMConfig:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1/chat/completions"
    temperature: float = 0.1
    cache_path: str = ".llm_cache.json"
    timeout: int = 20


def _load_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_cache(path: str, cache: Dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    except Exception:
        pass


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()


def _call_llm(prompt: str, config: LLMConfig) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not config.api_key:
        return None, "LLM_API_KEY missing; skipping LLM call."

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You are a concise trading assistant. Respond ONLY with JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    try:
        resp = requests.post(config.base_url, headers=headers, json=payload, timeout=config.timeout)
    except Exception as exc:
        return None, f"LLM call failed: {exc}"

    if resp.status_code != 200:
        truncated = resp.text[:200]
        return None, f"LLM HTTP {resp.status_code}: {truncated}"

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed, None
    except Exception as exc:
        return None, f"Failed to parse LLM response: {exc}"


def build_prompt(ticker: str, df_5m, df_1m) -> str:
    """
    Create a compact prompt describing recent indicator context.
    Expects df_5m columns: ts, close, macd, macd_signal, macd_hist, sma_5m
    Expects df_1m columns: ts, close, vwap_1m
    """
    g5 = df_5m.sort_values("ts").tail(6)
    g1 = df_1m.sort_values("ts").tail(20)
    last_close = g5["close"].iloc[-1] if not g5.empty else None
    macd = g5[["macd", "macd_signal", "macd_hist"]].tail(3).round(4).to_dict("records") if not g5.empty else []
    sma = g5["sma_5m"].iloc[-1] if not g5.empty else None
    vwap = g1["vwap_1m"].iloc[-1] if not g1.empty else None

    prompt = (
        f"Ticker: {ticker}. Latest close: {last_close}. "
        f"SMA_5m: {sma}, VWAP_1m: {vwap}. "
        f"Recent MACD rows: {macd}. "
        "Return JSON with keys: "
        '{"sentiment": float between -1 and 1, '
        '"risk_scale": int 1-5 where 1=avoid, 5=aggressive, '
        '"do_not_trade": boolean, '
        '"note": short string insight}. '
        "Keep note under 140 chars."
    )
    return prompt


def get_llm_annotation(ticker: str, df_5m, df_1m, config: LLMConfig) -> Tuple[Dict[str, Any], str]:
    """
    Fetch annotation for a ticker, using cache when available.
    Returns (annotation_dict, source), where source is 'cache', 'live', or 'stub'.
    """
    prompt = build_prompt(ticker, df_5m, df_1m)
    key = _cache_key(config.model, prompt)
    cache = _load_cache(config.cache_path)

    if key in cache:
        return cache[key], "cache"

    payload, err = _call_llm(prompt, config)
    if payload:
        cache[key] = payload
        _save_cache(config.cache_path, cache)
        return payload, "live"

    # Fallback stub when the call fails
    stub = {"sentiment": 0.0, "risk_scale": 3, "do_not_trade": False, "note": err or "stub"}
    cache[key] = stub
    _save_cache(config.cache_path, cache)
    return stub, "stub"
