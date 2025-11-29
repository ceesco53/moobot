#!/usr/bin/env python
"""
Sample a few days of OHLCV, build a compact JSON feature blob, prompt an LLM for rule JSON, and write it to disk.

Usage:
  DB_URL=postgresql://moobot:moobot@localhost:5432/marketdata \
  LLM_API_KEY=... \
  python scripts/generate_llm_rules.py --tickers NVDA AMD --start 2025-06-01 --end 2025-11-27 \
    --sample-days 3 --out llm_rules.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from sqlalchemy import create_engine, text

# Ensure project root on path for sibling imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_adapter import LLMConfig  # noqa: E402

DB_URL = os.getenv("DB_URL", "postgresql://moobot:moobot@localhost:5432/marketdata")


def fetch_data(engine, tickers: List[str], start: str, end: str) -> pd.DataFrame:
    placeholders = ", ".join([f":t{i}" for i in range(len(tickers))])
    params = {f"t{i}": t for i, t in enumerate(tickers)}
    params.update({"start": start, "end": end})
    sql = text(
        f"""
        SELECT ticker, window_start AT TIME ZONE 'UTC' AS ts_utc,
               open, high, low, close, volume
        FROM ohlcv
        WHERE ticker IN ({placeholders})
          AND window_start >= :start
          AND window_start < :end
        ORDER BY ticker, window_start
        """
    )
    df = pd.read_sql(sql, engine, params=params, parse_dates=["ts_utc"])
    if df.empty:
        return df
    df["ts"] = df["ts_utc"].dt.tz_localize("UTC")
    df["date"] = df["ts"].dt.date
    return df.drop(columns=["ts_utc"])


def build_features(df: pd.DataFrame, sample_days: int) -> Dict[str, List[Dict]]:
    feats: Dict[str, List[Dict]] = {}
    for ticker, g in df.groupby("ticker"):
        days = sorted(g["date"].unique())[:sample_days]
        g = g[g["date"].isin(days)]
        per_day = []
        for d, gd in g.groupby("date"):
            o = gd.iloc[0]["open"]
            c = gd.iloc[-1]["close"]
            h = gd["high"].max()
            l = gd["low"].min()
            v = gd["volume"].sum()
            ret = (c - o) / o if o else 0.0
            hl = (h - l) / o if o else 0.0
            # Simple vwap approximation using dollar volume / volume
            vwap = (gd["close"] * gd["volume"]).sum() / v if v else c
            gd = gd.sort_values("ts")
            gd["ret1"] = gd["close"].pct_change()
            vol_intraday = gd["ret1"].std(skipna=True)
            per_day.append(
                {
                    "date": str(d),
                    "open": round(float(o), 4),
                    "close": round(float(c), 4),
                    "high": round(float(h), 4),
                    "low": round(float(l), 4),
                    "volume": int(v),
                    "return": round(ret, 6),
                    "hl_range": round(hl, 6),
                    "vwap": round(float(vwap), 4),
                    "intraday_vol": round(float(vol_intraday) if pd.notna(vol_intraday) else 0.0, 6),
                }
            )
        feats[ticker] = per_day
    return feats


def call_llm(prompt: str, cfg: LLMConfig):
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You design concise, testable trading rules. Respond ONLY with JSON following the schema.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(cfg.base_url, headers=headers, json=payload, timeout=cfg.timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def main():
    parser = argparse.ArgumentParser(description="Generate LLM rule JSON from sampled OHLCV features.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--sample-days", type=int, default=3, help="Number of earliest days per ticker to include.")
    parser.add_argument("--out", default="llm_rules.json", help="Output JSON file path.")
    args = parser.parse_args()

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("LLM_API_KEY is not set; aborting.")
        return
    cfg = LLMConfig(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        cache_path=os.getenv("LLM_CACHE", ".llm_cache.json"),
        timeout=20,
    )

    engine = create_engine(DB_URL)
    df = fetch_data(engine, [t.upper() for t in args.tickers], args.start, args.end)
    if df.empty:
        print("No data returned for requested tickers/date range.")
        return
    feats = build_features(df, args.sample_days)

    schema = {
        "rules": [
            {
                "name": "string",
                "timeframe": "1m|5m",
                "entries": ["condition string referencing fields like close, vwap, sma, macd_diff, atr"],
                "exits": ["condition string"],
                "params": {"stop_multiple": "e.g., 1.0 * atr", "target_rr": "e.g., 2.0", "max_hold_minutes": 30},
                "notes": "short rationale",
            }
        ],
        "meta": {"assumptions": "any global notes", "risk_scale": "1-5"},
    }

    prompt = (
        "Given these sampled daily features per ticker, propose 1-3 concise rule sets as JSON. "
        "Keep conditions numeric and testable (e.g., close > vwap, macd_diff > 0, hl_range < 0.02). "
        "Do not include any free-form text outside JSON.\n"
        f"Feature summary: {json.dumps(feats)[:5000]}\n"
        f"Respond with JSON matching this schema: {json.dumps(schema)}"
    )

    try:
        result = call_llm(prompt, cfg)
    except Exception as exc:
        print(f"LLM call failed: {exc}")
        return

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote LLM rules to {out_path}")


if __name__ == "__main__":
    main()
