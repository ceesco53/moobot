#!/usr/bin/env python
"""
Quick-and-dirty backtester for Your_Strategy (MACD 5m + SMA 5m + VWAP 1m)
using 1m bars from the ClickHouse `ohlcv` table.

Requires:
  pip install clickhouse-connect pandas ta python-dotenv plotly

Usage examples:
  DB_URL=clickhouse://moobot:moobot@localhost:8123/marketdata \
  python scripts/backtest_strategy.py --tickers AMD TSLA MU --start 2024-01-01 --end 2024-01-31
"""

import argparse
import ast
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# Ensure project root is on sys.path for local imports when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.volume import VolumeWeightedAveragePrice
from strategies.your_strategy import Strategy, StrategyConfig, Trade
from llm_adapter import LLMConfig, get_llm_annotation
from dotenv import load_dotenv
from clickhouse_connect import get_client
from urllib.parse import urlparse


# Strategy parameters (mirror Your_Strategy)
MACD_FAST_DEFAULT = 12
MACD_SLOW_DEFAULT = 26
MACD_SIGNAL_DEFAULT = 9
SMA_WINDOW_5M = 20
VWAP_WINDOW_1M = 390  # up to full session
TRADING_QTY = {"AMD": 2, "TSLA": 1, "MU": 2, "NVDA": 1}  # override with --qty if desired

# Connection
# Load .env if present so WALLET_BACKTEST/DB_URL/etc are picked up without manual export
load_dotenv()

DB_URL = os.getenv("DB_URL", "clickhouse://moobot:moobot@localhost:8123/marketdata")
STARTING_CASH = int(os.getenv("WALLET_BACKTEST", "2000"))


def _ch_client():
    parsed = urlparse(DB_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8123
    username = parsed.username or "moobot"
    password = parsed.password or "moobot"
    database = (parsed.path or "").lstrip("/") or "marketdata"
    secure = parsed.scheme == "https"
    return get_client(host=host, port=port, username=username, password=password, database=database, secure=secure)


def load_1m(client, tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Fetch 1m bars from ClickHouse."""
    sql = """
        SELECT ticker,
               window_start AS ts_utc,
               open, high, low, close, volume
        FROM ohlcv
        WHERE ticker IN %(tickers)s
          AND window_start >= parseDateTime64BestEffort(%(start)s, 3)
          AND window_start < parseDateTime64BestEffort(%(end)s, 3)
        ORDER BY ticker, window_start
    """
    df = client.query_df(sql, parameters={"tickers": tickers, "start": start, "end": end})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_convert("America/New_York")
    df = df.drop(columns=["ts_utc"])
    return df


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample per ticker to 5m OHLCV."""
    pieces = []
    for ticker, g in df_1m.groupby("ticker"):
        g = g.set_index("ts").sort_index()
        r = g.resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        r["ticker"] = ticker
        r = r.reset_index()
        pieces.append(r)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def compute_indicators(df_1m: pd.DataFrame, df_5m: pd.DataFrame, macd_fast: int, macd_slow: int, macd_signal: int):
    """Add VWAP (1m) and MACD/SMA (5m)."""
    # 1m VWAP rolling up to session
    df_1m = df_1m.copy()
    df_1m["session"] = df_1m["ts"].dt.date
    vwap_frames = []
    for (ticker, session), g in df_1m.groupby(["ticker", "session"]):
        vw = VolumeWeightedAveragePrice(
            high=g["high"], low=g["low"], close=g["close"], volume=g["volume"], window=min(VWAP_WINDOW_1M, len(g))
        )
        g = g.copy()
        g["vwap_1m"] = vw.volume_weighted_average_price()
        vwap_frames.append(g)
    df_1m = pd.concat(vwap_frames, ignore_index=True) if vwap_frames else df_1m

    # 5m MACD + SMA
    df_5m = df_5m.copy()
    ind_frames = []
    for ticker, g in df_5m.groupby("ticker"):
        macd_ind = MACD(
            close=g["close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal
        )
        g = g.copy()
        g["macd"] = macd_ind.macd()
        g["macd_signal"] = macd_ind.macd_signal()
        g["macd_hist"] = macd_ind.macd_diff()
        g["sma_5m"] = SMAIndicator(close=g["close"], window=SMA_WINDOW_5M).sma_indicator()
        ind_frames.append(g)
    df_5m = pd.concat(ind_frames, ignore_index=True) if ind_frames else df_5m

    return df_1m, df_5m


def compute_indicators_1m_only(df_1m: pd.DataFrame, macd_fast: int, macd_slow: int, macd_signal: int) -> pd.DataFrame:
    """Compute VWAP + MACD + SMA directly on 1m bars (for --one-min mode)."""
    df_1m = df_1m.copy()
    df_1m["session"] = df_1m["ts"].dt.date
    vwap_frames = []
    for (ticker, session), g in df_1m.groupby(["ticker", "session"]):
        vw = VolumeWeightedAveragePrice(
            high=g["high"], low=g["low"], close=g["close"], volume=g["volume"], window=min(VWAP_WINDOW_1M, len(g))
        )
        g = g.copy()
        g["vwap_1m"] = vw.volume_weighted_average_price()
        vwap_frames.append(g)
    df_1m = pd.concat(vwap_frames, ignore_index=True) if vwap_frames else df_1m

    ind_frames = []
    for ticker, g in df_1m.groupby("ticker"):
        macd_ind = MACD(
            close=g["close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal
        )
        g = g.copy()
        g["macd"] = macd_ind.macd()
        g["macd_signal"] = macd_ind.macd_signal()
        g["macd_hist"] = macd_ind.macd_diff()
        g["sma_1m"] = SMAIndicator(close=g["close"], window=SMA_WINDOW_5M).sma_indicator()
        ind_frames.append(g)
    return pd.concat(ind_frames, ignore_index=True) if ind_frames else pd.DataFrame()


def maybe_plot_debug(
    ticker: str,
    df_1m: pd.DataFrame,
    df_drive: pd.DataFrame,
    trades_df: pd.DataFrame,
    out_path: str,
    one_min_mode: bool,
    limit: int,
    interactive: bool = False,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> None:
    """Render a quick plot showing proximity to triggers (always interactive HTML)."""
    try:
        from plotly.subplots import make_subplots  # type: ignore
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        print("plotly not installed; skipping interactive plot. pip install plotly")
        return

    p = df_1m[df_1m["ticker"] == ticker].copy()
    if p.empty:
        print(f"No 1m data for {ticker}; skipping debug plot.")
        return
    d = df_drive[df_drive["ticker"] == ticker].copy()
    if d.empty:
        print(f"No driving frame data for {ticker}; skipping debug plot.")
        return

    sma_col = "sma_1m" if one_min_mode else "sma_5m"
    if sma_col not in d.columns:
        # Fall back to whichever SMA is present, or compute a quick rolling mean to avoid crashing the plot.
        alt = "sma_5m" if sma_col == "sma_1m" else "sma_1m"
        if alt in d.columns:
            sma_col = alt
        else:
            d = d.sort_values("ts")
            d[sma_col] = d["close"].rolling(window=SMA_WINDOW_5M, min_periods=1).mean()

    d = d[["ts", sma_col, "macd", "macd_signal"]]
    df_plot = p.merge(d, on="ts", how="left").sort_values("ts")
    # Handle duplicate column suffixes from the merge
    if sma_col not in df_plot.columns:
        for candidate in (f"{sma_col}_y", f"{sma_col}_x"):
            if candidate in df_plot.columns:
                df_plot[sma_col] = df_plot[candidate]
                break
    # MACD columns may also get suffixed; try to normalize them, otherwise skip plotting.
    if "macd" not in df_plot.columns:
        for candidate in ("macd_y", "macd_x"):
            if candidate in df_plot.columns:
                df_plot["macd"] = df_plot[candidate]
                break
    if "macd_signal" not in df_plot.columns:
        for candidate in ("macd_signal_y", "macd_signal_x"):
            if candidate in df_plot.columns:
                df_plot["macd_signal"] = df_plot[candidate]
                break
    if "macd" not in df_plot.columns or "macd_signal" not in df_plot.columns:
        print("MACD columns missing in plot data; skipping debug plot.")
        return
    df_plot[sma_col] = df_plot[sma_col].ffill()
    df_plot["macd"] = df_plot["macd"].ffill()
    df_plot["macd_signal"] = df_plot["macd_signal"].ffill()

    # Make timestamps naive for plotting
    for col in ["ts"]:
        if pd.api.types.is_datetime64_any_dtype(df_plot[col]) and df_plot[col].dt.tz is not None:
            df_plot[col] = df_plot[col].dt.tz_convert(None)

    # Apply start/end bounds if provided
    if start is not None:
        df_plot = df_plot[df_plot["ts"] >= start]
    if end is not None:
        df_plot = df_plot[df_plot["ts"] < end]
    if len(df_plot) > limit:
        df_plot = df_plot.tail(limit)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.35, 0.25])
    fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["close"], name="close", line=dict(color="black")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["vwap_1m"], name="vwap_1m", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot[sma_col], name=sma_col, line=dict(color="blue")), row=1, col=1)
    if trades_df is not None and not trades_df.empty:
        tt = trades_df[trades_df["ticker"] == ticker].copy()
        if not tt.empty:
            if pd.api.types.is_datetime64_any_dtype(tt["timestamp"]) and tt["timestamp"].dt.tz is not None:
                tt["timestamp"] = tt["timestamp"].dt.tz_convert(None)
            for side, color, symbol in [("BUY", "green", "triangle-up"), ("SELL", "red", "triangle-down"), ("FLAT", "gray", "circle")]:
                sub = tt[tt["side"] == side]
                if not sub.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sub["timestamp"],
                            y=sub["price"],
                            mode="markers",
                            marker=dict(color=color, size=8, symbol=symbol),
                            name=side,
                        ),
                        row=1,
                        col=1,
                    )
    fig.add_trace(
        go.Scatter(
            x=df_plot["ts"],
            y=df_plot["macd"] - df_plot["macd_signal"],
            name="macd - signal",
            line=dict(color="purple"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_plot["ts"], y=df_plot["close"] - df_plot["vwap_1m"], name="close - vwap_1m", line=dict(color="orange")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_plot["ts"], y=df_plot["close"] - df_plot[sma_col], name=f"close - {sma_col}", line=dict(color="blue")),
        row=2,
        col=1,
    )
    # Equity curve
    equity_ts, equity_vals = compute_equity_curve(df_plot["ts"], df_plot["close"], trades_df, ticker, STARTING_CASH)
    fig.add_trace(go.Scatter(x=equity_ts, y=equity_vals, name="equity", line=dict(color="teal")), row=3, col=1)

    if trades_df is not None and not trades_df.empty and "side" in trades_df.columns:
        total_trades = len(trades_df)
        buy_count = len(trades_df[trades_df["side"] == "BUY"])
        sell_count = len(trades_df[trades_df["side"] == "SELL"])
    else:
        total_trades = buy_count = sell_count = 0
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.02,
        xanchor="right",
        yanchor="bottom",
        text=f"Trades: {total_trades} | Buys: {buy_count} | Sells: {sell_count}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="right",
        bgcolor="rgba(255,255,255,0.8)",
    )

    fig.update_layout(height=900, width=1200, title=f"Debug plot for {ticker}", hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Proximity", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=3, col=1)
    if out_path.lower().endswith(".png"):
        out_path = out_path[:-4] + "html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved interactive debug plot to {out_path}")


class SafeEvalVisitor(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Eq,
        ast.NotEq,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
    )

    def visit(self, node):
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
        return super().visit(node)


def make_condition_fn(expr: str, allowed_names: List[str]) -> Callable[[Dict[str, Any]], bool]:
    tree = ast.parse(expr, mode="eval")
    SafeEvalVisitor().visit(tree)
    code = compile(tree, "<rule>", "eval")

    def fn(ctx: Dict[str, Any]) -> bool:
        for k in ctx:
            if not isinstance(ctx[k], (int, float, bool)):
                ctx[k] = float(ctx[k])
        for name in tree.body._fields:
            pass  # noop to appease lint
        bad = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in allowed_names]
        if bad:
            raise ValueError(f"Disallowed identifiers in rule: {bad}")
        return bool(eval(code, {"__builtins__": {}}, ctx))

    return fn


def load_rules_file(path: Path):
    raw = path.read_text(encoding="utf-8")
    text = raw.strip()
    # Strip code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception as exc:
        snippet = text[:200].replace("\n", " ")
        raise ValueError(f"Failed to parse JSON ({exc}); snippet: {snippet}")


def compute_equity_curve(ts_series, close_series, trades_df, ticker: str, starting_cash: float):
    """Compute equity curve (cash + marked-to-market position) for a single ticker using bar closes."""
    ts_list = list(ts_series)
    closes = list(close_series)
    if not ts_list:
        return [], []
    if trades_df is None or trades_df.empty or "ticker" not in trades_df.columns:
        return ts_list, [float(starting_cash)] * len(ts_list)
    cash = float(starting_cash)
    pos = 0
    trades = trades_df[trades_df["ticker"] == ticker].copy()
    if trades.empty:
        return ts_list, [float(starting_cash)] * len(ts_list)
    if pd.api.types.is_datetime64_any_dtype(trades["timestamp"]) and trades["timestamp"].dt.tz is not None:
        trades["timestamp"] = trades["timestamp"].dt.tz_convert(None)
    trades = trades.sort_values("timestamp")
    equity = []
    ti = 0
    for ts, close in zip(ts_list, closes):
        while not trades.empty and ti < len(trades) and trades.iloc[ti]["timestamp"] <= ts:
            tr = trades.iloc[ti]
            if tr["side"] == "BUY":
                pos += tr["qty"]
                cash -= tr["qty"] * float(tr["price"])
            elif tr["side"] in ("SELL", "FLAT"):
                pos -= tr["qty"]
                cash += tr["qty"] * float(tr["price"])
            ti += 1
        equity.append(cash + pos * float(close))
    return ts_list, equity


def backtest(
    strategy: Strategy,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    qty_override: Dict[str, int],
    llm_config: Optional[LLMConfig] = None,
    one_min_mode: bool = False,
    default_qty: int = 1,
) -> Dict:
    """Run the MACD/VWAP/SMA rules and return trades + equity."""
    trades: List[Trade] = []
    positions = {}
    cash = float(STARTING_CASH)

    # Index 1m data for quick lookup of latest bar at or before a timestamp
    one_min_map = {}
    for t, g in df_1m.groupby("ticker"):
        g = g.sort_values("ts").copy()
        g["session"] = g["ts"].dt.date
        # Session open and returns relative to session open
        g["session_open"] = g.groupby("session")["open"].transform("first")
        g["ret"] = (g["close"] - g["session_open"]) / g["session_open"]
        # Rolling intraday volatility of 1m returns (pct change), 30-bar window
        g["ret1"] = g["close"].pct_change()
        g["intraday_vol"] = g["ret1"].rolling(window=30, min_periods=5).std().fillna(0.0)
        g = g.drop(columns=["ret1"])
        one_min_map[t] = g.set_index("ts").sort_index()

    # Optional LLM annotations per ticker
    llm_annotations: Dict[str, Dict] = {}
    llm_sources: Dict[str, str] = {}
    stats: Dict[str, Dict[str, int]] = {}
    if llm_config:
        for ticker, g5 in df_5m.groupby("ticker"):
            one_min = one_min_map.get(ticker)
            one_min_df = one_min.reset_index() if one_min is not None else pd.DataFrame()
            ann, source = get_llm_annotation(ticker, g5, one_min_df, llm_config)
            llm_annotations[ticker] = ann
            llm_sources[ticker] = source

    # Choose driving frame: 5m or 1m depending on mode
    driving = df_1m if one_min_mode else df_5m

    for ticker, g_drive in driving.groupby("ticker"):
        qty = qty_override.get(ticker, TRADING_QTY.get(ticker, default_qty))
        if ticker not in stats:
            stats[ticker] = {"bull_cross": 0, "bear_cross": 0, "filter_pass": 0, "llm_skip": 0, "qty_skip": 0}
        if qty <= 0:
            stats[ticker]["qty_skip"] += 1
            continue
        info = llm_annotations.get(ticker)
        if info and info.get("do_not_trade"):
            stats[ticker]["llm_skip"] += 1
            continue

        risk_scale = int(info.get("risk_scale", 3)) if info else 3
        risk_scale = max(1, min(5, risk_scale))
        scaled_qty = max(1, int(round(qty * (risk_scale / 3.0)))) if qty > 0 else 0

        one_min = one_min_map[ticker]
        trades_out, pos, end_cash = strategy.generate_trades(
            ticker=ticker, g_drive=g_drive, one_min=one_min, info=info, qty=scaled_qty, stats=stats, starting_cash=cash
        )
        trades.extend(trades_out)
        positions[ticker] = pos
        cash = end_cash

    # Mark-to-market any residual positions at last price
    for ticker, pos in positions.items():
        if pos <= 0:
            continue
        last_price = one_min_map[ticker]["close"].iloc[-1]
        cash += pos * float(last_price)
        trades.append(Trade(one_min_map[ticker].index[-1], ticker, "FLAT", float(last_price), pos))
        positions[ticker] = 0

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    trades_df = trades_df.sort_values("timestamp") if not trades_df.empty else trades_df
    return {
        "cash_pnl": cash,
        "trades": trades_df,
        "llm_annotations": llm_annotations,
        "llm_sources": llm_sources,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest Your_Strategy using Postgres ohlcv data.")
    parser.add_argument("--tickers", nargs="+", default=["AMD", "TSLA", "MU"])
    parser.add_argument("--start", required=True, help="Start datetime (e.g., 2024-01-01)")
    parser.add_argument("--end", required=True, help="End datetime (exclusive)")
    parser.add_argument("--qty", nargs="*", help="Override qty as TICKER=QTY (e.g., AMD=5 TSLA=1)")
    parser.add_argument("--default-qty", type=int, default=1, help="Default qty for tickers not in TRADING_QTY.")
    parser.add_argument("--use-llm", action="store_true", help="Annotate tickers with an LLM and gate trades.")
    parser.add_argument("--llm-cache", default=os.getenv("LLM_CACHE", ".llm_cache.json"), help="Path to LLM cache file.")
    parser.add_argument(
        "--loose-filters",
        action="store_true",
        help="Allow VWAP OR SMA confirmation (instead of requiring both) to see where trades are filtered.",
    )
    parser.add_argument(
        "--one-min",
        action="store_true",
        help="Run strategy on 1m candles (MACD/SMA/VWAP on 1m) instead of resampled 5m.",
    )
    parser.add_argument(
        "--debug-plot-ticker",
        help="If set, render a debug plot for this ticker showing price/VWAP/SMA and proximity to triggers.",
    )
    parser.add_argument("--debug-plot-path", default="debug_plot.html", help="Path to save the debug plot.")
    parser.add_argument("--debug-plot-limit", type=int, default=10000000, help="Max points to plot (tail).")
    parser.add_argument(
        "--debug-plot-interactive",
        action="store_true",
        default=True,
        help="Save debug plot as interactive HTML (plotly) instead of static PNG (matplotlib).",
    )
    parser.add_argument(
        "--macd-loose",
        action="store_true",
        help="Ignore MACD diff magnitude and slope checks (just use the cross).",
    )
    parser.add_argument("--macd-fast", type=int, default=MACD_FAST_DEFAULT, help="MACD fast window.")
    parser.add_argument("--macd-slow", type=int, default=MACD_SLOW_DEFAULT, help="MACD slow window.")
    parser.add_argument("--macd-signal", type=int, default=MACD_SIGNAL_DEFAULT, help="MACD signal window.")
    parser.add_argument(
        "--macd-diff-pct",
        type=float,
        default=0.0,
        help="Minimum (macd - signal) as a fraction of price (e.g., 0.0008 for 0.08%%).",
    )
    parser.add_argument(
        "--macd-diff-lookback",
        type=int,
        default=0,
        help="Require macd diff to be monotonically rising over this many bars (e.g., 2).",
    )
    parser.add_argument(
        "--proximity-max-pct",
        type=float,
        default=None,
        help="Require price above VWAP/SMA but within this fraction of price (e.g., 0.005 for 0.5%%).",
    )
    parser.add_argument("--rules-json", help="Path to LLM-generated rules JSON.")
    parser.add_argument("--cost-per-share", type=float, default=0.0, help="Per-share commission/fee for backtests.")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage cost in basis points of notional.")
    parser.add_argument("--min-qty", type=int, default=1, help="Minimum tradable size after downsizing.")
    parser.add_argument(
        "--cooldown-minutes", type=int, default=0, help="Minimum minutes between successful entries per ticker."
    )
    parser.add_argument("--take-profit-pct", type=float, default=0.0, help="Take profit barrier as fraction (e.g., 0.01).")
    parser.add_argument("--stop-loss-pct", type=float, default=0.0, help="Stop loss barrier as fraction (e.g., 0.005).")
    parser.add_argument(
        "--max-holding-minutes", type=int, default=0, help="Time barrier in minutes (0 means no time barrier)."
    )
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for per-ticker backtest work.")
    parser.add_argument("--wf-test-days", type=int, default=0, help="Walk-forward test window size in days (0 to disable).")
    parser.add_argument(
        "--wf-embargo-days", type=int, default=0, help="Days to skip between walk-forward folds to avoid leakage."
    )
    args = parser.parse_args()

    qty_override = {}
    if args.qty:
        for item in args.qty:
            if "=" in item:
                sym, q = item.split("=", 1)
                qty_override[sym.upper()] = int(q)
    # Default debug plot ticker to the first requested ticker if not provided
    if args.tickers:
        args.debug_plot_ticker = args.tickers[0]

    llm_config = None
    if args.use_llm:
        api_key = os.getenv("LLM_API_KEY", "")
        if not api_key:
            print("LLM requested but LLM_API_KEY is not set; continuing without LLM.")
        else:
            llm_config = LLMConfig(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1/chat/completions"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
                cache_path=args.llm_cache,
            )

    try:
        client = _ch_client()
        df_1m = load_1m(client, [t.upper() for t in args.tickers], args.start, args.end)
    except Exception as exc:
        print(f"Could not query ClickHouse at {DB_URL}: {exc}")
        return
    if df_1m.empty:
        print("No data returned for requested tickers/date range.")
        return
    # Quick visibility into what we're processing
    counts = df_1m.groupby("ticker").size().to_dict()
    print(f"Loaded rows per ticker: {counts} for [{args.start}, {args.end})")
    day_counts = (
        df_1m.groupby(["ticker", df_1m["ts"].dt.date]).size().reset_index(name="rows").sort_values(["ticker", "ts"])
    )
    print(
        "Rows per day (first 10): "
        f"{day_counts.head(10).to_dict(orient='records')} ... total days={len(day_counts)}, "
        f"min/max rows={day_counts['rows'].min()}/{day_counts['rows'].max()}"
    )
    # Identify missing date gaps per ticker within the requested range
    start_date = pd.to_datetime(args.start).date()
    end_date = pd.to_datetime(args.end).date()
    for ticker, g in day_counts.groupby("ticker"):
        dates = sorted(g["ts"].tolist())
        gaps = []
        prev = start_date
        for d in dates:
            if (d - prev).days > 1:
                gaps.append((prev + pd.Timedelta(days=1), d - pd.Timedelta(days=1)))
            prev = d
        if (end_date - prev).days > 1:
            gaps.append((prev + pd.Timedelta(days=1), end_date - pd.Timedelta(days=1)))
        if gaps:
            gap_str = "; ".join([f"{g[0]}->{g[1]}" for g in gaps])
            print(f"Missing date ranges for {ticker}: {gap_str}")

    rule_fns = None
    if args.rules_json:
        try:
            rules_data = load_rules_file(Path(args.rules_json))
            rules = rules_data.get("rules") if isinstance(rules_data, dict) else None
            if rules and isinstance(rules, list):
                target_tf = "1m" if args.one_min else "5m"
                chosen = next((r for r in rules if r.get("timeframe") == target_tf), None)
                if chosen and chosen.get("entries"):
                    allowed = {"close", "vwap", "sma", "macd", "macd_signal", "macd_diff", "high", "low", "hl_range", "intraday_vol", "ret"}
                    parsed = []
                    for expr in chosen["entries"]:
                        expr_clean = expr.replace("return", "ret")
                        try:
                            parsed.append(make_condition_fn(expr_clean, list(allowed)))
                        except Exception as inner_exc:
                            print(f"Skipping rule '{expr}' due to error: {inner_exc}")
                    rule_fns = parsed if parsed else None
                    if rule_fns:
                        print(f"Loaded {len(rule_fns)} entry rule(s) from {args.rules_json} for timeframe {target_tf}.")
        except Exception as exc:
            print(f"Failed to load rules from {args.rules_json}: {exc}")

    strat = Strategy(
        StrategyConfig(
            macd_diff_pct=args.macd_diff_pct,
            macd_diff_lookback=args.macd_diff_lookback,
            macd_loose=args.macd_loose,
            rule_fns=rule_fns,
            cost_per_share=args.cost_per_share,
            slippage_bps=args.slippage_bps,
            min_qty=args.min_qty,
            cooldown_minutes=args.cooldown_minutes,
            take_profit_pct=args.take_profit_pct,
            stop_loss_pct=args.stop_loss_pct,
            max_holding_minutes=args.max_holding_minutes,
        )
    )

    def run_backtest_on_df(df_input: pd.DataFrame, tag: str = "full"):
        if df_input.empty:
            print(f"[{tag}] No data.")
            return None
        if args.one_min:
            df1 = compute_indicators_1m_only(df_input, args.macd_fast, args.macd_slow, args.macd_signal)
            df5 = pd.DataFrame()
        else:
            df5 = resample_5m(df_input)
            df1, df5 = compute_indicators(df_input, df5, args.macd_fast, args.macd_slow, args.macd_signal)
        res = backtest(
            strat,
            df1,
            df5,
            qty_override,
            llm_config=llm_config,
            one_min_mode=args.one_min,
            default_qty=args.default_qty,
        )
        trades = res["trades"]
        pnl = res["cash_pnl"]
        print(f"[{tag}] Cash PnL (no costs): {pnl:.2f} | trades: {len(trades)}")
        return res, df1, df5

    results, df_1m_ind, df_5m_ind = run_backtest_on_df(df_1m, tag="full")
    trades_df = results["trades"] if results else pd.DataFrame()

    if llm_config and results and results.get("llm_annotations"):
        print("\nLLM annotations:")
        for ticker, ann in results["llm_annotations"].items():
            src = results["llm_sources"].get(ticker, "?")
            print(
                f"{ticker}: sentiment={ann.get('sentiment')} risk={ann.get('risk_scale')} "
                f"do_not_trade={ann.get('do_not_trade')} note={ann.get('note')} ({src})"
            )

    if results and results.get("stats"):
        print("\nDebug stats (per ticker):")
        for ticker, st in results["stats"].items():
            print(
                f"{ticker}: bull_cross={st.get('bull_cross')} bear_cross={st.get('bear_cross')} "
                f"filter_pass={st.get('filter_pass')} llm_skips={st.get('llm_skip')} qty_skips={st.get('qty_skip')}"
            )

    if args.wf_test_days > 0:
        print("\nWalk-forward evaluation:")
        dates = sorted(df_1m["ts"].dt.date.unique())
        start_idx = 0
        fold = 1
        while start_idx < len(dates):
            test_dates = dates[start_idx : start_idx + args.wf_test_days]
            if not test_dates:
                break
            df_fold = df_1m[df_1m["ts"].dt.date.isin(test_dates)]
            res_fold, _, _ = run_backtest_on_df(df_fold, tag=f"fold{fold} {test_dates[0]}->{test_dates[-1]}")
            start_idx += args.wf_test_days + args.wf_embargo_days
            fold += 1

    if args.debug_plot_ticker and results:
        driving = df_1m_ind if args.one_min else df_5m_ind
        maybe_plot_debug(
            ticker=args.debug_plot_ticker.upper(),
            df_1m=df_1m_ind,
            df_drive=driving,
            trades_df=trades_df,
            out_path=args.debug_plot_path,
            one_min_mode=args.one_min,
            limit=args.debug_plot_limit,
            interactive=args.debug_plot_interactive,
            start=pd.to_datetime(args.start),
            end=pd.to_datetime(args.end),
        )
        # Auto-open interactive plot if saved as HTML
        try:
            if args.debug_plot_interactive and args.debug_plot_path.lower().endswith(".html"):
                webbrowser.open(f"file://{Path(args.debug_plot_path).resolve()}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
