"""
Shared strategy logic usable by both backtester and live trade bot.

Provides indicator-driven signal generation so only one strategy class
needs to be maintained across environments.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Trade:
    timestamp: pd.Timestamp
    ticker: str
    side: str  # BUY/SELL
    price: float
    qty: int


@dataclass
class StrategyConfig:
    macd_diff_pct: float = 0.0
    macd_diff_lookback: int = 0
    macd_loose: bool = False
    rule_fns: Optional[List] = None
    cost_per_share: float = 0.0  # fixed fee per share
    slippage_bps: float = 0.0    # bps cost on notional


class Strategy:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def generate_trades(
        self,
        ticker: str,
        g_drive: pd.DataFrame,
        one_min: pd.DataFrame,
        info: Optional[Dict[str, Any]],
        qty: int,
        stats: Dict[str, Dict[str, int]],
    ) -> Tuple[List[Trade], int, float]:
        """
        Core MACD-driven trade generation.
        Returns (trades, ending_position, cash_delta).
        """
        trades: List[Trade] = []
        cash = 0.0
        pos = 0

        g_drive = g_drive.sort_values("ts").reset_index(drop=True)
        macd_col = "macd"
        macd_sig_col = "macd_signal"
        sma_col = "sma_1m" if "sma_1m" in g_drive.columns else "sma_5m"

        for i in range(1, len(g_drive)):
            row = g_drive.iloc[i]
            prev = g_drive.iloc[i - 1]
            macd_bull = (row[macd_col] > row[macd_sig_col]) and (prev[macd_col] <= prev[macd_sig_col])
            macd_bear = (row[macd_col] < row[macd_sig_col]) and (prev[macd_col] >= prev[macd_sig_col])
            if macd_bull:
                stats[ticker]["bull_cross"] += 1
            if macd_bear:
                stats[ticker]["bear_cross"] += 1

            ts = row["ts"]
            one_row = one_min[one_min.index <= ts].iloc[-1]
            price = one_row["close"]

            macd_diff = row[macd_col] - row[macd_sig_col]
            macd_diff_ok = True
            if not self.config.macd_loose and self.config.macd_diff_pct > 0:
                macd_diff_ok = macd_diff >= self.config.macd_diff_pct * float(price)
            macd_rising = True
            if not self.config.macd_loose and self.config.macd_diff_lookback > 0 and i >= self.config.macd_diff_lookback:
                diffs = g_drive.loc[i - self.config.macd_diff_lookback : i, macd_col] - g_drive.loc[
                    i - self.config.macd_diff_lookback : i, macd_sig_col
                ]
                macd_rising = all(diffs.iloc[j] > diffs.iloc[j - 1] for j in range(1, len(diffs)))

            rule_ok = True
            if self.config.rule_fns:
                high_val = float(row.get("high", one_row.get("high", price)))
                low_val = float(row.get("low", one_row.get("low", price)))
                ctx = {
                    "close": float(price),
                    "vwap": float(one_row.get("vwap_1m", price)),
                    "sma": float(row.get(sma_col, price)),
                    "macd": float(row.get(macd_col, 0.0)),
                    "macd_signal": float(row.get(macd_sig_col, 0.0)),
                    "macd_diff": float(macd_diff),
                    "high": high_val,
                    "low": low_val,
                    "hl_range": (high_val - low_val) / float(price) if price else 0.0,
                    "intraday_vol": float(one_row.get("intraday_vol", 0.0)),
                    "ret": float(one_row.get("ret", 0.0)),
                }
                try:
                    rule_ok = all(fn(ctx) for fn in self.config.rule_fns)
                except Exception:
                    rule_ok = False

            if macd_bull and macd_diff_ok and macd_rising and rule_ok:
                if info and (info.get("sentiment", 0) < -0.25):
                    stats[ticker]["llm_skip"] += 1
                    continue
                stats[ticker]["filter_pass"] += 1
                trade_qty = qty
                exec_price = float(price)
                fee = trade_qty * self.config.cost_per_share + trade_qty * exec_price * (self.config.slippage_bps / 10000.0)
                trades.append(Trade(ts, ticker, "BUY", exec_price, trade_qty))
                pos += trade_qty
                cash -= trade_qty * exec_price
                cash -= fee

            if macd_bear:
                if pos <= 0:
                    continue
                sell_qty = pos
                exec_price = float(price)
                fee = sell_qty * self.config.cost_per_share + sell_qty * exec_price * (self.config.slippage_bps / 10000.0)
                trades.append(Trade(ts, ticker, "SELL", exec_price, sell_qty))
                pos -= sell_qty
                cash += sell_qty * exec_price
                cash -= fee

        return trades, pos, cash
