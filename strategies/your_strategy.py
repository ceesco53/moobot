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
    min_qty: int = 1             # minimum tradable size after downsizing
    cooldown_minutes: int = 0    # minimum minutes between successful entries
    take_profit_pct: float = 0.0  # e.g., 0.01 for +1%
    stop_loss_pct: float = 0.0    # e.g., 0.005 for -0.5%
    max_holding_minutes: int = 0  # time barrier; 0 = no limit


class Strategy:
    def __init__(self, config: StrategyConfig):
        self.config = config

    def _evaluate_barrier(
        self, entry_ts: pd.Timestamp, entry_price: float, one_min: pd.DataFrame
    ) -> Optional[Tuple[pd.Timestamp, float, str]]:
        """Return first barrier hit (tp/sl/time) using 1m closes; None if insufficient data."""
        horizon_ts = None
        if self.config.max_holding_minutes > 0:
            horizon_ts = entry_ts + pd.Timedelta(minutes=self.config.max_holding_minutes)
        future = one_min[one_min.index > entry_ts]
        if horizon_ts is not None:
            future = future[future.index <= horizon_ts]
        if future.empty:
            return None
        tp = entry_price * (1 + self.config.take_profit_pct) if self.config.take_profit_pct > 0 else None
        sl = entry_price * (1 - self.config.stop_loss_pct) if self.config.stop_loss_pct > 0 else None
        for ts, row in future.iterrows():
            px = float(row["close"])
            if tp is not None and px >= tp:
                return ts, px, "TP"
            if sl is not None and px <= sl:
                return ts, px, "SL"
        # If horizon set, exit at last bar in horizon; else at last future bar
        last_ts = future.index[-1]
        last_px = float(future.iloc[-1]["close"])
        return last_ts, last_px, "TIME"

    def generate_trades(
        self,
        ticker: str,
        g_drive: pd.DataFrame,
        one_min: pd.DataFrame,
        info: Optional[Dict[str, Any]],
        qty: int,
        stats: Dict[str, Dict[str, int]],
        starting_cash: float,
    ) -> Tuple[List[Trade], int, float]:
        """
        Core MACD-driven trade generation.
        Returns (trades, ending_position, ending_cash).
        """
        trades: List[Trade] = []
        cash = float(starting_cash)
        pos = 0
        last_entry_ts: Optional[pd.Timestamp] = None

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
                if self.config.cooldown_minutes > 0 and last_entry_ts is not None:
                    delta_minutes = (ts - last_entry_ts).total_seconds() / 60.0
                    if delta_minutes < self.config.cooldown_minutes:
                        continue
                trade_qty = qty
                if trade_qty < self.config.min_qty:
                    stats[ticker]["qty_skip"] += 1
                    continue
                exec_price = float(price)
                fee = trade_qty * self.config.cost_per_share + trade_qty * exec_price * (self.config.slippage_bps / 10000.0)
                total_cost = trade_qty * exec_price + fee
                while total_cost > cash and trade_qty > self.config.min_qty:
                    trade_qty -= 1
                    fee = trade_qty * self.config.cost_per_share + trade_qty * exec_price * (self.config.slippage_bps / 10000.0)
                    total_cost = trade_qty * exec_price + fee
                if total_cost > cash or trade_qty < self.config.min_qty:
                    stats[ticker]["qty_skip"] += 1
                    continue
                trades.append(Trade(ts, ticker, "BUY", exec_price, trade_qty))
                pos += trade_qty
                cash -= total_cost
                last_entry_ts = ts
                # Triple-barrier exit if configured
                if (
                    self.config.take_profit_pct > 0
                    or self.config.stop_loss_pct > 0
                    or self.config.max_holding_minutes > 0
                ):
                    barrier = self._evaluate_barrier(ts, exec_price, one_min)
                    if barrier is None:
                        continue
                    exit_ts, exit_px, _ = barrier
                    fee_sell = trade_qty * self.config.cost_per_share + trade_qty * exit_px * (self.config.slippage_bps / 10000.0)
                    trades.append(Trade(exit_ts, ticker, "SELL", float(exit_px), trade_qty))
                    pos -= trade_qty
                    cash += trade_qty * float(exit_px)
                    cash -= fee_sell

            # If no barrier configured, use MACD bear to exit
            if (
                self.config.take_profit_pct == 0
                and self.config.stop_loss_pct == 0
                and self.config.max_holding_minutes == 0
            ):
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
