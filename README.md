# MooBot (MooMoo / Futu Trading Bot)

Python trading bot built on the MooMoo OpenAPI with an example intraday strategy (MACD + VWAP + SMA), optional Discord alerts, and a local paper-trading simulator. Forked from LukeWang01/WallTrading-Bot-MooMoo-Futu with additional utilities and a richer strategy template.

---

## Features

- Intraday strategy using 1m/5m MACD/VWAP/SMA signals (`strategies/your_strategy.py`)
- Works with MooMoo/Futu OpenD for SIMULATE or REAL environments
- Local simulator wallet (json-backed) for paper trading without a broker connection
- Discord webhook trade notifications and a slash-command bot (`/positions`, `/pause`, `/resume`)
- Order history persistence (`order_history.json`) and runtime logging (`app_running.log`)
- Pause/resume toggle persisted in `env/bot_state.json`

## Project Structure

- `TradingBOT.py` — main loop, scheduling, trading context, local simulator
- `strategies/your_strategy.py` — shared MACD/VWAP/SMA strategy used by both backtester and trade bot
- `discord_bot.py` — optional Discord slash command bot (positions, pause/resume)
- `discord_notification/discord_notify_webhook.py` — webhook helper for trade alerts
- `utils/` — time utilities, logging, bot state, optional email/sound helpers
- `order_history.json` — saved trades; `env/local_sim_wallet.json` — simulator state

## Prerequisites

- Python 3.9+ recommended
- MooMoo/Futu OpenD installed and running (<https://www.moomoo.com/download/OpenAPI>)
- MooMoo API docs: <https://openapi.moomoo.com/moomoo-api-doc/en/intro/intro.html>

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` in the project root (values optional unless using those features):

```bash
DISCORD_WEBHOOK_URL=...   # for webhook trade messages
DISCORD_BOT_TOKEN=...     # for the slash-command bot
MOOMOO_PW=...             # only needed for REAL trading to unlock trades
```

## Configure the bot

Key settings live near the top of `TradingBOT.py`:

- `MOOMOOOPEND_ADDRESS` / `MOOMOOOPEND_PORT` — match your OpenD host/port (default 127.0.0.1:11111/11112)
- `TRADING_ENVIRONMENT` — `TrdEnv.SIMULATE` (default) or `TrdEnv.REAL`
- `USE_LOCAL_SIM_WALLET` — `True` to bypass broker in SIMULATE mode and trade against the local wallet
- `SIM_STARTING_CASH` — starting balance for the local simulator
- `DISCORD_LOG_TRADES` — toggle webhook notifications for trades
- `TRADING_MARKET`, `FILL_OUTSIDE_MARKET_HOURS`, `SECURITY_FIRM` — broker/market options

Strategy defaults (`strategy/Your_Strategy.py`):

- Tickers: `["AMD", "TSLA", "MU"]`
- Size per ticker set in `self.trading_qty`
- Signals: 5m MACD cross + 1m VWAP + 5m SMA filters
- Data: 1m bars from yfinance, resampled to 5m for indicators

## Run

1) Start MooMoo/Futu OpenD and set the port to match `MOOMOOOPEND_PORT` (11111 or 11112 by default).
2) From the project root:

```bash
python TradingBOT.py
```

- The scheduler checks the strategy every minute at `:05`.
- When `USE_LOCAL_SIM_WALLET=True` and `TRADING_ENVIRONMENT=SIMULATE`, orders are filled instantly in `env/local_sim_wallet.json` and priced with yfinance.

## Discord integrations

- Webhook alerts: enable `DISCORD_LOG_TRADES=True` and set `DISCORD_WEBHOOK_URL`.
- Slash command bot:
  
  ```bash
  DISCORD_BOT_TOKEN=your_token python discord_bot.py
  ```

  Commands: `/positions`, `/pause`, `/resume` (pause state stored in `env/bot_state.json`).

## Customize your strategy

- Edit `strategies/your_strategy.py`:
  - Adjust indicator windows or replace the signal logic in `Strategy.generate_trades`
  - Update default sizes in `TRADING_QTY` or pass sizes at call time
- Order history is appended to `order_history.json`; logs go to `app_running.log`.

## Shared Strategy (backtest & live)

`strategies/your_strategy.py` exposes a shared MACD-driven `Strategy` class so both the backtester and trade bot use the same decision logic.

- Core API:
  - `StrategyConfig(macd_diff_pct=..., macd_diff_lookback=..., macd_loose=..., rule_fns=[...], cost_per_share=..., slippage_bps=...)`
  - `Strategy.generate_trades(ticker, g_drive, one_min, info, qty, stats) -> (trades, ending_pos, cash_delta)`
    - `g_drive`: indicator DataFrame on the driving timeframe (5m by default, 1m if `--one-min`)
    - `one_min`: 1m DataFrame with `vwap_1m`, `ret`, `intraday_vol` already computed
    - `info`: optional dict from LLM gating (sentiment/do_not_trade/risk_scale)
    - `qty`: size to trade (after any scaling)
    - `stats`: mutable counters for debug (bull/bear crosses, filter_pass, skips)
- Backtester usage (already wired):
  - Env: `WALLET_BACKTEST` sets starting cash (default 2000); `DB_URL` sets data source.
  - Example:
    ```bash
    ./.venv/bin/python scripts/backtest_strategy.py \
      --tickers NVDA --start 2025-06-01 --end 2025-11-27 \
      --one-min \
      --macd-fast 12 --macd-slow 26 --macd-signal 9 \
      --macd-loose --macd-diff-pct 0 --macd-diff-lookback 0 \
      --rules-json llm_rules.json --default-qty 1 \
      --cost-per-share 0.005 --slippage-bps 1
    ```
  - Flags:
    - `--tickers`, `--start`, `--end` (end is exclusive)
    - Timeframe/indicators: `--one-min`, `--macd-fast/--macd-slow/--macd-signal`, `--macd-loose`, `--macd-diff-pct`, `--macd-diff-lookback`
    - Sizing: `--default-qty`, `--qty TICKER=QTY`
    - LLM/rules: `--use-llm`, `--llm-cache`, `--rules-json`
    - Costs: `--cost-per-share`, `--slippage-bps`
    - Plots: defaults to interactive HTML, auto-opens; `--debug-plot-limit` caps points, `--debug-plot-path` sets filename (ticker defaults to first ticker)
- Trade bot integration (outline):
  - Import `Strategy, StrategyConfig`, build once at startup with your params/costs.
  - Feed live bars into DataFrames matching the backtester:
    - 1m frame with `ts` (tz-aware), `open/high/low/close/volume`, plus `vwap_1m`, `ret`, `intraday_vol`
    - Driving frame (5m or 1m) with `macd`, `macd_signal`, `sma_*`
  - Call `generate_trades(ticker, g_drive, one_min, info, qty, stats)` on bar close; translate returned `trades` into broker orders.
  - Keep I/O (DB, broker, Discord) outside the strategy; only pass plain data and sizes in.

Tip: If you add new signals, keep them contained in `Strategy` so both environments stay in sync. When you extend the context (e.g., new fields for rules), be sure to populate them in both the live feed and backtester preprocessing.

## Safety and notes

- REAL trading requires `TRADING_ENVIRONMENT=TrdEnv.REAL` and a valid `MOOMOO_PW` to unlock trades.
- The local simulator is for testing only and will not match broker execution quality.
- Ensure compliance with MooMoo/Futu terms and your local regulations before live use.

## License

MIT License (see `LICENSE`).
