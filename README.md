# MooBot (MooMoo / Futu Trading Bot)

Python trading bot built on the MooMoo OpenAPI with an example intraday strategy (MACD + VWAP + SMA), optional Discord alerts, and a local paper-trading simulator. Forked from LukeWang01/WallTrading-Bot-MooMoo-Futu with additional utilities and a richer strategy template.

---

## Features

- Intraday strategy using 1m VWAP and 5m MACD + SMA signals (`strategy/Your_Strategy.py`)
- Works with MooMoo/Futu OpenD for SIMULATE or REAL environments
- Local simulator wallet (json-backed) for paper trading without a broker connection
- Discord webhook trade notifications and a slash-command bot (`/positions`, `/pause`, `/resume`)
- Order history persistence (`order_history.json`) and runtime logging (`app_running.log`)
- Pause/resume toggle persisted in `env/bot_state.json`

## Project Structure

- `TradingBOT.py` — main loop, scheduling, trading context, local simulator
- `strategy/Your_Strategy.py` — configurable MACD/VWAP/SMA strategy; edit to build your own
- `strategy/Strategy.py` — base class used by all strategies
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

- Edit `strategy/Your_Strategy.py`:
  - Update `stock_trading_list` and `trading_qty`
  - Adjust indicator windows or replace the signal logic in `strategy_decision`
  - Reuse `strategy_make_trade` to route orders through the Trader (broker or local sim)
- Order history is appended to `order_history.json`; logs go to `app_running.log`.

## Safety and notes

- REAL trading requires `TRADING_ENVIRONMENT=TrdEnv.REAL` and a valid `MOOMOO_PW` to unlock trades.
- The local simulator is for testing only and will not match broker execution quality.
- Ensure compliance with MooMoo/Futu terms and your local regulations before live use.

## License

MIT License (see `LICENSE`).
