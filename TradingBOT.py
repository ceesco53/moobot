# Dev
# 03/29/2024
# LukeLab for LookAtWallStreet
# Version 1.0
# Programming Trading based on MooMoo API/OpenD

"""
# updated: 11/17/2024, final version for open source only
# Version 2.0
# for more info, please visit: https://www.patreon.com/LookAtWallStreet
"""

# MooMoo API Documentation, English:
# https://openapi.moomoo.com/moomoo-api-doc/en/intro/intro.html
# 官方文档，中文:
# https://openapi.moomoo.com/moomoo-api-doc/intro/intro.html

from moomoo import (
    RET_OK,
    TrdEnv,
    TrdSide,
    OrderType,
    SecurityFirm,
    TrdMarket,
    OpenSecTradeContext,
    TimeInForce,
    Session,
)
import schedule
import os
import json
import time
import pandas as pd
import yfinance as yf
import dotenv
from discord_notification.discord_notify_webhook import send_webhook_message

from strategy.Your_Strategy import Your_Strategy
from utils.dataIO import get_current_time, print_current_time, logging_info
from utils.time_tool import is_market_and_extended_hours, is_trading_day
from utils.bot_state import is_bot_paused, set_bot_paused

""" ⬇️ project setup ⬇️ """
'''
Step 1: Set up the environment information
'''
dotenv.load_dotenv()
# Environment Variables
MOOMOOOPEND_ADDRESS = "127.0.0.1"  # should be same as the OpenD host IP, just keep as default
MOOMOOOPEND_PORT = 11111  # should be same as the OpenD port number, make sure keep both the same
TRADING_ENVIRONMENT = TrdEnv.SIMULATE  # set up trading environment, real, or simulate/paper trading
# REAL = "REAL"
# SIMULATE = "SIMULATE"
DISCORD_LOG_TRADES = True

'''
Step 2: Set up the account information
'''
SECURITY_FIRM = SecurityFirm.FUTUINC  # set up the security firm based on your broker account registration
# for U.S. account, use FUTUINC, (default)
# for HongKong account, use FUTUSECURITIES
# for Singapore account, use FUTUSG
# for Australia account, use FUTUAU

'''
Step 3: Set up the trading information
'''
FILL_OUTSIDE_MARKET_HOURS = True  # enable if order fills on extended hours
TRADING_MARKET = TrdMarket.US  # set up the trading market, US market, HK for HongKong, etc.
# NONE = "N/A"
# HK = "HK"
# US = "US"
# CN = "CN"
# HKCC = "HKCC"
# FUTURES = "FUTURES"
USE_LOCAL_SIM_WALLET = True  # when TRADING_ENVIRONMENT is SIMULATE, bypass broker and use local wallet
SIM_STARTING_CASH = int(os.getenv("WALLET_SIM", "2000"))  # starting cash balance for local simulated wallet
DISCORD_LOG_TRADES = False  # set True to send trade messages via Discord webhook

""" ⏫ project setup ⏫ """


class LocalSimWallet:
    """
    Minimal local simulator for paper trading:
    - persists cash/positions/orders to a JSON file
    - immediately fills orders (FILLED_ALL) at requested price
    - mirrors the moomoo place_order return shape (single-row DataFrame)
    """

    def __init__(self, wallet_path="env/local_sim_wallet.json", start_cash=SIM_STARTING_CASH):
        self.wallet_path = wallet_path
        self.start_cash = start_cash
        self.state = self._load_state()
        # ensure file exists with current state so restarts pick up persisted balances
        self._save_state()

    def _default_state(self):
        return {
            "cash": float(self.start_cash),
            "positions": {},
            "last_order_id": 100000,
            "orders": {}
        }

    def _load_state(self):
        if os.path.exists(self.wallet_path):
            try:
                with open(self.wallet_path, "r") as f:
                    data = json.load(f)
                # ensure required fields exist
                for k, v in self._default_state().items():
                    data.setdefault(k, v if not isinstance(v, dict) else {})
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _save_state(self):
        os.makedirs(os.path.dirname(self.wallet_path), exist_ok=True)
        with open(self.wallet_path, "w") as f:
            json.dump(self.state, f, indent=2)
        return self.state

    def _next_order_id(self):
        self.state["last_order_id"] += 1
        self._save_state()
        return self.state["last_order_id"]

    def _symbol_from_code(self, code):
        return code.split(".", 1)[-1]

    def _ensure_position(self, symbol):
        if symbol not in self.state["positions"]:
            self.state["positions"][symbol] = {
                "qty": 0,
                "avg_cost": 0.0,
                "last_price": 0.0
            }

    def refresh_position_prices(self):
        """
        Pull fresh last prices for all held symbols so equity reflects live market moves.
        Uses yfinance 1m data to grab the latest close for the session.
        """
        positions = self.state.get("positions", {})
        if not positions:
            return positions

        for symbol, pos in positions.items():
            qty = pos.get("qty", 0)
            if qty <= 0:
                continue
            try:
                hist = yf.Ticker(symbol).history(
                    period="1d",
                    interval="1m",
                    actions=False,
                    prepost=False,
                )
                if hist.empty or "Close" not in hist:
                    continue
                last_close = hist["Close"].iloc[-1]
                if pd.isna(last_close):
                    continue
                pos["last_price"] = float(last_close)
            except Exception as e:
                print(f"LocalSimWallet: failed to refresh price for {symbol}: {e}")

        self._save_state()
        return positions

    def get_positions(self, refresh_prices=True):
        if refresh_prices:
            self.refresh_position_prices()
        return self.state["positions"]

    def get_account_info(self, refresh_prices=True):
        if refresh_prices:
            self.refresh_position_prices()
        positions = self.state.get("positions", {})
        market_val = 0.0
        for pos in positions.values():
            px = pos.get("last_price", 0.0) or pos.get("avg_cost", 0.0)
            market_val += pos.get("qty", 0) * px
        total_assets = self.state["cash"] + market_val
        return {
            "cash": round(self.state["cash"], 2),
            "total_assets": round(total_assets, 2),
            "market_value": round(market_val, 2),
        }

    def get_order_status(self, order_id):
        status = self.state["orders"].get(str(order_id)) or self.state["orders"].get(order_id)
        if status:
            return RET_OK, status
        return -1, None

    def place_order(self, price, qty, code, trd_side, order_type, **kwargs):
        symbol = self._symbol_from_code(code)
        order_id = self._next_order_id()
        side_str = "BUY" if trd_side == TrdSide.BUY else "SELL"
        err_msg = ""

        self._ensure_position(symbol)
        pos = self.state["positions"][symbol]

        if side_str == "BUY":
            cost = qty * price
            if self.state["cash"] < cost:
                err_msg = "Not enough cash"
                return -1, err_msg
            new_qty = pos["qty"] + qty
            new_cost = (pos["avg_cost"] * pos["qty"] + cost) / new_qty if new_qty else 0
            pos["qty"] = new_qty
            pos["avg_cost"] = new_cost
            pos["last_price"] = price
            self.state["cash"] -= cost
        else:  # SELL
            if pos["qty"] < qty:
                err_msg = "Not enough position to sell"
                return -1, err_msg
            pos["qty"] -= qty
            pos["last_price"] = price
            self.state["cash"] += qty * price
            if pos["qty"] == 0:
                # remove empty position to keep output clean
                self.state["positions"].pop(symbol, None)

        status = "FILLED_ALL"
        self.state["orders"][str(order_id)] = status
        self._save_state()

        row = {
            "code": code,
            "trd_side": side_str,
            "order_type": str(order_type),
            "order_status": status,
            "order_id": order_id,
            "qty": qty,
            "price": price,
            "dealt_qty": qty,
            "dealt_avg_price": price,
            "last_err_msg": err_msg or "N/A",
            "create_time": get_current_time()
        }
        data = pd.DataFrame([row])
        return RET_OK, data


# Trader class:
class Trader:
    def __init__(self, name='John'):
        self.name = name
        self.trade_context = None
        self.use_local_sim = USE_LOCAL_SIM_WALLET and TRADING_ENVIRONMENT == TrdEnv.SIMULATE
        self.local_wallet = LocalSimWallet() if self.use_local_sim else None
        self.log_to_discord = DISCORD_LOG_TRADES

    def init_context(self):
        if self.use_local_sim:
            return
        self.trade_context = OpenSecTradeContext(filter_trdmarket=TRADING_MARKET, host=MOOMOOOPEND_ADDRESS,
                                                 port=MOOMOOOPEND_PORT, security_firm=SECURITY_FIRM)

    def close_context(self):
        if self.trade_context:
            self.trade_context.close()

    def notify_trade_discord(self, message):
        if not self.log_to_discord:
            return
        try:
            send_webhook_message(message)
        except Exception as e:
            print(f'Discord notify failed: {e}')

    def print_sim_snapshot(self):
        """Debug helper: print current cash and equity table for local simulator."""
        if not self.use_local_sim:
            return
        if self.local_wallet is None:
            self.local_wallet = LocalSimWallet()
        acct = self.local_wallet.get_account_info(refresh_prices=True)
        positions = self.local_wallet.get_positions(refresh_prices=False)
        rows = []
        for sym, pos in positions.items():
            qty = pos.get("qty", 0)
            avg_cost = pos.get("avg_cost", 0.0)
            last_px = pos.get("last_price", avg_cost)
            mkt_val = qty * last_px
            rows.append({"symbol": sym, "qty": qty, "avg_cost": avg_cost, "last_price": last_px, "market_value": mkt_val})
        df = pd.DataFrame(rows)
        total_mkt = df["market_value"].sum() if not df.empty else 0.0
        total_equity = acct["cash"] + total_mkt
        print('--- Local Sim Wallet Snapshot ---')
        print(f"Time: {get_current_time()}")
        print(f"Cash: {acct['cash']}")
        print(f"Equity Value: {total_equity}")
        if df.empty:
            print("Positions: none")
        else:
            print(df.to_string(index=False))
        print('---------------------------------')

    def unlock_trade(self):
        if self.use_local_sim:
            return True
        if self.trade_context is None:
            self.init_context()
        if self.trade_context is None:
            print('Unlock trade failed: trade context not initialized')
            return False
        if TRADING_ENVIRONMENT == TrdEnv.REAL:
            MOOMOO_PW=os.getenv("MOOMOO_PW")
            ret, data = self.trade_context.unlock_trade(MOOMOO_PW)
            if ret != RET_OK:
                print('Unlock trade failed: ', data)
                return False
            print('Unlock Trade success!')
        return True

    def market_sell(self, stock, quantity, price):
        code = f'US.{stock}'
        if self.use_local_sim:
            if self.local_wallet is None:
                self.local_wallet = LocalSimWallet()
            ret, data = self.local_wallet.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.SELL,
                order_type=OrderType.MARKET
            )
            if ret != RET_OK:
                print('Trader: Market Sell failed (sim): ', data)
                return ret, data
            print('Trader: Market Sell success! (sim)')
            return ret, data

        self.init_context()
        if not self.trade_context:
            return -1, "Trade context not initialized"
        if self.unlock_trade():
            # per place_order doc (https://openapi.moomoo.com/moomoo-api-doc/en/trade/place-order.html)
            ret, data = self.trade_context.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.SELL,
                order_type=OrderType.MARKET,
                adjust_limit=0,
                trd_env=TRADING_ENVIRONMENT,
                time_in_force=TimeInForce.DAY,
                fill_outside_rth=FILL_OUTSIDE_MARKET_HOURS,
                session=Session.NONE
            )
            if ret != RET_OK:
                print('Trader: Market Sell failed: ', data)
                self.close_context()
                return ret, data
            print('Trader: Market Sell success!')
            self.close_context()
            return ret, data
        else:
            data = 'Trader: Market Sell failed: unlock trade failed'
            print(data)
            self.close_context()
            return -1, data

    def market_buy(self, stock, quantity, price):
        code = f'US.{stock}'
        if self.use_local_sim:
            if self.local_wallet is None:
                self.local_wallet = LocalSimWallet()
            ret, data = self.local_wallet.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.BUY,
                order_type=OrderType.MARKET
            )
            if ret != RET_OK:
                print('Trader: Market Buy failed (sim): ', data)
                return ret, data
            print('Trader: Market Buy success! (sim)')
            return ret, data

        self.init_context()
        if not self.trade_context:
            return -1, "Trade context not initialized"
        if self.unlock_trade():
            ret, data = self.trade_context.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.BUY,
                order_type=OrderType.MARKET,
                adjust_limit=0,
                trd_env=TRADING_ENVIRONMENT,
                time_in_force=TimeInForce.DAY,
                fill_outside_rth=FILL_OUTSIDE_MARKET_HOURS,
                session=Session.NONE
            )
            if ret != RET_OK:
                print('Trader: Market Buy failed: ', data)
                self.close_context()
                return ret, data
            print('Trader: Market Buy success!')
            self.close_context()
            return ret, data
        else:
            data = 'Trader: Market Buy failed: unlock trade failed'
            print(data)
            self.close_context()
            return -1, data

    def limit_sell(self, stock, quantity, price):
        code = f'US.{stock}'
        if self.use_local_sim:
            if self.local_wallet is None:
                self.local_wallet = LocalSimWallet()
            ret, data = self.local_wallet.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.SELL,
                order_type=OrderType.NORMAL
            )
            if ret != RET_OK:
                print('Trader: Limit Sell failed (sim): ', data)
                return ret, data
            print('Trader: Limit Sell success! (sim)')
            return ret, data

        self.init_context()
        if not self.trade_context:
            return -1, "Trade context not initialized"
        if self.unlock_trade():
            ret, data = self.trade_context.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.SELL,
                order_type=OrderType.NORMAL,
                adjust_limit=0,
                trd_env=TRADING_ENVIRONMENT,
                time_in_force=TimeInForce.DAY,
                fill_outside_rth=FILL_OUTSIDE_MARKET_HOURS,
                session=Session.NONE
            )
            if ret != RET_OK:
                print('Trader: Limit Sell failed: ', data)
                self.close_context()
                return ret, data
            print('Trader: Limit Sell success!')
            self.close_context()
            return ret, data
        else:
            data = 'Trader: Limit Sell failed: unlock trade failed'
            print(data)
            self.close_context()
            return -1, data

    def limit_buy(self, stock, quantity, price):
        code = f'US.{stock}'
        if self.use_local_sim:
            if self.local_wallet is None:
                self.local_wallet = LocalSimWallet()
            ret, data = self.local_wallet.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.BUY,
                order_type=OrderType.NORMAL
            )
            if ret != RET_OK:
                print('Trader: Limit Buy failed (sim): ', data)
                return ret, data
            print('Trader: Limit Buy success! (sim)')
            return ret, data

        self.init_context()
        if not self.trade_context:
            return -1, "Trade context not initialized"
        if self.unlock_trade():
            ret, data = self.trade_context.place_order(
                price=price,
                qty=quantity,
                code=code,
                trd_side=TrdSide.BUY,
                order_type=OrderType.NORMAL,
                adjust_limit=0,
                trd_env=TRADING_ENVIRONMENT,
                time_in_force=TimeInForce.DAY,
                fill_outside_rth=FILL_OUTSIDE_MARKET_HOURS,
                session=Session.NONE
            )
            if ret != RET_OK:
                print('Trader: Limit Buy failed: ', data)
                self.close_context()
                return ret, data
            print('Trader: Limit Buy success!')
            self.close_context()
            return ret, data
        else:
            data = 'Trader: Limit Buy failed: unlock trade failed'
            print(data)
            self.close_context()
            return -1, data

    def get_account_info(self):
        if self.use_local_sim:
            if self.local_wallet is None:
                return -1, "Local simulator not initialized"
            acct_info = self.local_wallet.get_account_info()
            logging_info('Trader: Get Account Info success! (sim)')
            return RET_OK, acct_info

        self.init_context()
        if not self.trade_context:
            return -1, "Trade context not initialized"
        if self.unlock_trade():
            ret, data = self.trade_context.accinfo_query()
            if ret != RET_OK:
                print('Trader: Get Account Info failed: ', data)
                self.close_context()
                return ret, data

            cash_val = float(data["us_cash"].iloc[0])
            total_assets_val = float(data["total_assets"].iloc[0])
            market_val = float(data["market_val"].iloc[0])

            acct_info = {
                # https://openapi.moomoo.com/moomoo-api-doc/en/trade/get-funds.html
                # Obsolete. Please use 'us_cash' or other fields to get the cash of each currency.
                # updated 01-07-2025
                'cash': round(cash_val, 2),
                'total_assets': round(total_assets_val, 2),
                'market_value': round(market_val, 2),
            }
            self.close_context()
            logging_info('Trader: Get Account Info success!')
            return ret, acct_info
        else:
            data = 'Trader: Get Account Info failed: unlock trade failed'
            print(data)
            self.close_context()
            return -1, data

    def get_positions(self):
        if self.use_local_sim:
            if self.local_wallet is None:
                return -1, "Local simulator not initialized"
            positions = self.local_wallet.get_positions()
            logging_info('Trader: Get Positions success! (sim)')
            return RET_OK, positions

        self.init_context()
        if not self.trade_context:
            return -1, "Trade context not initialized"
        if self.unlock_trade():
            ret, data = self.trade_context.position_list_query()
            if ret != RET_OK:
                print('Trader: Get Positions failed: ', data)
                self.close_context()
                return ret, data
            # refactor the data
            data['code'] = data['code'].str[3:]

            # ensure last_price exists for downstream display (e.g., /positions)
            if 'last_price' not in data.columns:
                if 'nominal_price' in data.columns:
                    data['last_price'] = data['nominal_price']
                elif 'market_val' in data.columns and 'qty' in data.columns:
                    data['last_price'] = data.apply(
                        lambda r: (r['market_val'] / r['qty']) if r['qty'] else 0, axis=1
                    )

            data_dict = data.set_index('code').to_dict(orient='index')
            self.close_context()
            logging_info('Trader: Get Positions success!')
            return ret, data_dict
        else:
            data = 'Trader: Get Positions failed: unlock trade failed'
            print(data)
            self.close_context()
            return -1, data

    def get_order_status(self, order_id):
        """
        Query a single order status by order_id.
        Returns (ret, status_str or None).
        """
        if self.use_local_sim:
            if self.local_wallet is None:
                return -1, None
            return self.local_wallet.get_order_status(order_id)

        self.init_context()
        if not self.trade_context:
            return -1, None
        if self.unlock_trade():
            ret, data = self.trade_context.order_list_query(order_id=order_id, trd_env=TRADING_ENVIRONMENT)
            if ret != RET_OK or data.empty:
                print('Trader: Get Order Status failed: ', data)
                self.close_context()
                return ret, None
            status_series = data["order_status"]
            status = str(status_series.iloc[0]) if not status_series.empty else None
            self.close_context()
            return ret, status
        else:
            data = 'Trader: Get Order Status failed: unlock trade failed'
            print(data)
            self.close_context()
            return -1, None


if __name__ == '__main__':

    print(get_current_time(), 'TradingBOT is running...')
    # Create a trader and strategy object
    trader = Trader()
    strategy = Your_Strategy(trader)
    print("trader and strategy objects created...")

    # schedule the task
    bot_task = schedule.Scheduler()
    bot_task.every().minute.at(":05").do(strategy.strategy_decision)    # please change the interval as needed

    # print the time every hour showing bot running...
    bkg_task = schedule.Scheduler()
    bkg_task.every().hour.at(":00").do(print_current_time)

    # local sim debug snapshot every 2 minutes
    sim_debug_task = schedule.Scheduler()
    if trader.use_local_sim:
        sim_debug_task.every(2).minutes.do(trader.print_sim_snapshot)

    print("schedule the task...")

    # loop and keep the schedule running
    set_bot_paused(is_bot_paused())  # ensure state file exists on boot
    last_pause_print = 0
    while True:
        bkg_task.run_pending()
        if trader.use_local_sim:
            sim_debug_task.run_pending()
        if is_bot_paused():
            # throttle pause log to avoid spamming
            now = time.time()
            if now - last_pause_print > 60:
                print(f"{get_current_time()} Bot is PAUSED. Use /resume to restart trading checks.")
                last_pause_print = now
            time.sleep(1)
            continue
        if is_market_and_extended_hours() and is_trading_day():
            bot_task.run_pending()  # please handle all error in your strategy

        time.sleep(1)
