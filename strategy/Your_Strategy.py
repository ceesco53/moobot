"""
By: LukeLab (base template)
Modified by: ChatGPT (MACD + VWAP + SMA, intraday 1m/5m)
Version: 2.2
"""

import time
import yfinance as yf
from moomoo import RET_OK
from strategy.Strategy import Strategy
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.volume import VolumeWeightedAveragePrice
from utils.dataIO import read_json_file, write_json_file, logging_info
from utils.time_tool import is_market_hours
from utils.bot_state import is_bot_paused


class Your_Strategy(Strategy):
    """
    Intraday strategy using 1m data for VWAP & execution,
    and 5m data for MACD + SMA trend signal.

    Per symbol (AMD, TSLA, MU):

    Data:
        - Pull 1m intraday data from yfinance.
        - Resample to 5m for MACD & SMA.
        - Use 1m VWAP + last 1m close as filters.

    BUY:
        - MACD (5m) bullish cross: MACD > signal and previously <= signal
        - Last 1m close > 1m VWAP
        - Last 1m close > 5m SMA

    SELL:
        - MACD (5m) bearish cross: MACD < signal and previously >= signal
        - Last 1m close < 1m VWAP
        - Last 1m close < 5m SMA
    """

    def __init__(self, trader):
        super().__init__(trader)
        self.strategy_name = "MACD_VWAP_SMA_1m5m_Intraday"

        """⬇️⬇️⬇️ Strategy Settings ⬇️⬇️⬇️"""

        # strictly intraday symbols
        self.stock_trading_list = ["AMD", "TSLA", "MU"]

        # position size per ticker
        self.trading_qty = {
            "AMD": 2,
            "TSLA": 1,
            "MU": 2
        }

        self.trading_confirmation = True    # True to enable trading confirmation

        # indicator parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.sma_window_5m = 20        # 20 x 5m bars ~ 100 minutes
        self.vwap_window_1m = 390      # up to a full intraday session of 1m bars

        # yfinance data settings
        self.data_interval_1m = "1m"
        # yfinance only gives ~7 days of 1m; "1d" period keeps it intraday recent
        self.data_period = "1d"

        """⬆️⬆️⬆️ Strategy Settings ⬆️⬆️⬆️"""

        print(f"Strategy {self.strategy_name} initialized...")

    def strategy_decision(self):
        print("Strategy Decision running...")
        """ ⬇️⬇️⬇️ MACD + VWAP + SMA 1m/5m Strategy starts here ⬇️⬇️⬇️"""

        if is_bot_paused():
            print("Bot is paused: skipping strategy cycle.")
            return

        for stock in self.stock_trading_list:
            try:
                # 1. get 1m intraday data from yfinance
                df_1m = yf.Ticker(stock).history(
                    period=self.data_period,
                    interval=self.data_interval_1m,
                    actions=False,
                    prepost=False,
                    raise_errors=True
                )

                if df_1m.empty:
                    print(f"{stock}: no data returned, skipping.")
                    continue

                # ensure we have enough bars for 5m indicators
                if df_1m.shape[0] < max(self.macd_slow, self.sma_window_5m) + 3:
                    print(f"{stock}: not enough 1m bars yet, skipping this cycle.")
                    continue

                # 2. build 5m candles from 1m (for smoother MACD + SMA)
                # pandas will deprecate 'T' alias; use 'min' to avoid future warning
                df_5m = df_1m.resample("5min").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna()

                if df_5m.shape[0] < max(self.macd_slow, self.sma_window_5m) + 2:
                    print(f"{stock}: not enough 5m bars for indicators, skipping.")
                    continue

                # 3. calculate indicators on 5m
                df_5m["sma_5m"] = SMAIndicator(
                    close=df_5m["Close"],
                    window=self.sma_window_5m
                ).sma_indicator()

                macd_ind_5m = MACD(
                    close=df_5m["Close"],
                    window_fast=self.macd_fast,
                    window_slow=self.macd_slow,
                    window_sign=self.macd_signal
                )
                df_5m["macd"] = macd_ind_5m.macd()
                df_5m["macd_signal"] = macd_ind_5m.macd_signal()
                df_5m["macd_hist"] = macd_ind_5m.macd_diff()

                # 4. calculate VWAP on 1m (intraday)
                vwap_ind_1m = VolumeWeightedAveragePrice(
                    high=df_1m["High"],
                    low=df_1m["Low"],
                    close=df_1m["Close"],
                    volume=df_1m["Volume"],
                    window=min(self.vwap_window_1m, len(df_1m))
                )
                df_1m["vwap_1m"] = vwap_ind_1m.volume_weighted_average_price()

                # 5. get latest values
                last_1m = df_1m.iloc[-1]
                price_1m = last_1m["Close"]
                vwap_1m = last_1m["vwap_1m"]

                last_5m = df_5m.iloc[-1]
                prev_5m = df_5m.iloc[-2]

                curr_macd = last_5m["macd"]
                curr_signal = last_5m["macd_signal"]
                prev_macd = prev_5m["macd"]
                prev_signal = prev_5m["macd_signal"]
                sma_5m = last_5m["sma_5m"]

                qty = self.trading_qty.get(stock, 0)
                if qty <= 0:
                    print(f"{stock}: qty not configured or zero, skipping.")
                    continue

                # 6. define signals (crosses on 5m, filters on 1m)
                macd_bull_cross = (curr_macd > curr_signal) and (prev_macd <= prev_signal)
                macd_bear_cross = (curr_macd < curr_signal) and (prev_macd >= prev_signal)

                price_above_vwap = price_1m > vwap_1m
                price_below_vwap = price_1m < vwap_1m
                price_above_sma = price_1m > sma_5m
                price_below_sma = price_1m < sma_5m

                # BUY: bull MACD cross on 5m + price>VWAP + price>SMA
                if macd_bull_cross and price_above_vwap and price_above_sma:
                    print(f'{stock}: BUY signal -> 5m MACD bull cross, 1m price > VWAP & 5m SMA')
                    self.strategy_make_trade(action='BUY', stock=stock, qty=qty, price=price_1m)

                # SELL: bear MACD cross on 5m + price<VWAP + price<SMA
                if macd_bear_cross and price_below_vwap and price_below_sma:
                    print(f'{stock}: SELL signal -> 5m MACD bear cross, 1m price < VWAP & 5m SMA')
                    self.strategy_make_trade(action='SELL', stock=stock, qty=qty, price=price_1m)

                time.sleep(1)  # avoid hitting APIs too hard

            except Exception as e:
                print(f"Strategy Error for {stock}: {e}")
                logging_info(f'{self.strategy_name} ({stock}): {e}')

        """ ⏫⏫⏫ MACD + VWAP + SMA 1m/5m Strategy ends here ⏫⏫⏫ """

        print("Strategy checked... Waiting next decision called...")
        print('-----------------------------------------------')

    """ ⬇️⬇️⬇️ Order related functions (same as your template) ⬇️⬇️⬇️"""

    def strategy_make_trade(self, action, stock, qty, price):
        if self.trading_confirmation:
            # check if trading confirmation is enabled first
            if action == 'BUY':
                # check the current buying power first
                acct_ret, acct_info = self.trader.get_account_info()
                if acct_ret == RET_OK:
                    current_cash = acct_info['cash']
                else:
                    print('Trader: Get Account Info failed: ', acct_info)
                    return False

                if current_cash > qty * price:
                    # before buy action, check if it has enough cash
                    if is_market_hours():
                        # market order
                        ret, data = self.trader.market_buy(stock, qty, price)
                    else:
                        # limit order for extended hours
                        ret, data = self.trader.limit_buy(stock, qty, price)

                    if ret == RET_OK:
                        # order placed successfully:
                        print(data)
                        self.save_order_history(data)
                        order_id = data["order_id"].iloc[0] if "order_id" in data else None
                        self._notify_trade_discord(action='BUY', stock=stock, qty=qty, price=price, data=data)
                        filled = self._wait_for_order_fill(order_id)
                        if filled:
                            print('order filled, show latest position:')
                            print(self.get_current_position())  # show the latest position after trade
                        else:
                            print('order not filled yet (or status uncertain), skipping position check for now')
                    else:
                        print('Trader: Buy failed: ', data)
                        logging_info(f'{self.strategy_name}: Buy failed: {data}')
                else:
                    print('Trader: Buy failed: Not enough cash to buy')
                    logging_info(f'{self.strategy_name}: Buy failed: Not enough cash to buy')

            if action == 'SELL':
                position_data = self.get_current_position()
                if not position_data:
                    # check current position first
                    return False

                if stock not in position_data:
                    print(f'Trader: Sell failed: No position data for {stock}')
                    logging_info(f'{self.strategy_name}: Sell failed: No position data for {stock}')
                    return False

                if qty <= position_data[stock]["qty"]:
                    # before sell action, check if it has enough position to sell
                    if is_market_hours():
                        # market order
                        ret, data = self.trader.market_sell(stock, qty, price)
                    else:
                        # limit order for extended hours
                        ret, data = self.trader.limit_sell(stock, qty, price)

                    if ret == RET_OK:
                        print(data)
                        logging_info(f'{self.strategy_name}: {data}')
                        self.save_order_history(data)
                        order_id = data["order_id"].iloc[0] if "order_id" in data else None
                        self._notify_trade_discord(action='SELL', stock=stock, qty=qty, price=price, data=data)
                        filled = self._wait_for_order_fill(order_id)
                        if filled:
                            print('order filled, show latest position:')
                            print(self.get_current_position())  # show the latest position after trade
                        else:
                            print('order not filled yet (or status uncertain), skipping position check for now')
                    else:
                        print('Trader: Sell failed: ', data)
                        logging_info(f'{self.strategy_name}: Sell failed: {data}')
                else:
                    print('Trader: Sell failed: Not enough position to sell')
                    logging_info(f'{self.strategy_name}: Sell failed: Not enough position to sell')

    def save_order_history(self, data):
        file_data = read_json_file("order_history.json")
        data_dict = data.to_dict()
        new_dict = {}
        for key, v in data_dict.items():
            new_dict[key] = v[0]
        logging_info(f'{self.strategy_name}: {str(new_dict)}')

        if file_data:
            file_data.append(new_dict)
        else:
            file_data = [new_dict]
        write_json_file("order_history.json", file_data)

    # add any other functions you need here

    def _wait_for_order_fill(self, order_id, timeout=20, poll_interval=1.5):
        """
        Poll order status until FILLED* (or similar) or timeout.
        Returns True if filled, False otherwise.
        """
        if order_id is None:
            print('order_id missing, cannot poll order status')
            return False

        deadline = time.time() + timeout
        last_status = None
        while time.time() < deadline:
            ret, status = self.trader.get_order_status(order_id)
            if ret == RET_OK and status:
                last_status = status
                status_upper = status.upper()
                print(f'order {order_id} status: {status_upper}')
                if "FILLED" in status_upper:
                    return True
                if any(term in status_upper for term in ("CANCEL", "REJECT", "FAIL")):
                    break
            else:
                print(f'order {order_id} status check failed')
            time.sleep(poll_interval)

        if last_status:
            print(f'order {order_id} final status after wait: {last_status}')
        else:
            print(f'order {order_id}: no status retrieved during wait window')
        return False

    def _notify_trade_discord(self, action, stock, qty, price, data):
        """Send a concise trade message to Discord if enabled."""
        try:
            status = data["order_status"].iloc[0] if "order_status" in data else "UNKNOWN"
            order_id = data["order_id"].iloc[0] if "order_id" in data else "N/A"
            dealt_qty = data["dealt_qty"].iloc[0] if "dealt_qty" in data else qty
            dealt_avg = data["dealt_avg_price"].iloc[0] if "dealt_avg_price" in data else price
            msg = (
                f"Trade {action} {stock} x{dealt_qty} @ {dealt_avg} "
                f"(status: {status}, order_id: {order_id})"
            )
            if hasattr(self.trader, "notify_trade_discord"):
                self.trader.notify_trade_discord(msg)
        except Exception as e:
            print(f"Discord trade notify error: {e}")
