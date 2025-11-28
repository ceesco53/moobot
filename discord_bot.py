"""
Minimal Discord bot exposing a `/positions` slash command.
- Uses the existing Trader class (with local simulator if configured).
- Requires environment variable DISCORD_BOT_TOKEN to be set.

Run:
    DISCORD_BOT_TOKEN=your_token_here python discord_bot.py
"""

import os
import discord
from discord import app_commands
import pandas as pd
from dotenv import load_dotenv
from TradingBOT import Trader, TRADING_ENVIRONMENT, USE_LOCAL_SIM_WALLET, TrdEnv
from utils.bot_state import is_bot_paused, set_bot_paused, status_label
load_dotenv()

def format_positions(trader: Trader) -> str:
    def fmt_money(val):
        try:
            return f"{float(val):,.2f}"
        except Exception:
            return str(val)

    def fmt_num(val):
        try:
            return f"{float(val):,.2f}"
        except Exception:
            return str(val)

    # account info
    acct_ret, acct = trader.get_account_info()
    if acct_ret != 0 or not acct:
        return "Failed to fetch account info."

    pos_ret, positions = trader.get_positions()
    if pos_ret != 0 or positions is False:
        return "Failed to fetch positions."

    lines = []
    env_label = getattr(TRADING_ENVIRONMENT, "name", str(TRADING_ENVIRONMENT))
    lines.append(f"Env: {env_label} | LocalSim: {trader.use_local_sim}")
    lines.append(f"Cash: {fmt_money(acct.get('cash', 'N/A'))}")
    lines.append(
        f"Total Assets: {fmt_money(acct.get('total_assets', 'N/A'))}  |  "
        f"Market Value: {fmt_money(acct.get('market_value', 'N/A'))}"
    )
    lines.append("")
    lines.append("Positions:")
    if not positions:
        lines.append("  (none)")
    else:
        rows = []
        for sym, pos in positions.items():
            qty = pos.get("qty") if isinstance(pos, dict) else None
            avg_cost = pos.get("avg_cost") if isinstance(pos, dict) else None
            last_px = pos.get("last_price") if isinstance(pos, dict) else None
            mkt_val = qty * (last_px or 0) if qty is not None else None
            rows.append({
                "SYMBOL": sym,
                "QTY": qty,
                "AVG_COST": fmt_money(avg_cost),
                "LAST_PRICE": fmt_money(last_px),
                "MKT_VALUE": fmt_money(mkt_val)
            })
        df = pd.DataFrame(rows, columns=["SYMBOL", "QTY", "AVG_COST", "LAST_PRICE", "MKT_VALUE"])
        table_str = df.to_string(index=False)
        lines.append("```")
        lines.append(table_str)
        lines.append("```")
    return "\n".join(lines)


class PositionsBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.trader = Trader(name="discord-bot")

    async def setup_hook(self):
        # register commands
        @self.tree.command(name="positions", description="Show current positions and cash.")
        async def positions(interaction: discord.Interaction):
            await interaction.response.defer(thinking=True, ephemeral=True)
            msg = format_positions(self.trader)
            await interaction.followup.send(msg, ephemeral=True)

        @self.tree.command(name="pause", description="Pause trading bot (no new trades).")
        async def pause(interaction: discord.Interaction):
            set_bot_paused(True)
            await interaction.response.send_message(
                f"Trading bot paused. Current state: {status_label()}",
                ephemeral=True
            )

        @self.tree.command(name="resume", description="Resume trading bot.")
        async def resume(interaction: discord.Interaction):
            set_bot_paused(False)
            await interaction.response.send_message(
                f"Trading bot resumed. Current state: {status_label()}",
                ephemeral=True
            )

        await self.tree.sync()


def main():
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN env var not set.")
    bot = PositionsBot()
    bot.run(token)


if __name__ == "__main__":
    main()
