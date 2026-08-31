import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Optional tuning-profile overlay. .env always loads first (API keys, Telegram
# secrets, whatever you already have). If GRID_BOT_ENV_FILE points at one of the
# presets in configs/ (e.g. GRID_BOT_ENV_FILE=configs/high_frequency.env), its
# values are layered on top and win on any key it defines -- so a profile only
# needs to list the parameters it tunes, not your secrets. Unset, behavior is
# identical to before (just .env).
_PROFILE_FILE = os.environ.get("GRID_BOT_ENV_FILE")
_ENV_FILES = (".env", _PROFILE_FILE) if _PROFILE_FILE else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Mode ---
    demo_mode: bool = Field(default=True, description="Use Binance testnet")

    # --- Exchange ---
    api_key: str = Field(default="", description="Binance API key")
    api_secret: str = Field(default="", description="Binance API secret")

    # --- Grid ---
    symbol: str = Field(default="BTCUSDT", description="Trading pair")
    leverage: int = Field(
        default=2, ge=1, le=25,
        description=(
            "Exchange leverage. With CAPITAL_PER_GRID_USDT set, this also multiplies the "
            "order size: notional per order = CAPITAL_PER_GRID_USDT x LEVERAGE.\n"
            "\n"
            "The ceiling was an undocumented 20; raised to 25 on request. It is a rail, "
            "not a recommendation, and two things bound what is sensible above it:\n"
            "  - One side of the ladder must still fit inside MAX_POSITION_PCT, or the "
            "outer rungs can never fill and the book goes permanently one-sided.\n"
            "  - On ISOLATED margin, liquidation sits near 1/leverage minus the "
            "maintenance rate. At 25x on DOGEUSDT that is ~3.4% against a 3% hard stop -- "
            "barely half a percent of daylight, and liquidation prices off the MARK, "
            "which wanders from last. Startup now reads the account's margin mode and "
            "refuses to trade if the stop does not clear liquidation, instead of finding "
            "out during a drawdown (AUDIT #69)."
        ),
    )
    grid_count: int = Field(default=15, ge=3, le=100, description="Number of grid levels")
    capital_per_grid_pct: float = Field(
        default=0.05, gt=0, le=0.25,
        description="Fraction of balance allocated per grid level",
    )
    capital_per_grid_usdt: float = Field(
        default=0.0, ge=0, le=10000,
        description=(
            "Fixed USDT of YOUR CAPITAL committed per grid level. When > 0 this is "
            "AUTHORITATIVE: notional per order = this x LEVERAGE x volatility multiplier, "
            "and CAPITAL_PER_GRID_PCT is ignored entirely. Set it to 0 to size by percent "
            "of balance instead.\n"
            "\n"
            "It used to take the LARGER of this and the percent allocation, which made "
            "the setting silently inert whenever percent was bigger -- on a 4930 balance "
            "the config asked for 25 and every order went out at 88.74 (AUDIT #63).\n"
            "\n"
            "Still bounded by MAX_EXPOSURE_PCT across all rungs, and by MAX_POSITION_PCT "
            "on the resulting position."
        ),
    )
    replacement_cooldown: int = Field(
        default=20, ge=0, le=600,
        description="Minimum seconds between replacement orders per grid level after a fill",
    )
    order_pacing_seconds: float = Field(
        default=0.6, ge=0.0, le=30.0,
        description="Delay between consecutive order placements (bursty placement trips exchange system-level protection)",
    )
    grid_timeframe: str = Field(
        default="1h", description="Timeframe for grid range calculation (1m, 5m, 15m, 30m, 1h, 4h)",
    )
    trend_timeframe: str = Field(
        default="1h", description="Medium timeframe for trend filter (15m, 30m, 1h, 4h, 1d)",
    )
    trend_timeframe_fast: str = Field(
        default="30m", description="Fast timeframe for trend filter (1m, 5m, 15m, 30m)",
    )

    # --- Auto Range ---
    range_lookback_days: int = Field(
        default=14, ge=1, le=90,
        description="Days of price history to calculate grid range",
    )
    range_atr_multiplier: float = Field(
        default=1.5, ge=0.3, le=50.0,
        description="Grid bounds = current_price +/- ATR * multiplier",
    )
    range_min_spacing_pct: float = Field(
        default=0.01, ge=0.0005, le=0.05,
        description="Minimum grid spacing as fraction of price (1.0% = 0.01)",
    )
    range_mode: str = Field(
        default="atr",
        description=(
            "How the grid's half-width is estimated. 'atr' is ATR(14) x "
            "RANGE_ATR_MULTIPLIER, the long-standing behaviour. 'realised' is the mean "
            "candle range over the last 24 bars x 5, which predicts the next day's "
            "actual span better in every walk-forward fold tested -- and the range is "
            "for exactly that. Too wide and the rungs sit where price never goes "
            "(AUDIT #111)."
        ),
    )
    min_profit_multiplier: float = Field(
        default=3.0, ge=1.0, le=10.0,
        description=(
            "A grid level is only placed if its spacing covers this many round-trip "
            "fees. 1.0 means break-even (the old hardcoded behaviour, which let levels "
            "trade for near-zero edge); 3.0 keeps roughly two thirds of gross after fees."
        ),
    )

    pnl_divergence_alert_usdt: float = Field(
        default=2.0, ge=0.0,
        description=(
            "Alert when the engine's own P&L and the account's reconciled P&L "
            "disagree by more than this over the same interval. The two numbers "
            "already sit side by side in every status line -- net= from the engine, "
            "account= from Binance income -- and nothing compared them: on "
            "2026-08-20 they read net=2.11 and account=-71.43 for days. Forced "
            "closes (hard stop, reconcile, emergency) never reach the engine ledger, "
            "so it reports profit while the account bleeds. 0 disables."
        ),
    )
    account_recheck_seconds: float = Field(
        default=900.0, ge=0.0,
        description=(
            "How often to re-verify that the exchange account still matches the "
            "bot's sizing assumptions. verify_account_config ran once at startup "
            "and never again, so leverage, margin mode, position mode or fee rates "
            "changed under a running bot went unnoticed until a restart -- and every "
            "notional and margin figure computed after that point was wrong. "
            "0 disables the recheck."
        ),
    )
    empty_book_alert_seconds: float = Field(
        default=900.0, ge=0.0,
        description=(
            "Alert after this long holding a position with no working ladder orders. "
            "The 2026-08-19 deadlock ran 3h13m with an open 6,307 ADA short and an "
            "empty book while the loop polled 4,770 times without raising once. "
            "0 disables the alert."
        ),
    )
    empty_book_restart_seconds: float = Field(
        default=2700.0, ge=0.0,
        description=(
            "Exit non-zero after this long dormant, so supervise.py restarts the bot. "
            "A restart re-lays the ladder with the position cap seeded, which is what "
            "actually broke the deadlock on 2026-08-19 at 16:29:58. Must exceed "
            "EMPTY_BOOK_ALERT_SECONDS so the alert always lands first. 0 disables the "
            "restart and leaves detection alerting only."
        ),
    )
    rung_loss_cap_pct: float = Field(
        default=0.01, ge=0.0, le=0.10,
        description=(
            "How far past break-even an exit may price WHILE THE OTHER SIDE OF THE "
            "LADDER IS CAP-BLOCKED. AUDIT #32 refuses to book any loss, on the grounds "
            "that the levels will unwind the inventory -- true only while those levels "
            "can still trade. Once the position eats the position cap the other side is "
            "blocked, no rung can be placed, and waiting earns nothing. 0.0 restores the "
            "never-book-a-loss behaviour."
        ),
    )

    # --- Grid Recentering ---
    recenter_enabled: bool = Field(
        default=True, description="Recenter grid when price moves outside bounds",
    )
    recenter_margin_pct: float = Field(
        default=0.008, ge=0.002, le=0.05,
        description="Price must exceed grid bound by this fraction to trigger recenter",
    )
    recenter_cooldown: int = Field(
        default=180, ge=30, le=3600,
        description="Minimum seconds between recentering events",
    )

    # --- Fees ---
    maker_fee_pct: float = Field(
        default=0.02, ge=0.0, le=0.1,
        description="Maker fee as percent (0.02% = 0.0002)",
    )
    taker_fee_pct: float = Field(
        default=0.04, ge=0.0, le=0.1,
        description="Taker fee as percent (0.04% = 0.0004)",
    )
    taker_fill_share_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description=(
            "Share of a GRID CYCLE's fill volume that pays the TAKER rate, as a percent. "
            "Every level-profitability gate and break-even price blends the maker and "
            "taker rates by this weight.\n"
            "\n"
            "Measured over 30 days of userTrades, split by the maker flag:\n"
            "  maker fills  106,398 notional, 21.280 commission -> 0.0200%/side exactly\n"
            "  taker fills   62,948 notional, 25.179 commission -> 0.0400%/side exactly\n"
            "\n"
            "So grid cycles pay PURE MAKER. Taker is 37.2% of all traded notional, but "
            "every bit of it is forced exits -- stop-markets, reconcile closes, crossed "
            "unwinds -- not grid cycles. Do NOT put 37.2 here: it would charge ordinary "
            "levels for stop-outs and reject levels that are genuinely profitable. The "
            "5% default is a small buffer for reduce-only exits, which are placed "
            "postOnly=False and can cross (AUDIT #51, corrected in #56)."
        ),
    )

    # --- Reporting ---
    pnl_epoch: str = Field(
        default="",
        description=(
            "UTC date (YYYY-MM-DD) from which cumulative PnL is reported. Empty means a "
            "rolling BOOTSTRAP_LOOKBACK_DAYS window.\n"
            "\n"
            "The rolling window keeps dragging old history into the headline figure: on "
            "2026-08-14 it read -29.08, of which -50.49 was a single day (08-08) caused "
            "by the position-cap and stop-loss defects fixed in #49/#50. That day stays "
            "in the window until early November, so the bot would report a loss for "
            "months no matter how well it traded.\n"
            "\n"
            "Set this to the date the current code went live and the number means "
            "something: PnL under the bot as it actually is. Changing it re-bootstraps "
            "from scratch rather than silently keeping stale totals (AUDIT #60)."
        ),
    )

    # --- Risk ---
    stop_loss_pct: float = Field(
        default=0.05, gt=0, le=0.15,
        description="Kill switch: exit if loss exceeds this below lowest grid",
    )
    trailing_sl_trigger_pct: float = Field(
        default=0.05, gt=0, le=0.20,
        description="Trailing stop-loss trigger: activate SL when price drops this % from peak",
    )
    daily_loss_limit_pct: float = Field(
        default=0.02, gt=0, le=0.20,
        description="Stop trading if daily loss exceeds this fraction of balance",
    )
    max_drawdown_pct: float = Field(
        default=0.08, gt=0, le=0.30,
        description="Emergency stop if account drops below this fraction of starting balance",
    )
    cooldown_seconds: int = Field(
        default=7200, ge=60, le=86400,
        description="Wait time after kill switch before allowing restart",
    )
    max_exposure_pct: float = Field(
        default=0.60, gt=0, le=1.0,
        description="Max portfolio exposure as fraction of equity",
    )
    max_position_pct: float = Field(
        default=0.12, gt=0, le=2.0,
        description="Max position size as fraction of equity (stops unlimited accumulation)",
    )
    max_open_loss_usdt: float = Field(
        default=0.0, ge=0,
        description=(
            "Loss budget for the OPEN position, in USDT. Once its unrealised loss "
            "reaches this, the side that would ADD to the position is blocked (exits "
            "stay legal) until the loss recedes or the position closes.\n"
            "\n"
            "The cap bounds how big a position can get; nothing bounded how much "
            "adverse room it was handed, so a trend running through the ladder could "
            "hand a capped position to the hard stop as one taker print worth hundreds "
            "of grid cycles (the -74.39 vs +2.61 asymmetry of 2026-07-22..08-20). This "
            "bounds the averaging instead: the grid stops digging at the budget.\n"
            "\n"
            "0 disables the guard. Size it against cycle economics: with 25 USDT rungs "
            "earning ~0.03 per cycle, even 10 USDT is generous -- it is roughly one "
            "worst-case stop-out of a MAX_POSITION_PCT=0.05 cap."
        ),
    )
    daily_profit_lock_usdt: float = Field(
        default=0.0, ge=0,
        description=(
            "Profit budget for the DAY, in USDT. Once the day's REALISED P&L reaches "
            "this, the grid stops OPENING new exposure on either side for the rest of "
            "the day -- exits stay legal, so any position already open can still be "
            "managed and closed normally.\n"
            "\n"
            "Every other risk knob here bounds LOSSES -- the hard stop-loss, "
            "MAX_OPEN_LOSS_USDT, DAILY_LOSS_LIMIT_PCT, MAX_DRAWDOWN_PCT -- and none of "
            "them touch the profit side: a good day's gains just ride as continued "
            "exposure, with nothing to lock them in. That is the mirror image of the "
            "shape MAX_OPEN_LOSS_USDT exists for (see its docstring) -- hours of small "
            "grid profit erased in under a minute by one bad move -- except the fix "
            "here is to stop ADDING risk once the day is already won, rather than "
            "bounding the damage after it turns.\n"
            "\n"
            "0 disables the guard. Size it as an absolute USDT figure, like "
            "MAX_OPEN_LOSS_USDT, not as a percent of equity like DAILY_LOSS_LIMIT_PCT: "
            "this strategy's daily gains are a tiny fraction of equity, so a "
            "percent-of-equity threshold at DAILY_LOSS_LIMIT_PCT's scale (3%) would "
            "never fire. This bot's own 16-day daily net P&L (logs/trades_demo.csv, "
            "cycle_pnl minus fee, grouped by day) ranged -3.26 to +6.08 USDT, averaging "
            "+0.80/day, and the best day was only ~0.12% of equity. A few USDT sits "
            "meaningfully above routine daily variance while staying reachable on a "
            "genuinely good day."
        ),
    )
    sl_scale_out_pct: float = Field(
        default=0.50, ge=0, le=0.95,
        description=(
            "Fraction of the open position closed at the trailing stop-loss; the remainder "
            "is kept until the hard stop-loss level. Reduces the impact of market-stop dumps.\n"
            "\n"
            "0 disables the split: one full-size hard stop and no trailing leg. That is the "
            "right setting whenever STOP_LOSS_PCT is tight, because the trailing leg is "
            "anchored at peak*(1-STOP_LOSS_PCT) -- so a 0.5% stop puts it 0.5% under the "
            "peak, which is two grid steps and INSIDE the ladder. It would then fire on "
            "ordinary movement and force taker exits all day. The bound was gt=0, so the "
            "one configuration that makes a tight stop safe could not be expressed; "
            "build_scale_out_orders has always handled scale=0 correctly."
        ),
    )
    max_recovery_count: int = Field(
        default=5, ge=1, le=20,
        description="Maximum recovery attempts before bot shuts down entirely",
    )
    max_consecutive_losses: int = Field(
        default=10, ge=1, le=100,
        description=(
            "Kill switch: stop trading after this many consecutive losing completed "
            "cycles. Every other risk knob is configurable here except this one used "
            "to be -- RiskManager silently used its hardcoded default (10) regardless "
            "of .env."
        ),
    )

    # --- Trend Filter ---
    ema_fast: int = Field(default=20, ge=5, le=50)
    ema_slow: int = Field(default=50, ge=20, le=200)
    adx_period: int = Field(default=14, ge=5, le=50)
    adx_trend_threshold: float = Field(default=30.0, ge=15, le=40)
    adx_range_threshold: float = Field(default=20.0, ge=5, le=30)
    regime_trend_min_votes: int = Field(
        default=2, ge=1, le=3,
        description=(
            "How many timeframes must agree on a trend before the router hands the "
            "symbol to the trend follower. 2 (default) is conservative: a trend verdict "
            "PAUSES the grid, so one timeframe's opinion does not stop grid trading. "
            "Set to 1 to let a single trending timeframe hand over -- on DOGEUSDT the 1h "
            "ADX often sits in the dead band while the 30m trends alone, which leaves the "
            "follower permanently ineligible. Raising frequency this way is a trade-off: "
            "the follower's measured edge was -0.14%/trade over 70 trades."
        ),
    )
    trend_check_interval: int = Field(
        default=300, ge=60, le=3600,
        description="Seconds between trend filter checks",
    )
    trend_confirmation_seconds: int = Field(
        default=300, ge=0, le=3600,
        description="Seconds a new regime must persist before acting on it (0 = instant)",
    )
    lone_trend_alert_seconds: int = Field(
        default=3600, ge=0, le=86400,
        description=(
            "Alert (log + Telegram) when a single timeframe has voted the same trend "
            "direction this long without REGIME_TREND_MIN_VOTES worth of agreement to "
            "act on it -- e.g. 30m trending for hours while 1h/1d never agree. "
            "Observational only: this does not pause the grid or change sizing, it "
            "just makes a genuinely persistent lone trend visible instead of reading "
            "identically to a fresh five-minute one. 0 disables the alert."
        ),
    )
    flat_range_window: int = Field(
        default=6, ge=2, le=48,
        description="Candles used to measure the recent price range for the flat-market override",
    )
    flat_range_pct: float = Field(
        default=0.01, gt=0.0, le=0.05,
        description="Max recent (high-low)/close range below which a trending regime is overridden to RANGING",
    )

    # --- Telegram ---
    telegram_enabled: bool = Field(default=False)
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")

    # --- Strategy selection ---
    strategy_mode: str = Field(
        default="grid",
        description=(
            "'grid' runs the grid engine alone (the long-standing behaviour). "
            "'router' installs the regime router, which trades the grid in ranging "
            "markets and the trend follower in confirmed trends. Defaults to 'grid' "
            "because the router's switching thresholds are not yet validated -- see "
            "AUDIT.md on the backtest noise floor."
        ),
    )
    trend_capital_pct: float = Field(
        default=0.10, gt=0, le=0.50,
        description="Fraction of equity the trend follower commits to one position",
    )
    trend_capital_usdt: float = Field(
        default=0.0, ge=0.0, le=10000.0,
        description=(
            "Own capital committed to ONE trend trade, in USDT. Notional is this times "
            "LEVERAGE, exactly as CAPITAL_PER_GRID_USDT works for a grid rung. "
            "AUTHORITATIVE when set: TREND_CAPITAL_PCT is ignored rather than taken as "
            "a maximum, because a setting that silently loses to a larger percent path "
            "reads as a note in the log and not as 'your number is not being used' "
            "(the AUDIT #63 mistake, which this deliberately does not repeat). "
            "0 keeps the percent-of-equity behaviour. "
            "The exposure and position caps still apply on top of either path."
        ),
    )
    trend_atr_stop_multiplier: float = Field(
        default=2.0, ge=0.5, le=10.0,
        description="Trend follower's trailing stop distance, in ATR multiples",
    )
    trend_trail_atr_multiplier: float = Field(
        default=0.0, ge=0.0, le=20.0,
        description=(
            "Trailing-stop distance in ATR multiples, for the trail ONLY. 0 means "
            "same as TREND_ATR_STOP_MULTIPLIER, which is how it behaved when the "
            "two were one number. "
            "Separating them is what makes TREND_TAKE_PROFIT_R reachable: the "
            "target sits at R times the stop the trade OPENED with, while the "
            "trail follows one stop-width behind the extreme -- so with both "
            "equal, price must run R widths without ever giving back a single "
            "one, and it rarely does. Measured over 62 days of DOGEUSDT on 5m "
            "bars with a 3R target: at 2x/2x the target was reached 3 times in "
            "70 trades for a realised win:loss of 1.46:1; widening the trail to "
            "3x gave 5 in 45 and 2.44:1 (AUDIT #105). "
            "Wider is not free -- it gives back more of an open winner before "
            "closing, and in that same measurement no width made the average "
            "trade profitable."
        ),
    )
    trend_take_profit_r: float = Field(
        default=0.0, ge=0.0, le=20.0,
        description=(
            "Take-profit for the trend follower, in units of the risk taken on that "
            "trade (R). One R is the distance from entry to the stop the trade OPENED "
            "with, so 3.0 puts the target three times as far away as the stop: one "
            "winner pays for three losers.\n"
            "\n"
            "0 disables it and restores the original behaviour -- no target, ride the "
            "trailing stop for as long as the trend runs. That is the classical choice "
            "and it is not obviously worse: a fixed target caps the rare very large "
            "winner, and trend following is usually paid BY that winner. Set this when "
            "a defined reward:risk matters more than an open-ended one.\n"
            "\n"
            "Note the ratio is a ceiling, not a promise. The trailing stop ratchets up "
            "as price runs and can still close the trade below the target."
        ),
    )
    trend_min_hold_seconds: int = Field(
        default=300, ge=0, le=86400,
        description="Minimum time a trend position is held before its stop can fire",
    )
    router_handoff_grace_seconds: int = Field(
        default=1800, ge=0, le=604800,
        description=(
            "How long the outgoing strategy is given to unwind its own position before "
            "the router force-closes it to complete a switch. Market-dumping a grid's "
            "inventory realises exactly the loss its levels exist to avoid, so the "
            "default is generous: wait, take the switch free the moment it goes flat, "
            "and only pay to close if it never gets there."
        ),
    )
    router_min_regime_seconds: int = Field(
        default=900, ge=0, le=86400,
        description=(
            "A regime must persist this long before the router pays to switch. Every "
            "handoff costs a taker fee to flatten plus the spread to re-enter, so "
            "switching on regime noise bleeds on transitions alone."
        ),
    )

    # --- Signals (read-only observer) ---
    signals_enabled: bool = Field(
        default=True,
        description=(
            "Record what each confirmed regime change implied, and score it against "
            "what price did before the next one. Costs nothing -- the observer has no "
            "exchange handle and cannot place orders -- and the accuracy it accumulates "
            "is the evidence the router's thresholds currently lack (AUDIT.md #24)."
        ),
    )
    signals_notify: bool = Field(
        default=False,
        description=(
            "Push actionable signals to Telegram. Off by default: the CSV is the point, "
            "and alerting on every regime flip trains you to ignore the channel."
        ),
    )

    # --- Polling ---
    poll_interval: int = Field(default=30, ge=5, le=300, description="Seconds between fill checks")
    force_trade_now: bool = Field(default=False, description="If true, bypass trend gating and activate grid immediately.")
    close_on_exit: bool = Field(default=False, description="If true, close all open positions when the bot shuts down.")

    # --- Paths ---
    state_dir: str = Field(default="state")
    log_dir: str = Field(default="logs")

    @property
    def pnl_epoch_ms(self) -> int | None:
        """PNL_EPOCH as a UTC epoch in milliseconds, or None for the rolling window.

        Raises on a malformed date rather than falling back: silently reverting to the
        rolling window would leave the operator reading a number they thought they had
        changed (AUDIT #60).
        """
        raw = (self.pnl_epoch or "").strip()
        if not raw:
            return None
        from datetime import datetime, timezone
        try:
            day = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(
                f"PNL_EPOCH must be a UTC date as YYYY-MM-DD, got {raw!r}"
            ) from e
        return int(day.timestamp() * 1000)

    @property
    def exchange_config(self) -> dict:
        return {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "timeout": 60000,
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True,
            },
        }

    def validate(self) -> None:
        # Required in both DEMO and LIVE mode: the bot only ever trades against a real
        # Binance account (Demo Trading or live) now -- there is no mock/simulated
        # trading fallback to fall back to silently. Get demo keys from demo.binance.com.
        if not self.api_key or not self.api_secret:
            mode = "DEMO" if self.demo_mode else "LIVE"
            raise ValueError(
                f"{mode} mode requires API_KEY and API_SECRET to be set in .env "
                "(demo keys: https://demo.binance.com)."
            )

        valid_timeframes = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
        if self.grid_timeframe not in valid_timeframes:
            raise ValueError(f"GRID_TIMEFRAME must be one of {valid_timeframes}, got '{self.grid_timeframe}'")
        if self.trend_timeframe not in valid_timeframes:
            raise ValueError(f"TREND_TIMEFRAME must be one of {valid_timeframes}, got '{self.trend_timeframe}'")
        if self.trend_timeframe_fast not in valid_timeframes:
            raise ValueError(f"TREND_TIMEFRAME_FAST must be one of {valid_timeframes}, got '{self.trend_timeframe_fast}'")

        total_allocation = self.grid_count * self.capital_per_grid_pct
        if total_allocation > 0.5:
            raise ValueError(
                f"Total grid allocation is too high: {total_allocation:.2f}. "
                "Use fewer grid levels or reduce CAPITAL_PER_GRID_PCT to keep the total <= 0.50."
            )

        if self.range_min_spacing_pct < 0.0005:
            raise ValueError(
                "RANGE_MIN_SPACING_PCT is too low; use at least 0.0005 to avoid excessively tight grids."
            )

        # --- Spacing must clear fees by a real margin, not a hair ---------------
        # Every completed cycle earns one grid spacing and pays a round trip of fees.
        # Spacing set near the fee floor is how a bot books thousands of fills and still
        # ends the day negative: the exchange takes most of the gross. The round trip is
        # priced at the BLENDED rate, not the maker rate -- assuming an all-maker book
        # understated the true cost by 12% (AUDIT #51).
        share = self.taker_fill_share_pct / 100
        round_trip_fee = 2 * (
            (self.maker_fee_pct / 100) * (1 - share) + (self.taker_fee_pct / 100) * share
        )
        required_spacing = round_trip_fee * self.min_profit_multiplier
        if self.range_min_spacing_pct < required_spacing:
            raise ValueError(
                f"RANGE_MIN_SPACING_PCT ({self.range_min_spacing_pct:.5f} = "
                f"{self.range_min_spacing_pct * 100:.3f}%) does not clear fees. A round trip "
                f"costs {round_trip_fee * 100:.3f}% at the blended rate "
                f"({self.taker_fill_share_pct:.1f}% taker), and "
                f"MIN_PROFIT_MULTIPLIER={self.min_profit_multiplier} requires spacing of at "
                f"least {required_spacing:.5f} ({required_spacing * 100:.3f}%). "
                "Raise RANGE_MIN_SPACING_PCT, lower GRID_COUNT, or lower MIN_PROFIT_MULTIPLIER."
            )

        # --- The grid must fit inside the position cap --------------------------
        # Each filled level adds capital_per_grid_pct of equity to the position, and
        # max_position_pct caps the total. If one side of the grid holds more levels
        # than the cap allows, the surplus levels can never fill: the cap blocks that
        # side partway through, the book goes permanently one-sided, and the bot spends
        # its life in the capped state that makes recentering destructive.
        #
        # Only meaningful when the PERCENT path governs. With CAPITAL_PER_GRID_USDT set
        # the size is an absolute USDT figure and max_position_pct is a fraction of
        # equity, so the two cannot be compared without a balance -- which config
        # validation does not have. That case is bounded at runtime instead, by the
        # exposure ceiling in _calc_usdt_per_grid and the position cap in
        # set_position_limit. Checking the percent path regardless would be validating a
        # number that no longer decides anything (AUDIT #63).
        levels_per_side = self.grid_count / 2
        affordable_levels = self.max_position_pct / self.capital_per_grid_pct
        if self.capital_per_grid_usdt <= 0 and levels_per_side > affordable_levels:
            max_coherent_count = int(2 * affordable_levels)
            raise ValueError(
                f"GRID_COUNT ({self.grid_count}) puts {levels_per_side:.0f} levels on each "
                f"side, but MAX_POSITION_PCT ({self.max_position_pct:.0%}) only affords "
                f"{affordable_levels:.1f} levels at CAPITAL_PER_GRID_PCT "
                f"({self.capital_per_grid_pct:.1%}). The rest can never fill. "
                f"Use GRID_COUNT <= {max_coherent_count}, or raise MAX_POSITION_PCT, "
                "or lower CAPITAL_PER_GRID_PCT."
            )

        if self.strategy_mode not in {"grid", "router"}:
            raise ValueError(
                f"STRATEGY_MODE must be 'grid' or 'router', got '{self.strategy_mode}'."
            )

        if self.ema_fast >= self.ema_slow:
            raise ValueError(
                f"EMA_FAST ({self.ema_fast}) must be less than EMA_SLOW ({self.ema_slow})."
            )

        if self.adx_range_threshold >= self.adx_trend_threshold:
            raise ValueError(
                f"ADX_RANGE_THRESHOLD ({self.adx_range_threshold}) must be less than "
                f"ADX_TREND_THRESHOLD ({self.adx_trend_threshold})."
            )


settings = Settings()
