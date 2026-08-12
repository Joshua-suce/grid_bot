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
    leverage: int = Field(default=2, ge=1, le=20)
    grid_count: int = Field(default=15, ge=3, le=100, description="Number of grid levels")
    capital_per_grid_pct: float = Field(
        default=0.05, gt=0, le=0.25,
        description="Fraction of balance allocated per grid level",
    )
    capital_per_grid_usdt: float = Field(
        default=0.0, ge=0, le=10000,
        description=(
            "Fixed USDT margin per grid level. If > 0, the bot uses the larger of this "
            "fixed allocation and the percentage-based allocation to avoid overly small "
            "grid sizing."
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
    min_profit_multiplier: float = Field(
        default=3.0, ge=1.0, le=10.0,
        description=(
            "A grid level is only placed if its spacing covers this many round-trip "
            "fees. 1.0 means break-even (the old hardcoded behaviour, which let levels "
            "trade for near-zero edge); 3.0 keeps roughly two thirds of gross after fees."
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
    sl_scale_out_pct: float = Field(
        default=0.50, gt=0, le=0.95,
        description=(
            "Fraction of the open position closed at the trailing stop-loss; the remainder "
            "is kept until the hard stop-loss level. Reduces the impact of market-stop dumps."
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
    trend_check_interval: int = Field(
        default=300, ge=60, le=3600,
        description="Seconds between trend filter checks",
    )
    trend_confirmation_seconds: int = Field(
        default=300, ge=0, le=3600,
        description="Seconds a new regime must persist before acting on it (0 = instant)",
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
    trend_atr_stop_multiplier: float = Field(
        default=2.0, ge=0.5, le=10.0,
        description="Trend follower's trailing stop distance, in ATR multiples",
    )
    trend_min_hold_seconds: int = Field(
        default=300, ge=0, le=86400,
        description="Minimum time a trend position is held before its stop can fire",
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
        # Every completed cycle earns one grid spacing and pays a round trip of maker
        # fees. Spacing set near the fee floor is how a bot books thousands of fills
        # and still ends the day negative: the exchange takes most of the gross.
        round_trip_fee = 2 * (self.maker_fee_pct / 100)
        required_spacing = round_trip_fee * self.min_profit_multiplier
        if self.range_min_spacing_pct < required_spacing:
            raise ValueError(
                f"RANGE_MIN_SPACING_PCT ({self.range_min_spacing_pct:.5f} = "
                f"{self.range_min_spacing_pct * 100:.3f}%) does not clear fees. A round trip "
                f"costs {round_trip_fee * 100:.3f}% at the maker rate, and "
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
        levels_per_side = self.grid_count / 2
        affordable_levels = self.max_position_pct / self.capital_per_grid_pct
        if levels_per_side > affordable_levels:
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
