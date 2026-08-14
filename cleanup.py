"""One-shot cleanup script: cancels ALL orders and closes ALL positions.

Run this before restarting the bot to clean up accumulated orphans.
Stop the bot first, then run:  python cleanup.py
"""

from config import settings
from exchange import Exchange
from loguru import logger
from logger import setup_logging


def main() -> None:
    setup_logging(settings.log_dir, "INFO")
    logger.info("CLEANUP SCRIPT | connecting to exchange...")

    config = settings.exchange_config
    exchange = Exchange(config, demo=settings.demo_mode)

    logger.info("Cancelling ALL orders (limits + stops + conditional)...")
    cancelled = exchange.cancel_everything(settings.symbol)
    logger.info("Cancelled {} orders", cancelled)

    logger.info("Closing ALL open positions...")
    closed = exchange.close_all_positions(settings.symbol)
    logger.info("Closed {} positions", closed)

    logger.info("Verifying clean state...")
    remaining_orders = exchange.get_open_orders(settings.symbol)
    remaining_stops = exchange.get_stop_orders(settings.symbol)
    positions = exchange.get_positions(settings.symbol)
    has_positions = any(float(p.get("contracts", 0) or 0) != 0 for p in positions)

    # Unreadable is not clean. Announcing "exchange is clean" when the stop book could
    # not be READ would be a lie about the one thing this script exists to confirm
    # (AUDIT #54).
    stops_unknown = remaining_stops is None
    if stops_unknown:
        remaining_stops = []

    if stops_unknown:
        logger.error(
            "CLEANUP UNVERIFIED | the stop/conditional book for {} could not be read — "
            "state is UNKNOWN, not clean. Check the exchange manually before starting.",
            settings.symbol,
        )
        for o in remaining_orders:
            logger.error("  Remaining order: {}", o.get("id"))
        for p in positions:
            amt = float(p.get("contracts", 0) or 0)
            if amt != 0:
                logger.error("  Remaining position: {} {} @ {}", p.get("side"), amt, p.get("entryPrice"))
    elif remaining_orders or remaining_stops or has_positions:
        logger.error("CLEANUP INCOMPLETE | orders={} stops={} positions={}", len(remaining_orders), len(remaining_stops), has_positions)
        for o in remaining_orders:
            logger.error("  Remaining order: {}", o.get("id"))
        for o in remaining_stops:
            logger.error("  Remaining stop: {}", o.get("id"))
        for p in positions:
            amt = float(p.get("contracts", 0) or 0)
            if amt != 0:
                logger.error("  Remaining position: {} {} @ {}", p.get("side"), amt, p.get("entryPrice"))
    else:
        logger.info("CLEANUP COMPLETE | exchange is clean")

    info = exchange.get_balance_info()
    logger.info("Balance: free={:.2f} used={:.2f} total={:.2f} USDT", info["free"], info["used"], info["total"])


if __name__ == "__main__":
    main()
