from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import pandas as pd
import yfinance as yf

from config import get_all_tickers


LOGGER_NAME: Final[str] = "data_collector"
TICKER_SYMBOL: Final[str] = "^GSPC"
DEFAULT_OUTPUT_PATH: Final[Path] = Path("data") / "sp500_historical.csv"
LOOKBACK_YEARS: Final[int] = 2


logger = logging.getLogger(LOGGER_NAME)


def _build_default_output_path(ticker: str) -> Path:
    """
    Build a default CSV output path for a given ticker symbol.

    The ticker is sanitized to create a filesystem-friendly filename.

    Parameters
    ----------
    ticker : str
        Asset identifier or ticker symbol (e.g., "^GSPC", "AAPL",
        "GC=F").

    Returns
    -------
    Path
        Path within the `data/` directory where historical prices for
        this ticker will be stored.
    """
    safe_name = (
        ticker.strip()
        .replace("^", "")
        .replace("=", "")
        .replace("/", "_")
        .replace(" ", "_")
    )
    if not safe_name:
        safe_name = "asset"
    return Path("data") / f"{safe_name.lower()}_historical.csv"


def _configure_logging() -> None:
    """
    Configure basic logging for the module.

    This sets up a simple stream handler with an informative log format.
    It is safe to call multiple times; repeated calls will not add
    duplicate handlers.
    """
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def download_asset_history(
    ticker: str,
    output_path: Path | None = None,
    lookback_years: int = LOOKBACK_YEARS,
) -> Path:
    """
    Download historical daily price data for a given asset.

    This function uses the `yfinance` package to fetch daily historical
    OHLCV data for the specified ticker (equity, index, ETF, or
    commodity) over the requested lookback window and saves it as a CSV
    file.

    Parameters
    ----------
    ticker : str
        Asset identifier or ticker symbol understood by Yahoo Finance
        (e.g., "^GSPC", "AAPL", "GC=F").
    output_path : Path, optional
        Filesystem path where the CSV file will be saved. If omitted,
        a sensible default will be derived from the ticker inside the
        `data/` directory.
    lookback_years : int, optional
        Number of years of historical data to retrieve, counting backward
        from the current date. Defaults to 2.

    Returns
    -------
    Path
        The path to the saved CSV file.

    Raises
    ------
    ValueError
        If `lookback_years` is not a positive integer or if no data is
        returned from `yfinance`.
    RuntimeError
        If there is an unexpected error while fetching data or writing
        the CSV file.
    """
    _configure_logging()

    if output_path is None:
        output_path = _build_default_output_path(ticker)

    if lookback_years <= 0:
        msg = "Parameter 'lookback_years' must be a positive integer."
        logger.error(msg)
        raise ValueError(msg)

    end_date = date.today()
    # Use a simple year-based delta; for most practical purposes this is
    # sufficient for a rolling 2-year window.
    start_date = end_date - timedelta(days=365 * lookback_years)

    logger.info(
        "Starting download of %s data from %s to %s.",
        ticker,
        start_date,
        end_date,
    )

    try:
        data: pd.DataFrame = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="1d",
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to download data for ticker {ticker!r}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if data.empty:
        msg = (
            f"No data returned for ticker {ticker!r} "
            f"between {start_date} and {end_date}."
        )
        logger.error(msg)
        raise ValueError(msg)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path)
        logger.info("Data successfully saved to %s.", output_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to save data to CSV at {output_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    return output_path


def download_sp500_history(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    lookback_years: int = LOOKBACK_YEARS,
) -> Path:
    """
    Convenience wrapper to download S&P 500 (^GSPC) historical data.

    This function preserves the original behavior for S&P 500-only use
    cases while delegating the actual work to
    :func:`download_asset_history`.
    """
    return download_asset_history(
        ticker=TICKER_SYMBOL,
        output_path=output_path,
        lookback_years=lookback_years,
    )


if __name__ == "__main__":
    _configure_logging()
    all_tickers = get_all_tickers()

    for ticker in all_tickers:
        output_path = Path("data") / "raw_prices" / f"{ticker}.csv"
        try:
            saved_path = download_asset_history(
                ticker=ticker,
                output_path=output_path,
            )
            print(f"Historical data for {ticker} successfully saved to: {saved_path}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Data collection failed for %s: %s", ticker, exc)


