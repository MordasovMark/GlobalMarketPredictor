from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

from config import ASSET_MAP


LOGGER_NAME: Final[str] = "data_preprocessor"
DEFAULT_STOCK_CSV: Final[Path] = Path("data") / "sp500_historical.csv"
DEFAULT_NEWS_CSV: Final[Path] = Path("data") / "news_with_sentiment.csv"
DEFAULT_OUTPUT_CSV: Final[Path] = Path("data") / "merged_dataset.csv"
ASSET_TYPE_COLUMN: Final[str] = "Asset_Type"


logger = logging.getLogger(LOGGER_NAME)


def _configure_logging() -> None:
    """
    Configure basic logging for the data preprocessor module.

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


def _load_csv(path: Path, description: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with error handling.

    Parameters
    ----------
    path : Path
        Filesystem path to the CSV file to be loaded.
    description : str
        Human-readable description of the CSV content, used in
        log messages and error reporting.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    RuntimeError
        If the CSV cannot be read for any other reason.
    ValueError
        If the resulting DataFrame is empty.
    """
    if not path.is_file():
        msg = f"{description} CSV file not found at {path}."
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to read {description} CSV file at {path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if df.empty:
        msg = f"{description} CSV file at {path} is empty."
        logger.error(msg)
        raise ValueError(msg)

    return df


def _detect_close_column(df: pd.DataFrame) -> str:
    """
    Detect the column representing the closing price in a DataFrame.

    This mirrors the logic used elsewhere in the project and supports
    different header conventions, including yfinance-style CSVs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing price information.

    Returns
    -------
    str
        Name of the detected price column.

    Raises
    ------
    ValueError
        If no suitable close-price column can be identified.
    """
    candidates = [
        "Close",
        "close",
        "Adj Close",
        "adj_close",
        "Adj_Close",
        "Price",
        "price",
        "Unnamed: 1",
    ]
    for col in candidates:
        if col in df.columns:
            logger.info("Using '%s' column as the Close price for preprocessing.", col)
            return col

    msg = (
        "Could not identify a close-price column in stock data. Expected one of "
        f"{candidates!r}, but found columns: {list(df.columns)!r}."
    )
    logger.error(msg)
    raise ValueError(msg)


def prepare_and_merge_data(
    stock_csv: str = str(DEFAULT_STOCK_CSV),
    news_csv: str = str(DEFAULT_NEWS_CSV),
    output_csv: str = str(DEFAULT_OUTPUT_CSV),
    asset_type: str = "Stock",
) -> pd.DataFrame:
    """
    Prepare and merge stock price data with sentiment data for modeling.

    This function:

    1. Loads historical stock data and sentiment-enriched news data.
    2. Extracts the date component from the news `published` column.
    3. Maps sentiment labels to numeric values (positive=1, neutral=0,
       negative=-1) and multiplies by the model confidence score to
       derive a weighted `sentiment_score`.
    4. Aggregates news sentiment by date (average daily sentiment).
    5. Normalizes stock dates to the same date format.
    6. Merges daily stock data with daily sentiment using a left join
       on date.
    7. Fills missing daily sentiment scores with 0 (neutral sentiment).
    8. Saves the merged dataset to the specified output CSV path.

    Parameters
    ----------
    stock_csv : str, optional
        Path to the stock historical data CSV file. Defaults to
        `"data/sp500_historical.csv"`.
    news_csv : str, optional
        Path to the sentiment-enriched news CSV file. Defaults to
        `"data/news_with_sentiment.csv"`.
    output_csv : str, optional
        Path where the merged dataset CSV will be saved. Defaults to
        `"data/merged_dataset.csv"`.
    asset_type : str, optional
        High-level asset classification associated with the price series
        (e.g., "Stock" or "Commodity"). This value is added as a feature
        column to the merged dataset so downstream models can learn
        cross-asset behavior. Defaults to "Stock".

    Returns
    -------
    pandas.DataFrame
        The merged dataset containing both stock features and daily
        sentiment scores.

    Raises
    ------
    FileNotFoundError
        If either the stock or news CSV file does not exist.
    ValueError
        If required columns are missing or if the loaded DataFrames are
        empty after processing.
    RuntimeError
        If an unexpected error occurs during transformation or while
        saving the merged dataset.
    """
    _configure_logging()

    stock_path = Path(stock_csv)
    news_path = Path(news_csv)
    output_path = Path(output_csv)

    logger.info(
        "Preparing and merging data. Stock CSV: %s, News CSV: %s, Output: %s",
        stock_path,
        news_path,
        output_path,
    )

    # Load CSV files
    stock_df = _load_csv(stock_path, "Stock")
    news_df = _load_csv(news_path, "News")

    # Validate required columns
    if "published" not in news_df.columns:
        msg = "News CSV must contain a 'published' column."
        logger.error(msg)
        raise ValueError(msg)

    if "sentiment_label" not in news_df.columns or "sentiment_score" not in news_df.columns:
        msg = (
            "News CSV must contain 'sentiment_label' and 'sentiment_score' "
            "columns produced by the sentiment analyzer."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Process news data: extract date and compute weighted sentiment score
    try:
        published_dt = pd.to_datetime(news_df["published"], errors="coerce")
        news_df = news_df.loc[~published_dt.isna()].copy()
        news_df["date"] = published_dt.loc[news_df.index].dt.normalize()

        if news_df.empty:
            msg = "No valid published dates found in news data after parsing."
            logger.error(msg)
            raise ValueError(msg)

        sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
        news_df["sentiment_label_numeric"] = news_df["sentiment_label"].str.lower().map(
            sentiment_mapping
        )

        # Drop rows where label mapping failed
        news_df = news_df.loc[~news_df["sentiment_label_numeric"].isna()].copy()
        if news_df.empty:
            msg = "No valid sentiment labels found in news data after mapping."
            logger.error(msg)
            raise ValueError(msg)

        news_df["sentiment_score"] = (
            news_df["sentiment_label_numeric"].astype(float)
            * news_df["sentiment_score"].astype(float)
        )

        daily_sentiment = (
            news_df.groupby("date", as_index=False)["sentiment_score"].mean()
        )
    except Exception as exc:  # noqa: BLE001
        msg = "Failed while processing news sentiment data."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if daily_sentiment.empty:
        msg = "Daily sentiment aggregation produced an empty result."
        logger.error(msg)
        raise ValueError(msg)

    # Process stock data: normalize dates and create multi-horizon targets
    try:
        # Handle yfinance CSVs that use multi-level headers and write the
        # date as a separate header row (commonly producing "Price,Close,..."
        # followed by "Ticker,...", "Date,...", then actual data). In such
        # cases, the straightforward read may not yield a 'Date' column.
        if "Date" not in stock_df.columns and "date" not in stock_df.columns:
            try:
                alt_stock_df = pd.read_csv(stock_path, skiprows=2)
            except Exception as exc:  # noqa: BLE001
                msg = (
                    "Failed to reload stock CSV with alternate header parsing "
                    f"from {stock_path}."
                )
                logger.exception(msg)
                raise RuntimeError(msg) from exc

            if not alt_stock_df.empty:
                stock_df = alt_stock_df

        if "Date" in stock_df.columns:
            stock_df["date"] = pd.to_datetime(
                stock_df["Date"], errors="coerce"
            ).dt.normalize()
        elif "date" in stock_df.columns:
            stock_df["date"] = pd.to_datetime(
                stock_df["date"], errors="coerce"
            ).dt.normalize()
        else:
            msg = (
                "Stock CSV must contain a 'Date' or 'date' column after "
                "standardization."
            )
            logger.error(msg)
            
            raise ValueError(msg)

        stock_df = stock_df.loc[~stock_df["date"].isna()].copy()

        # Create multi-horizon targets based on the detected close price.
        close_col = _detect_close_column(stock_df)
        stock_df["target_1d"] = (
            stock_df[close_col].shift(-1) > stock_df[close_col]
        ).astype(int)
        stock_df["target_7d"] = (
            stock_df[close_col].shift(-7) > stock_df[close_col]
        ).astype(int)
        stock_df["target_30d"] = (
            stock_df[close_col].shift(-30) > stock_df[close_col]
        ).astype(int)
    except Exception as exc:  # noqa: BLE001
        msg = "Failed while processing stock date information."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if stock_df.empty:
        msg = "Stock data is empty after processing date column."
        logger.error(msg)
        raise ValueError(msg)

    # Merge stock data with daily sentiment
    try:
        # Ensure both sides of the merge key are timezone-naive to avoid
        # mismatches such as datetime64[us] vs datetime64[us, UTC].
        for frame in (stock_df, daily_sentiment):
            try:
                frame["date"] = frame["date"].dt.tz_localize(None)
            except (TypeError, AttributeError, KeyError):
                # If the column is already tz-naive or missing, we simply
                # leave it as-is.
                continue

        merged_df = stock_df.merge(
            daily_sentiment,
            on="date",
            how="left",
        )
        merged_df["sentiment_score"] = merged_df["sentiment_score"].fillna(0.0)
    except Exception as exc:  # noqa: BLE001
        msg = "Failed while merging stock data with daily sentiment."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    # Save merged dataset: ensure Asset_Type is always set (for predictive_model).
    try:
        if ASSET_TYPE_COLUMN not in merged_df.columns and "ticker" in merged_df.columns:
            # Map ticker to category from config so Asset_Type matches our sectors.
            def _category_for_ticker(t: str) -> str:
                for category, tickers in ASSET_MAP.items():
                    if str(t).strip() in tickers:
                        return category
                return "Equity"
            merged_df[ASSET_TYPE_COLUMN] = merged_df["ticker"].astype(str).map(_category_for_ticker)
        if ASSET_TYPE_COLUMN not in merged_df.columns:
            normalized_asset_type = (asset_type or "Equity").strip().capitalize()
            merged_df[ASSET_TYPE_COLUMN] = normalized_asset_type
        if "asset_type" not in merged_df.columns:
            merged_df["asset_type"] = merged_df[ASSET_TYPE_COLUMN]
        normalized_asset_type = str(merged_df[ASSET_TYPE_COLUMN].iloc[0]) if len(merged_df) else (asset_type or "Equity")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        logger.info(
            "Merged dataset successfully saved to %s with Asset_Type set.",
            output_path,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to save merged dataset to CSV at {output_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    return merged_df


if __name__ == "__main__":
    try:
        merged = prepare_and_merge_data()
        print(
            "Data preparation and merge completed successfully. "
            f"Merged dataset saved to {DEFAULT_OUTPUT_CSV} "
            f"with {len(merged)} rows."
        )
    except Exception as exc:  # noqa: BLE001
        _configure_logging()
        logger.exception("Data preparation and merge failed: %s", exc)
        raise

