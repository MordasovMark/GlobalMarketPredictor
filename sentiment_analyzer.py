from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, List, Tuple

import pandas as pd
from transformers import Pipeline, pipeline

from config import ASSETS


LOGGER_NAME: Final[str] = "sentiment_analyzer"
DEFAULT_INPUT_CSV: Final[Path] = Path("data") / "news_data.csv"
DEFAULT_OUTPUT_CSV: Final[Path] = Path("data") / "news_with_sentiment.csv"
MODEL_NAME: Final[str] = "ProsusAI/finbert"
NEWS_LOOKBACK_DAYS: Final[int] = 7


logger = logging.getLogger(LOGGER_NAME)


def _configure_logging() -> None:
    """
    Configure basic logging for the sentiment analyzer module.

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


def _load_sentiment_pipeline(model_name: str = MODEL_NAME) -> Pipeline:
    """
    Load the Hugging Face sentiment analysis pipeline for financial text.

    Parameters
    ----------
    model_name : str, optional
        Name or path of the Hugging Face model to load. Defaults to
        "ProsusAI/finbert".

    Returns
    -------
    transformers.Pipeline
        A configured sentiment-analysis pipeline instance.

    Raises
    ------
    RuntimeError
        If the model or pipeline cannot be loaded for any reason.
    """
    try:
        sentiment_pipe: Pipeline = pipeline(
            task="sentiment-analysis",
            model=model_name,
        )
        return sentiment_pipe
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load sentiment analysis model {model_name!r}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc


def analyze_sentiment(
    input_csv: str = str(DEFAULT_INPUT_CSV),
    output_csv: str = str(DEFAULT_OUTPUT_CSV),
    sentiment_pipeline: Pipeline | None = None,
) -> pd.DataFrame:
    """
    Analyze sentiment of financial news titles and save results to CSV.

    This function reads a CSV file containing financial news, applies a
    FinBERT-based sentiment analysis model to the `title` column, and
    writes the enriched data (including sentiment label and confidence
    score) to an output CSV file. Any existing columns (e.g. ``source``
    from the targeted news resolver) are preserved for dashboard display.

    Parameters
    ----------
    input_csv : str, optional
        Path to the input CSV file containing financial news with at
        least a `title` column. Defaults to `"data/news_data.csv"`.
    output_csv : str, optional
        Path where the output CSV file with sentiment columns will be
        saved. Defaults to `"data/news_with_sentiment.csv"`.
    sentiment_pipeline : Pipeline, optional
        Preloaded Hugging Face sentiment-analysis pipeline. If not
        provided, a new pipeline instance will be loaded internally.

    Returns
    -------
    pandas.DataFrame
        A DataFrame that includes the original columns plus two new
        columns:

        - `sentiment_label`: model-predicted sentiment label
        - `sentiment_score`: model confidence score for that label

    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist.
    ValueError
        If the input CSV is empty or does not contain a `title` column.
    RuntimeError
        If an unexpected error occurs while reading data, running the
        model, or writing the output CSV.
    """
    _configure_logging()

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    logger.info("Starting sentiment analysis for input file: %s", input_path)

    if not input_path.is_file():
        msg = f"Input CSV file not found at {input_path}."
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to read input CSV file at {input_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    if df.empty:
        msg = f"Input CSV file at {input_path} is empty."
        logger.error(msg)
        raise ValueError(msg)

    if "title" not in df.columns:
        msg = "Input CSV must contain a 'title' column for sentiment analysis."
        logger.error(msg)
        raise ValueError(msg)

    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        pub = df["published"]
        if pub.dt.tz is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)).replace(tzinfo=None)
            pub = pub.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            cutoff = datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)
        df = df[pub.notna() & (pub >= cutoff)].copy()
        logger.info("Filtered to %d articles within the last %d days.", len(df), NEWS_LOOKBACK_DAYS)
    if df.empty:
        msg = "No news within the lookback window after date filter."
        logger.warning(msg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    sentiment_pipe = sentiment_pipeline or _load_sentiment_pipeline()

    try:
        titles = df["title"].astype(str).tolist()
        logger.info("Running sentiment analysis on %d titles.", len(titles))

        # The pipeline can operate on a list of strings and will return
        # a list of results in the same order.
        results = sentiment_pipe(titles)
    except Exception as exc:  # noqa: BLE001
        msg = "Failed during sentiment inference on news titles."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    try:
        df["sentiment_label"] = [str(r.get("label", "")) for r in results]
        df["sentiment_score"] = [float(r.get("score", 0.0)) for r in results]
    except Exception as exc:  # noqa: BLE001
        msg = "Failed to attach sentiment results to DataFrame."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Sentiment-enriched data successfully saved to %s.", output_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to save sentiment-enriched data to CSV at {output_path}."
        logger.exception(msg)
        raise RuntimeError(msg) from exc

    return df


def _iter_asset_news_files() -> List[Tuple[str, Path, Path]]:
    """
    Build a list of per-asset news input and output paths.

    Returns
    -------
    list[tuple[str, Path, Path]]
        A list of (ticker, input_path, output_path) tuples.
    """
    items: List[Tuple[str, Path, Path]] = []
    base_dir = Path("data") / "raw_news"

    for group_name, group in ASSETS.items():
        if isinstance(group, dict):
            tickers = [str(ticker) for ticker in group.values()]
        else:
            tickers = [str(ticker) for ticker in group]

        for ticker in tickers:
            input_path = base_dir / f"{ticker}_news.csv"
            output_path = base_dir / f"{ticker}_news_with_sentiment.csv"
            items.append((ticker, input_path, output_path))

    return items


if __name__ == "__main__":
    _configure_logging()
    try:
        shared_pipeline = _load_sentiment_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize sentiment pipeline: %s", exc)
        raise

    for ticker, input_path, output_path in _iter_asset_news_files():
        try:
            enriched_df = analyze_sentiment(
                input_csv=str(input_path),
                output_csv=str(output_path),
                sentiment_pipeline=shared_pipeline,
            )
            print(
                "Sentiment analysis completed successfully for "
                f"{ticker}. Results saved to {output_path} "
                f"with {len(enriched_df)} records."
            )
        except FileNotFoundError:
            logger.warning(
                "Skipping sentiment analysis for %s; input file not found at %s.",
                ticker,
                input_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Sentiment analysis failed for %s using %s: %s",
                ticker,
                input_path,
                exc,
            )


