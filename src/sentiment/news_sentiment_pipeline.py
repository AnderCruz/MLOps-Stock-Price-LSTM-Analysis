# news_sentiment_pipeline.py

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

from .news_sentiment_finnhub import process_finnhub_news_enhanced
from .synthetic_sentiment import create_intelligent_synthetic_sentiment

load_dotenv()


def apply_sentiment_pipeline(
    df_full: pd.DataFrame,
    ticker: str,
    use_sentiment: bool = True,
    min_news_required: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply sentiment enrichment and return both the enriched dataset
    and sentiment lineage metadata.
    """

    if not use_sentiment:
        df_full = df_full.copy()
        df_full["News_Sentiment"] = 0.0

        metadata = {
            "enabled": False,
            "source": "disabled",
            "headlines_processed": 0,
            "real_news_days": 0,
            "synthetic_days": 0,
            "fallback_used": False,
            "real_sentiment_coverage_percent": 0.0,
        }

        return df_full, metadata

    print("\n" + "=" * 60)
    print(f"🚀 STARTING SENTIMENT ANALYSIS FOR {ticker}")
    print("=" * 60)

    start_date = df_full.index.min().strftime("%Y-%m-%d")

    finnhub_api_key = os.getenv("FINNHUB_KEY", "")

    df_news_sentiment = process_finnhub_news_enhanced(
        ticker=ticker,
        start_date=start_date,
        api_key=finnhub_api_key,
    )

    close_col = f"{ticker}_Close"

    # --------------------------------------------------
    # No usable Finnhub data
    # --------------------------------------------------

    if (
        df_news_sentiment is None
        or len(df_news_sentiment) <= min_news_required
    ):
        print(
            "⚠️ Insufficient Finnhub data. "
            "Using intelligent synthetic sentiment."
        )

        synthetic_sentiment = create_intelligent_synthetic_sentiment(
            df_full,
            close_col,
        )

        df_full = df_full.copy()
        df_full["News_Sentiment"] = synthetic_sentiment

        metadata = {
            "enabled": True,
            "source": "synthetic_fallback",
            "headlines_processed": 0
            if df_news_sentiment is None
            else int(len(df_news_sentiment)),
            "real_news_days": 0,
            "synthetic_days": int(len(df_full)),
            "fallback_used": True,
            "real_sentiment_coverage_percent": 0.0,
        }

    # --------------------------------------------------
    # Real Finnhub data available
    # --------------------------------------------------

    else:
        print("✅ Using real news data from Finnhub")

        df_full = df_full.copy()

        df_full = df_full.join(
            df_news_sentiment[["sentiment_score"]],
            how="left",
        )

        missing_mask = df_full["sentiment_score"].isna()

        real_news_days = int((~missing_mask).sum())
        synthetic_days = int(missing_mask.sum())

        if missing_mask.any():
            print(
                f"🔧 Filling {synthetic_days} missing days "
                "with synthetic sentiment..."
            )

            synthetic_sentiment = create_intelligent_synthetic_sentiment(
                df_full,
                close_col,
            )

            df_full.loc[missing_mask, "sentiment_score"] = (
                synthetic_sentiment[missing_mask]
            )

        df_full["sentiment_score"] = (
            df_full["sentiment_score"].fillna(0)
        )

        df_full = df_full.rename(
            columns={"sentiment_score": "News_Sentiment"}
        )

        metadata = {
            "enabled": True,
            "source": "finnhub",
            "headlines_processed": int(len(df_news_sentiment)),
            "real_news_days": real_news_days,
            "synthetic_days": synthetic_days,
            "fallback_used": synthetic_days > 0,
            "real_sentiment_coverage_percent": round(
                (real_news_days / len(df_full)) * 100,
                2,
            ),
        }

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    print("\n📊 SENTIMENT STATISTICS:")
    print(
        f"➡ Mean:   "
        f"{df_full['News_Sentiment'].mean():.4f}"
    )
    print(
        f"➡ Std:    "
        f"{df_full['News_Sentiment'].std():.4f}"
    )
    print(
        f"➡ Min:    "
        f"{df_full['News_Sentiment'].min():.4f}"
    )
    print(
        f"➡ Max:    "
        f"{df_full['News_Sentiment'].max():.4f}"
    )

    correlation = df_full[close_col].corr(
        df_full["News_Sentiment"]
    )

    print(
        f"\n🔗 SENTIMENT vs PRICE CORRELATION: "
        f"{correlation:.4f}"
    )

    print("\n📋 SENTIMENT LINEAGE:")
    print(f"➡ Enabled: {metadata['enabled']}")
    print(f"➡ Source: {metadata['source']}")
    print(
        f"➡ Headlines processed: "
        f"{metadata['headlines_processed']}"
    )
    print(
        f"➡ Real-news days: "
        f"{metadata['real_news_days']}"
    )
    print(
        f"➡ Synthetic days: "
        f"{metadata['synthetic_days']}"
    )
    print(
        f"➡ Fallback used: "
        f"{metadata['fallback_used']}"
    )
    print(
        f"➡ Real sentiment coverage: "
        f"{metadata['real_sentiment_coverage_percent']:.2f}%"
    )

    return df_full, metadata
