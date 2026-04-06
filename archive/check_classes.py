import argparse
from pathlib import Path

import pandas as pd


def _load_dataframe(parquet_path: Path | None, csv_path: Path | None) -> pd.DataFrame:
    if parquet_path:
        return pd.read_parquet(parquet_path)
    if csv_path:
        return pd.read_csv(csv_path)
    raise ValueError("Provide --parquet or --csv")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["class", "subClass"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            raise KeyError(f"Missing column: {col}")
    df["subClass"] = df["subClass"].replace({"": "UNKNOWN", "nan": "UNKNOWN"}).fillna("UNKNOWN")
    return df


def summarize_subclasses(df: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    subclass_counts = (
        df.groupby("subClass")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    subclass_classes = (
        df.groupby("subClass")["class"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index(name="classes")
    )
    summary = subclass_counts.merge(subclass_classes, on="subClass", how="left")
    summary = summary[summary["count"] >= min_count]

    detailed = (
        df.groupby(["class", "subClass"])
        .size()
        .reset_index(name="count")
        .sort_values(["class", "count"], ascending=[True, False])
    )

    return summary, detailed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize SDSS subclasses: types, counts, and classes."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("./sdss_merged_full.parquet"),
        help="Path to merged parquet with class/subClass columns.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV with class/subClass columns (used if --parquet not set).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only show subclasses with at least this many rows.",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=None,
        help="Write subclass summary to CSV.",
    )
    parser.add_argument(
        "--output-detailed-csv",
        type=Path,
        default=None,
        help="Write class/subClass counts to CSV.",
    )

    args = parser.parse_args()

    parquet_path = args.parquet if args.parquet and args.parquet.exists() else None
    df = _load_dataframe(parquet_path, args.csv)
    df = _normalize_columns(df)

    print(f"Rows: {len(df):,}")
    print(f"Unique classes: {df['class'].nunique()}")
    print(f"Unique subclasses: {df['subClass'].nunique()}")

    summary, detailed = summarize_subclasses(df, args.min_count)

    print("\nSubclass summary (count, classes):")
    print(summary.to_string(index=False))

    print("\nClass/SubClass counts:")
    print(detailed.to_string(index=False))

    if args.output_summary_csv:
        summary.to_csv(args.output_summary_csv, index=False)
        print(f"Wrote summary CSV to {args.output_summary_csv}")

    if args.output_detailed_csv:
        detailed.to_csv(args.output_detailed_csv, index=False)
        print(f"Wrote detailed CSV to {args.output_detailed_csv}")


if __name__ == "__main__":
    main()

