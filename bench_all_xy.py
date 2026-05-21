#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

rows = []

for p in sorted(Path("outputs").glob("*/bench_xy.csv")):
    df = pd.read_csv(p)
    rows.append(df)

df = pd.concat(rows, ignore_index=True)

clean = df[~df["leaky"]].copy()

clean["rank"] = clean.groupby("dataset")["median_vs_center"].rank(method="min")

summary = (
    clean
    .groupby("measurement")
    .agg(
        datasets=("dataset", "nunique"),
        wins=("rank", lambda x: int((x == 1).sum())),
        rank_mean=("rank", "mean"),
        median_vs_center_mean=("median_vs_center", "mean"),
        median_vs_center_median=("median_vs_center", "median"),
        median_xy_error_median=("median_xy_error", "median"),
    )
    .reset_index()
    .sort_values(["wins", "rank_mean", "median_vs_center_median"], ascending=[False, True, True])
)

print(summary.to_string(index=False))

best = (
    clean
    .sort_values(["dataset", "median_vs_center"])
    .groupby("dataset")
    .head(1)
    [["dataset", "measurement", "median_vs_center", "median_xy_error"]]
)

print()
print(best.to_string(index=False))

best.to_csv("outputs/bench_xy_best.csv", index=False)

summary.to_csv("outputs/bench_xy_summary.csv", index=False)
df.to_csv("outputs/bench_xy_all.csv", index=False)
