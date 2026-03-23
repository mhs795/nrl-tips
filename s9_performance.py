#!/usr/bin/env python3
"""
NRL Tips — performance checker.
Finds the most recent tips CSV that has actual results available in
nrl_source_data.csv and prints a match-by-match breakdown plus summary.

Usage:
  python s9_performance.py              # auto-detect last completed round
  python s9_performance.py --round 4   # specific round
  python s9_performance.py --season 2025 --round 4
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")


def _find_tips_files():
    """Return list of (season, round, path) sorted newest-first."""
    pattern = os.path.join(SCRIPT_DIR, "tips_*_r*.csv")
    results = []
    for path in glob.glob(pattern):
        m = re.search(r"tips_(\d{4})_r(\d+)\.csv$", path)
        if m:
            results.append((int(m.group(1)), int(m.group(2)), path))
    results.sort(reverse=True)
    return results


def _load_actual(season: int, round_num: int) -> pd.DataFrame:
    """Return rows from source data for the given season/round."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    mask = (df["season"] == season) & (df["round"] == round_num)
    return df[mask][["home_team", "away_team", "home_score", "away_score", "winner", "date"]].copy()


def check_performance(season: int | None = None, round_num: int | None = None):
    tips_files = _find_tips_files()
    if not tips_files:
        print("No tips CSV files found.")
        sys.exit(1)

    # Auto-detect: walk newest-first, pick first one with actual results
    candidates = tips_files
    if season is not None:
        candidates = [(s, r, p) for s, r, p in candidates if s == season]
    if round_num is not None:
        candidates = [(s, r, p) for s, r, p in candidates if r == round_num]

    chosen = None
    for s, r, path in candidates:
        actual = _load_actual(s, r)
        if not actual.empty:
            chosen = (s, r, path, actual)
            break

    if chosen is None:
        if season or round_num:
            print(f"No results found in source data for the requested round.")
        else:
            print("No completed rounds found in source data yet — check back after the round is done.")
        sys.exit(0)

    s, r, path, actual = chosen
    tips = pd.read_csv(path)

    # Sort tips by game date (schedule order) using dates from source data
    tips = tips.merge(
        actual[["home_team", "away_team", "date"]],
        on=["home_team", "away_team"],
        how="left",
    ).sort_values("date").drop(columns=["date"])

    print(f"\n{'='*60}")
    print(f"  TIPS PERFORMANCE — {s} Round {r}")
    print(f"{'='*60}\n")

    correct = 0
    total   = 0

    for _, tip in tips.iterrows():
        home = tip["home_team"]
        away = tip["away_team"]
        pred = tip["predicted_winner"]
        conf = tip.get("confidence", tip.get("home_win_prob", "?"))

        # Match to actual result
        row = actual[
            (actual["home_team"] == home) & (actual["away_team"] == away)
        ]
        if row.empty:
            print(f"  {home} vs {away}")
            print(f"    Predicted : {pred}")
            print(f"    Actual    : result not found\n")
            continue

        row = row.iloc[0]
        actual_winner = home if row["winner"] == "home" else away
        home_score    = int(row["home_score"])
        away_score    = int(row["away_score"])
        result_str    = f"{home} {home_score} – {away_score} {away}"

        hit = pred == actual_winner
        correct += int(hit)
        total   += 1
        marker   = "✔" if hit else "✘"

        print(f"  {marker}  {home} vs {away}")
        print(f"     Predicted : {pred}  (conf {conf:.1f}%)")
        print(f"     Result    : {result_str}  →  {actual_winner} won\n")

    print(f"{'='*60}")
    if total:
        pct = correct / total * 100
        print(f"  RESULT: {correct}/{total} correct  ({pct:.0f}%)")
    else:
        print("  No matched games found.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Check NRL tips performance")
    parser.add_argument("--round",  type=int, default=None)
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    check_performance(season=args.season, round_num=args.round)


if __name__ == "__main__":
    main()
