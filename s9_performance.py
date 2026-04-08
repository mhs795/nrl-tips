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


def _find_tips_files(no_odds: bool = False):
    """Return list of (season, round, path) sorted newest-first."""
    pattern = os.path.join(SCRIPT_DIR, "tips_*_r*.csv")
    results = []
    for path in glob.glob(pattern):
        if no_odds:
            m = re.search(r"tips_(\d{4})_r(\d+)_no_odds\.csv$", path)
        else:
            m = re.search(r"tips_(\d{4})_r(\d+)\.csv$", path)
            # Exclude no_odds files when looking for standard tips
            if m and path.endswith("_no_odds.csv"):
                m = None
        if m:
            results.append((int(m.group(1)), int(m.group(2)), path))
    results.sort(reverse=True)
    return results


def _load_actual(season: int, round_num: int) -> pd.DataFrame:
    """Return rows from source data for the given season/round."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    mask = (df["season"] == season) & (df["round"] == round_num)
    return df[mask][["home_team", "away_team", "home_score", "away_score", "winner", "date"]].copy()


def check_performance(season: int | None = None, round_num: int | None = None, no_odds: bool = False):
    tips_files = _find_tips_files(no_odds=no_odds)
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

    # Sort tips by game date — use date from CSV if present, else join from source data
    if "date" not in tips.columns:
        tips = tips.merge(
            actual[["home_team", "away_team", "date"]],
            on=["home_team", "away_team"],
            how="left",
        )
    tips = tips.sort_values("date").drop(columns=["date"])

    model_label = "no odds" if no_odds else "with odds"
    print(f"\n{'='*60}")
    print(f"  TIPS PERFORMANCE — {s} Round {r}  [{model_label}]")
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


def compare_models(season: int | None = None, round_num: int | None = None):
    """Side-by-side comparison of the odds and no-odds models for the same round."""
    odds_files    = _find_tips_files(no_odds=False)
    no_odds_files = _find_tips_files(no_odds=True)

    if not odds_files and not no_odds_files:
        print("No tips CSV files found for either model.")
        sys.exit(1)

    # Build lookup for quick existence checks
    no_odds_map = {(s, r): p for s, r, p in no_odds_files}

    # Filter by season/round if supplied
    if season is not None:
        odds_files = [(s, r, p) for s, r, p in odds_files if s == season]
    if round_num is not None:
        odds_files = [(s, r, p) for s, r, p in odds_files if r == round_num]

    # Find the newest round with actual results for BOTH models
    chosen = None
    for s, r, odds_path in odds_files:
        if (s, r) not in no_odds_map:
            continue
        actual = _load_actual(s, r)
        if not actual.empty:
            chosen = (s, r, odds_path, no_odds_map[(s, r)], actual)
            break

    if chosen is None:
        print("No completed round found with tips from both models.")
        print("Make sure you have generated tips using both 'Tips (with Odds)' and 'Tips (No Odds)'.")
        sys.exit(0)

    s, r, odds_path, no_odds_path, actual = chosen
    tips_odds    = pd.read_csv(odds_path)
    tips_no_odds = pd.read_csv(no_odds_path)

    # Sort by game date — use date from CSV if present, else join from source data
    if "date" not in tips_odds.columns:
        tips_odds = tips_odds.merge(
            actual[["home_team", "away_team", "date"]], on=["home_team", "away_team"], how="left"
        )
    tips_odds = tips_odds.sort_values("date").drop(columns=["date"])

    print(f"\n{'='*72}")
    print(f"  MODEL COMPARISON — {s} Round {r}")
    print(f"  {'with odds':^30}  vs  {'no odds':^30}")
    print(f"{'='*72}\n")

    odds_correct = odds_total = no_odds_correct = no_odds_total = 0
    agree_correct = agree_total = disagree_odds_correct = disagree_total = 0

    for _, tip in tips_odds.iterrows():
        home = tip["home_team"]
        away = tip["away_team"]

        actual_row = actual[(actual["home_team"] == home) & (actual["away_team"] == away)]
        if actual_row.empty:
            continue
        actual_row   = actual_row.iloc[0]
        actual_winner = home if actual_row["winner"] == "home" else away
        home_score   = int(actual_row["home_score"])
        away_score   = int(actual_row["away_score"])

        pred_odds = tip["predicted_winner"]
        conf_odds = tip.get("confidence", tip.get("home_win_prob", 0))

        no_odds_row = tips_no_odds[
            (tips_no_odds["home_team"] == home) & (tips_no_odds["away_team"] == away)
        ]
        if no_odds_row.empty:
            continue
        no_odds_row  = no_odds_row.iloc[0]
        pred_no_odds = no_odds_row["predicted_winner"]
        conf_no_odds = no_odds_row.get("confidence", no_odds_row.get("home_win_prob", 0))

        hit_odds    = pred_odds    == actual_winner
        hit_no_odds = pred_no_odds == actual_winner
        agree       = pred_odds    == pred_no_odds

        odds_correct    += int(hit_odds);    odds_total    += 1
        no_odds_correct += int(hit_no_odds); no_odds_total += 1

        mk_odds    = "✔" if hit_odds    else "✘"
        mk_no_odds = "✔" if hit_no_odds else "✘"

        if agree:
            agree_total   += 1
            agree_correct += int(hit_odds)
            agree_str = "  [agree]"
        else:
            disagree_total        += 1
            disagree_odds_correct += int(hit_odds)
            agree_str = "  [DIFFER]"

        print(f"  {home} vs {away}")
        print(f"    Odds   : {mk_odds} {pred_odds:<28} ({conf_odds:.0f}%)")
        print(f"    No Odds: {mk_no_odds} {pred_no_odds:<28} ({conf_no_odds:.0f}%)  {agree_str}")
        print(f"    Result : {home} {home_score} – {away_score} {away}  →  {actual_winner} won\n")

    print(f"{'='*72}")
    if odds_total:
        o_pct  = odds_correct    / odds_total    * 100
        n_pct  = no_odds_correct / no_odds_total * 100
        winner = "with odds" if odds_correct > no_odds_correct else ("no odds" if no_odds_correct > odds_correct else "tie")
        print(f"  WITH ODDS:  {odds_correct}/{odds_total} correct  ({o_pct:.0f}%)")
        print(f"  NO ODDS:    {no_odds_correct}/{no_odds_total} correct  ({n_pct:.0f}%)")
        print(f"  Overall winner: {winner}")
        if agree_total:
            a_pct = agree_correct / agree_total * 100
            print(f"\n  Agreed on {agree_total} games → {agree_correct}/{agree_total} correct ({a_pct:.0f}%)")
        if disagree_total:
            d_pct = disagree_odds_correct / disagree_total * 100
            print(f"  Differed on {disagree_total} games → odds model got {disagree_odds_correct}/{disagree_total} ({d_pct:.0f}%) of those")
    else:
        print("  No matched games found.")
    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Check NRL tips performance")
    parser.add_argument("--round",    type=int, default=None)
    parser.add_argument("--season",   type=int, default=None)
    parser.add_argument("--no-odds",  action="store_true", help="Check no-odds model tips")
    parser.add_argument("--compare",  action="store_true", help="Compare odds vs no-odds model")
    args = parser.parse_args()
    if args.compare:
        compare_models(season=args.season, round_num=args.round)
    else:
        check_performance(season=args.season, round_num=args.round, no_odds=args.no_odds)


if __name__ == "__main__":
    main()
