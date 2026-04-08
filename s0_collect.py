#!/usr/bin/env python3
"""
Data collection pipeline.

Usage:
  python collect_data.py            # full rebuild (all seasons)
  python collect_data.py --new-only # only rounds added since last collection
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")


def run_step(label: str, script: str, extra_args: list[str] = []):
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'=' * 60}", flush=True)
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, script)] + extra_args,
        cwd=SCRIPT_DIR,
    )
    if result.returncode != 0:
        print(f"\n[!] {script} exited with code {result.returncode}", flush=True)


def existing_rounds() -> set[tuple]:
    """Return the set of (season, round) pairs already in the CSV."""
    if not os.path.exists(DATA_PATH):
        return set()
    df = pd.read_csv(DATA_PATH)
    completed = df[df["winner"].notna()]
    return set(zip(completed["season"].astype(int), completed["round"].astype(int)))


def collect_new():
    print("NRL Data Collection — new rounds only", flush=True)

    # Snapshot what rounds we already have before fetching
    before = existing_rounds()

    # Step 1: fetch only rounds not yet in the CSV
    run_step("Step 1/4 — New match results", "s1_history.py", ["--new-only"])

    # Work out exactly which rounds were just added
    after     = existing_rounds()
    new_rounds = sorted(after - before)   # list of (season, round) tuples

    if not new_rounds:
        print("\nNo new completed games found. Already up to date.", flush=True)
        print(f"{'=' * 60}", flush=True)
        return

    print(f"\n  {len(new_rounds)} new round(s) added: "
          f"{', '.join(f'S{s}R{r}' for s, r in new_rounds)}", flush=True)

    # Steps 2 & 3: run only for the newly added rounds
    for season, round_num in new_rounds:
        run_step(
            f"Step 2 — Advanced stats  S{season} R{round_num}",
            "s2_stats.py",
            ["--season", str(season), "--round", str(round_num)],
        )
        run_step(
            f"Step 3 — Squad info      S{season} R{round_num}",
            "s4_squads.py",
            ["--season", str(season), "--round", str(round_num)],
        )

    run_step("Step 4/4 — Weather data (new games)", "s3_weather.py")
    run_step("Step 5/5 — Odds data (download latest)", "s5_odds.py")

    print(f"\n{'=' * 60}", flush=True)
    print("  All done! Click Retrain Model to use the new data.", flush=True)
    print(f"{'=' * 60}", flush=True)


def collect_all():
    print("NRL Data Collection — full rebuild", flush=True)

    run_step("Step 1/5 — Full match history (2020→now)", "s1_history.py")
    run_step("Step 2/5 — Advanced stats (all seasons)",  "s2_stats.py")
    run_step("Step 3/5 — Squad info (all seasons)",      "s4_squads.py")
    run_step("Step 4/5 — Weather data (all games)",      "s3_weather.py")
    run_step("Step 5/5 — Odds data (download latest)",   "s5_odds.py")

    print(f"\n{'=' * 60}", flush=True)
    print("  All done! Click Retrain Model to update predictions.", flush=True)
    print(f"{'=' * 60}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only", action="store_true")
    args = parser.parse_args()
    collect_new() if args.new_only else collect_all()


if __name__ == "__main__":
    main()
