#!/usr/bin/env python3
"""
Export trained NRL models to Android-compatible .npz format,
copy to the android repo, and push to GitHub.

Automatically runs after Retrain in the desktop GUI.
Can also be run manually:
    python export_android.py
"""

import datetime
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ANDROID_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), "nrl_tips_android")

MODEL_PAIRS = [
    ("nrl_model.pkl",         "nrl_model.npz"),
    ("nrl_model_no_odds.pkl", "nrl_model_no_odds.npz"),
]
INFO_FILES = [
    "nrl_model_info.json",
    "nrl_model_no_odds_info.json",
]


# ── NPZ export ────────────────────────────────────────────────────────────────

def export_pkl_to_npz(pkl_path: str, npz_path: str):
    """Convert a CalibratedClassifierCV(GBC) .pkl to Android .npz format."""
    with open(pkl_path, "rb") as f:
        model, feature_cols = pickle.load(f)

    arrays = {
        "n_folds":      np.array([len(model.calibrated_classifiers_)]),
        "feature_cols": np.array(feature_cols),
    }

    for fi, cal_clf in enumerate(model.calibrated_classifiers_):
        gbc = cal_clf.estimator

        # Initial log-odds prediction from the dummy prior estimator
        p = float(gbc.init_.class_prior_[1])
        p = max(1e-7, min(1 - 1e-7, p))
        init_val = float(np.log(p / (1 - p)))

        # Isotonic calibration lookup tables
        calibrator = cal_clf.calibrators[0]
        iso_x = np.array(calibrator.X_thresholds_, dtype=np.float32)
        iso_y = np.array(calibrator.y_thresholds_, dtype=np.float32)

        # Decision trees — stored as object arrays (variable node counts per tree)
        n_trees   = len(gbc.estimators_)
        cl_list   = []
        cr_list   = []
        feat_list = []
        thr_list  = []
        val_list  = []

        for i in range(n_trees):
            tree = gbc.estimators_[i, 0].tree_
            arrays[f"f{fi}_t{i}_cl"]   = tree.children_left.astype(np.int32)
            arrays[f"f{fi}_t{i}_cr"]   = tree.children_right.astype(np.int32)
            arrays[f"f{fi}_t{i}_feat"] = tree.feature.astype(np.int32)
            arrays[f"f{fi}_t{i}_thr"]  = tree.threshold.astype(np.float32)
            arrays[f"f{fi}_t{i}_val"]  = tree.value[:, 0, 0].astype(np.float32)

        arrays[f"f{fi}_lr"]      = np.array([gbc.learning_rate])
        arrays[f"f{fi}_init"]    = np.array([init_val])
        arrays[f"f{fi}_n_trees"] = np.array([n_trees])
        arrays[f"f{fi}_iso_x"]   = iso_x
        arrays[f"f{fi}_iso_y"]   = iso_y

    np.savez(npz_path, **arrays)


# ── Round detection ───────────────────────────────────────────────────────────

def get_tip_round() -> tuple[int, int]:
    """
    Return (season, round_num) for the round we are currently tipping.
    = last completed round + 1, capped to current year.
    """
    data_path = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")
    if not os.path.exists(data_path):
        now = datetime.datetime.now()
        return now.year, 1

    df        = pd.read_csv(data_path, low_memory=False)
    completed = df[df["winner"].notna()][["season", "round"]]
    if completed.empty:
        return datetime.datetime.now().year, 1

    latest   = completed.sort_values(["season", "round"]).iloc[-1]
    season   = int(latest["season"])
    last_rnd = int(latest["round"])

    now = datetime.datetime.now()
    if season < now.year:
        season, last_rnd = now.year, 0

    return season, last_rnd + 1


# ── Main ──────────────────────────────────────────────────────────────────────

def _bundle_tip_caches() -> list[str]:
    """Generate tip cache files for all fully-completed rounds and copy to ANDROID_DIR.

    Only caches rounds where every game has a recorded winner (i.e. the round
    is completely done).  The current/upcoming round is intentionally excluded
    so the app still generates live tips right up until kickoff.

    Returns list of file names copied (for git add).
    """
    data_path = os.path.join(SCRIPT_DIR, "nrl_source_data.csv")
    if not os.path.exists(data_path):
        print("\n[skip] tip caches — nrl_source_data.csv not found")
        return []

    df = pd.read_csv(data_path)
    now_year = datetime.datetime.now().year

    # Find fully-completed rounds for the current season
    season_df  = df[df["season"] == now_year]
    all_rounds = sorted(season_df["round"].dropna().astype(int).unique())
    completed  = [
        r for r in all_rounds
        if season_df[season_df["round"] == r]["winner"].notna().all()
        and len(season_df[season_df["round"] == r]) > 0
    ]

    if not completed:
        print("\n[skip] tip caches — no fully completed rounds found")
        return []

    print(f"\nGenerating tip caches for {now_year} rounds: {completed} ...")
    tip_script = os.path.join(SCRIPT_DIR, "s6_tips.py")
    copied = []

    for rnd in completed:
        for no_odds in (False, True):
            model   = "no_odds" if no_odds else "odds"
            fname   = f"tips_cache_{now_year}_R{rnd:02d}_{model}.txt"
            dst     = os.path.join(ANDROID_DIR, fname)
            
            # 1. Copy the text cache (for _run_tips_cached in main.py)
            cmd = [sys.executable, tip_script,
                   "--season", str(now_year), "--round", str(rnd)]
            if no_odds:
                cmd.append("--no-odds")
            try:
                result = subprocess.run(
                    cmd, cwd=SCRIPT_DIR,
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0 and result.stdout.strip():
                    with open(dst, "w") as f:
                        f.write(result.stdout)
                    copied.append(fname)
                    print(f"  {fname}")
                else:
                    print(f"  [skip] {fname} — script returned no output")
            except Exception as e:
                print(f"  [skip] {fname} — {e}")

            # 2. Also copy the CSV file (for s9_performance.py / Compare Models)
            # Desktop s6_tips.py creates e.g. tips_2026_r1.csv and tips_2026_r1_no_odds.csv
            csv_suffix = "_no_odds" if no_odds else ""
            csv_name   = f"tips_{now_year}_r{rnd}{csv_suffix}.csv"
            src_csv    = os.path.join(SCRIPT_DIR, csv_name)
            dst_csv    = os.path.join(ANDROID_DIR, csv_name)

            if os.path.exists(src_csv):
                shutil.copy(src_csv, dst_csv)
                if csv_name not in copied:
                    copied.append(csv_name)
                print(f"  {csv_name}")

    return copied


def main():
    print("=" * 60)
    print("  Android model export")
    print("=" * 60)

    if not os.path.isdir(ANDROID_DIR):
        print(f"\n[error] Android repo not found at:\n  {ANDROID_DIR}")
        sys.exit(1)

    # Export each available pkl → npz
    exported_npz = []
    for pkl_name, npz_name in MODEL_PAIRS:
        pkl_path = os.path.join(SCRIPT_DIR, pkl_name)
        if not os.path.exists(pkl_path):
            print(f"  [skip] {pkl_name} — not trained yet")
            continue
        npz_path = os.path.join(SCRIPT_DIR, npz_name)
        print(f"\nConverting {pkl_name} → {npz_name} ...")
        try:
            export_pkl_to_npz(pkl_path, npz_path)
            print(f"  Done ({os.path.getsize(npz_path) / 1024:.0f} KB)")
            exported_npz.append(npz_name)
        except Exception as e:
            print(f"  [error] {e}")

    if not exported_npz:
        print("\nNo models exported — nothing to push.")
        sys.exit(1)

    # Copy npz + info files to android repo
    print(f"\nCopying to android repo ...")
    copied = []
    for npz_name in exported_npz:
        src = os.path.join(SCRIPT_DIR, npz_name)
        dst = os.path.join(ANDROID_DIR, npz_name)
        shutil.copy(src, dst)
        print(f"  {npz_name}")
        copied.append(npz_name)

    for info_name in INFO_FILES:
        src = os.path.join(SCRIPT_DIR, info_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(ANDROID_DIR, info_name))
            print(f"  {info_name}")
            copied.append(info_name)

    # Generate + bundle tip caches for all fully-completed rounds
    copied += _bundle_tip_caches()

    # Git commit + push
    season, rnd     = get_tip_round()
    date_str        = datetime.datetime.now().strftime("%Y-%m-%d")
    commit_msg      = f"Round {rnd} NRL tips model - {date_str}"

    print(f"\nPushing to GitHub ...")
    print(f"  Commit: \"{commit_msg}\"")

    try:
        subprocess.run(["git", "add"] + copied,
                       cwd=ANDROID_DIR, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ANDROID_DIR
        )
        if result.returncode == 0:
            print("  No changes to commit — models unchanged since last push.")
            return

        subprocess.run(["git", "commit", "-m", commit_msg],
                       cwd=ANDROID_DIR, check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"],
                       cwd=ANDROID_DIR, check=True, capture_output=True)
        print("  Pushed. GitHub Actions will rebuild the APK automatically.")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else ""
        print(f"  [error] git failed: {e}\n  {stderr}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
