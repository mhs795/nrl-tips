"""
NRL Tipping Model — XGBoost-based winner predictor
Usage:
  python nrl_model.py --train                   # train on historical data
  python nrl_model.py --predict round.csv       # predict upcoming round
  python nrl_model.py --train --predict round.csv
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_PATH  = "nrl_source_data.csv"
MODEL_PATH = "nrl_model.pkl"

NRL_TEAMS = [
    "Brisbane Broncos", "Canberra Raiders", "Canterbury Bulldogs",
    "Cronulla Sharks", "Gold Coast Titans", "Manly Sea Eagles",
    "Melbourne Storm", "Newcastle Knights", "New Zealand Warriors",
    "North Queensland Cowboys", "Parramatta Eels", "Penrith Panthers",
    "South Sydney Rabbitohs", "St George Illawarra Dragons",
    "Sydney Roosters", "Wests Tigers", "Dolphins", "Dolphins FC",
]

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw data columns."""
    df = df.copy()

    # --- Differential features (home minus away) ---
    df["ladder_diff"]          = df["away_ladder_pos"] - df["home_ladder_pos"]   # positive = home higher
    df["form_diff"]            = df["home_last5_wins"] - df["away_last5_wins"]
    df["pts_diff_diff"]        = df["home_last5_pts_diff_avg"] - df["away_last5_pts_diff_avg"]
    df["pts_for_diff"]         = df["home_season_pts_for_avg"] - df["away_season_pts_for_avg"]
    df["pts_against_diff"]     = df["away_season_pts_against_avg"] - df["home_season_pts_against_avg"]
    df["rest_advantage"]       = df["home_days_rest"] - df["away_days_rest"]
    df["travel_diff"]          = df["away_travel_km"] - df["home_travel_km"]
    df["streak_diff"]          = df["home_win_streak"] - df["away_win_streak"]
    df["key_players_diff"]     = df["away_key_players_out"] - df["home_key_players_out"]
    df["origin_diff"]          = df["away_origin_players_out"] - df["home_origin_players_out"]

    # --- Home ground win rate ---
    df["home_venue_winrate"]   = np.where(
        df["home_home_record_played"] > 0,
        df["home_home_record_wins"] / df["home_home_record_played"],
        0.5
    )
    df["away_venue_winrate"]   = np.where(
        df["away_away_record_played"] > 0,
        df["away_away_record_wins"] / df["away_away_record_played"],
        0.5
    )
    df["venue_winrate_diff"]   = df["home_venue_winrate"] - df["away_venue_winrate"]

    # --- Season win rate ---
    df["home_season_winrate"]  = (
        df["home_season_wins"] /
        (df["home_season_wins"] + df["home_season_losses"] + df["home_season_draws"]).clip(lower=1)
    )
    df["away_season_winrate"]  = (
        df["away_season_wins"] /
        (df["away_season_wins"] + df["away_season_losses"] + df["away_season_draws"]).clip(lower=1)
    )
    df["season_winrate_diff"]  = df["home_season_winrate"] - df["away_season_winrate"]

    # --- Head-to-head home win rate ---
    df["h2h_home_winrate"]     = np.where(
        df["h2h_total_meetings"] > 0,
        df["h2h_home_wins"] / df["h2h_total_meetings"],
        0.5
    )
    df["h2h_pts_diff"]         = df["h2h_home_pts_for_avg"] - df["h2h_away_pts_for_avg"]
    df["h2h_recent_winrate"]   = np.where(
        df["h2h_last3yr_meetings"] > 0,
        df["h2h_last3yr_home_wins"] / df["h2h_last3yr_meetings"],
        0.5
    )

    # --- Advanced stats differentials ---
    df["completion_diff"]      = df["home_completion_rate"] - df["away_completion_rate"]
    df["errors_diff"]          = df["away_errors_pg"] - df["home_errors_pg"]           # lower errors = better
    df["penalties_diff"]       = df["away_penalties_pg"] - df["home_penalties_pg"]
    df["tackle_eff_diff"]      = df["home_tackle_eff"] - df["away_tackle_eff"]
    df["line_breaks_diff"]     = df["home_line_breaks_pg"] - df["away_line_breaks_pg"]
    df["post_contact_diff"]    = df["home_post_contact_metres_pg"] - df["away_post_contact_metres_pg"]
    df["kick_metres_diff"]     = df["home_kick_metres_pg"] - df["away_kick_metres_pg"]

    # --- Market implied probability ---
    if "market_home_win_odds" in df.columns and "market_away_win_odds" in df.columns:
        home_impl = 1 / df["market_home_win_odds"].replace(0, np.nan)
        away_impl = 1 / df["market_away_win_odds"].replace(0, np.nan)
        total_impl = home_impl + away_impl
        df["market_home_implied_prob"] = (home_impl / total_impl).fillna(0.5)
        df["market_handicap"]          = df["market_home_handicap"].fillna(0)
        # line movement signal
        if "market_open_home_odds" in df.columns:
            df["odds_move"] = df["market_open_home_odds"] - df["market_home_win_odds"]
        else:
            df["odds_move"] = 0.0

    # --- Weather impact ---
    if "weather_rain_mm" in df.columns:
        df["wet_weather"] = (df["weather_rain_mm"] > 2).astype(int)
    if "weather_wind_kmh" in df.columns:
        df["strong_wind"] = (df["weather_wind_kmh"] > 30).astype(int)

    # --- Context flags ---
    df["is_finals"]            = df.get("is_finals", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["is_neutral_venue"]     = df.get("is_neutral_venue", pd.Series(0, index=df.index)).fillna(0).astype(int)

    return df


FEATURE_COLS = [
    # Differential / relative
    "ladder_diff", "form_diff", "pts_diff_diff", "pts_for_diff", "pts_against_diff",
    "rest_advantage", "travel_diff", "streak_diff", "key_players_diff", "origin_diff",
    "venue_winrate_diff", "season_winrate_diff",
    "h2h_home_winrate", "h2h_pts_diff", "h2h_recent_winrate",
    # Advanced stats
    "completion_diff", "errors_diff", "penalties_diff", "tackle_eff_diff",
    "line_breaks_diff", "post_contact_diff", "kick_metres_diff",
    # Market
    "market_home_implied_prob", "market_handicap", "odds_move",
    # Context
    "wet_weather", "strong_wind", "is_finals", "is_neutral_venue",
    # Raw individual (model learns interactions)
    "home_last5_wins", "away_last5_wins",
    "home_season_winrate", "away_season_winrate",
    "home_win_streak", "away_win_streak",
    "home_days_rest", "away_days_rest",
    "home_ladder_pos", "away_ladder_pos",
    "home_key_players_out", "away_key_players_out",
    "home_completion_rate", "away_completion_rate",
    "home_errors_pg", "away_errors_pg",
    "home_tackle_eff", "away_tackle_eff",
    "home_line_breaks_pg", "away_line_breaks_pg",
    "home_venue_winrate", "away_venue_winrate",
]

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = engineer_features(df)
    return df


def prepare_training(df: pd.DataFrame):
    """Return X, y for labelled games (winner column populated)."""
    labelled = df[df["winner"].notna()].copy()
    labelled["target"] = (labelled["winner"] == "home").astype(int)

    # Keep only columns that exist in FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in labelled.columns]
    X = labelled[available].fillna(0)
    y = labelled["target"]
    return X, y, available


# ─── MODEL TRAINING ───────────────────────────────────────────────────────────

def build_model():
    base = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    # Isotonic calibration gives better probability estimates
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    return model


def train(df: pd.DataFrame):
    X, y, feature_cols = prepare_training(df)

    if len(X) < 20:
        print(f"  [!] Only {len(X)} labelled games — model will have low confidence. Add more historical data.")

    model = build_model()

    # Cross-validate first
    raw_xgb = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    )
    cv_scores = cross_val_score(raw_xgb, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring="accuracy")
    print(f"\n  Cross-validated accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit calibrated model on full data
    model.fit(X, y)

    # In-sample metrics
    preds     = model.predict(X)
    probs     = model.predict_proba(X)[:, 1]
    acc       = accuracy_score(y, preds)
    ll        = log_loss(y, probs)
    brier     = brier_score_loss(y, probs)
    print(f"  Training accuracy:        {acc:.3f}")
    print(f"  Log loss:                 {ll:.4f}")
    print(f"  Brier score:              {brier:.4f}  (lower = better; random ≈ 0.25)")

    # Feature importance from the underlying estimators
    importances = np.zeros(len(feature_cols))
    for est in model.calibrated_classifiers_:
        importances += est.estimator.feature_importances_
    importances /= len(model.calibrated_classifiers_)
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
    print("\n  Top 15 features:")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"    {row['feature']:<35} {bar} ({row['importance']:.4f})")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, feature_cols), f)
    print(f"\n  Model saved → {MODEL_PATH}")
    return model, feature_cols


# ─── PREDICTION ───────────────────────────────────────────────────────────────

def predict(predict_path: str, model=None, feature_cols=None):
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run --train first.")
        with open(MODEL_PATH, "rb") as f:
            model, feature_cols = pickle.load(f)

    df_pred = load_data(predict_path)
    available = [c for c in feature_cols if c in df_pred.columns]
    missing   = [c for c in feature_cols if c not in df_pred.columns]
    if missing:
        print(f"  [!] Missing columns (will use 0): {missing}")

    X_pred = df_pred[available].reindex(columns=feature_cols, fill_value=0).fillna(0)

    probs  = model.predict_proba(X_pred)[:, 1]   # P(home wins)
    preds  = (probs >= 0.5).astype(int)

    results = df_pred[["round", "date", "home_team", "away_team"]].copy()
    results["home_win_prob"]  = (probs * 100).round(1)
    results["away_win_prob"]  = ((1 - probs) * 100).round(1)
    results["predicted_winner"] = np.where(preds == 1, df_pred["home_team"], df_pred["away_team"])
    results["confidence"]     = np.maximum(results["home_win_prob"], results["away_win_prob"])
    results = results.sort_values("confidence", ascending=False)

    print("\n" + "=" * 72)
    print(f"  NRL TIPPING PREDICTIONS — Round {df_pred['round'].iloc[0]}")
    print("=" * 72)
    header = f"  {'HOME':<28} {'AWAY':<28} {'TIP':<28} {'CONF':>5}"
    print(header)
    print("  " + "─" * 68)
    for _, row in results.iterrows():
        home_str = f"{row['home_team']} ({row['home_win_prob']}%)"
        away_str = f"{row['away_team']} ({row['away_win_prob']}%)"
        conf_str = f"{row['confidence']:.1f}%"
        tip      = row["predicted_winner"]
        marker   = "◀" if tip == row["home_team"] else "  ▶"
        print(f"  {home_str:<28} {away_str:<28} {tip:<28} {conf_str:>6}")
    print("=" * 72)

    out_path = predict_path.replace(".csv", "_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Predictions saved → {out_path}\n")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NRL Tipping Model")
    parser.add_argument("--train",   action="store_true", help="Train model on nrl_source_data.csv")
    parser.add_argument("--predict", metavar="FILE",      help="Predict winners for an upcoming round CSV")
    args = parser.parse_args()

    model, feature_cols = None, None

    if args.train:
        print("\n[TRAIN]")
        df = load_data(DATA_PATH)
        print(f"  Loaded {len(df)} games from {DATA_PATH}")
        model, feature_cols = train(df)

    if args.predict:
        print("\n[PREDICT]")
        predict(args.predict, model, feature_cols)

    if not args.train and not args.predict:
        parser.print_help()


if __name__ == "__main__":
    main()
