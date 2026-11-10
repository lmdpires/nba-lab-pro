#!/usr/bin/env python3
"""NBA LAB PRO — M1 v1.1 Moneyline — regular-season only.

Builds pre-game team features from historical results and evaluates a regularized
logistic-regression model with expanding, season-by-season walk-forward testing.

Input CSV columns required:
  game_id, season, date, home_team, away_team, home_score, away_score

Important safeguards:
- Features for every game on a calendar date are computed BEFORE any result
  from that date is applied (no same-day result leakage).
- Team rolling features use prior games only.
- Test seasons are never used to fit the model for that season.
- The 77-game disagreement audit and the existing V6 are not touched.

This is a reproducible research baseline, not a production betting system.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ELO_BASE = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0
ELO_SCALE = 400.0
ROLL_WINDOWS = (5, 10)
FEATURES = [
    "elo_diff", "home_court",
    "win_rate_5_diff", "margin_5_diff", "points_for_5_diff", "points_against_5_diff",
    "win_rate_10_diff", "margin_10_diff", "points_for_10_diff", "points_against_10_diff",
    "rest_days_diff",
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def elo_home_probability(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_elo + ELO_HOME_ADVANTAGE - away_elo) / ELO_SCALE))


def safe_mean(values, default=np.nan):
    return float(np.mean(values)) if len(values) else default


def load_games(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"game_id", "season", "date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "season", "home_team", "away_team"]).copy()
    # Only completed games with a valid final score are eligible for training/testing.
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["home_score"] != df["away_score"]].copy()  # NBA games have no draws.
    df["season"] = df["season"].astype(int)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["game_day"] = df["date"].dt.tz_convert("America/New_York").dt.date
    df = df.sort_values(["game_day", "date", "game_id"]).reset_index(drop=True)
    if df["game_id"].duplicated().any():
        raise ValueError("Foram encontrados game_id duplicados.")

    # Official NBA regular-season windows, keyed by the API season label.
    # Dates are Eastern Time calendar dates. Preseason, Play-In and playoffs
    # are excluded. NBA Cup championship games are also excluded because
    # they do not count in regular-season standings.
    regular_season_windows = {
        2021: ("2021-10-19", "2022-04-10"),
        2022: ("2022-10-18", "2023-04-09"),
        2023: ("2023-10-24", "2024-04-14"),
        2024: ("2024-10-22", "2025-04-13"),
        2025: ("2025-10-21", "2026-04-12"),
    }
    cup_finals = {
        (2023, "2023-12-09"),
        (2024, "2024-12-17"),
        (2025, "2025-12-16"),
    }
    if not set(df["season"].unique()).issubset(regular_season_windows):
        unknown = sorted(set(df["season"].unique()) - set(regular_season_windows))
        raise ValueError(f"Épocas sem calendário validado: {unknown}")

    df["game_day_str"] = df["game_day"].astype(str)
    in_regular_season = pd.Series(False, index=df.index)
    for season, (start, end) in regular_season_windows.items():
        mask = (
            (df["season"] == season)
            & (df["game_day_str"] >= start)
            & (df["game_day_str"] <= end)
        )
        in_regular_season |= mask
    is_cup_final = pd.Series(
        [(int(s), str(day)) in cup_finals for s, day in zip(df["season"], df["game_day_str"])],
        index=df.index,
    )
    df = df[in_regular_season & ~is_cup_final].copy()

    # The All-Star Game is not a regular-season fixture and uses temporary
    # team labels (e.g. Team LeBron / Team Durant).
    special_team = (
        df["home_team"].astype(str).str.startswith("Team ")
        | df["away_team"].astype(str).str.startswith("Team ")
    )
    df = df[~special_team].copy()
    df = df.drop(columns=["game_day_str"])

    # Integrity check: each of the five supplied regular seasons should have
    # exactly 1,230 NBA regular-season games (30 teams x 82 games / 2).
    counts = df.groupby("season").size()
    expected = {season: 1230 for season in regular_season_windows}
    actual = {int(k): int(v) for k, v in counts.items()}
    if actual != expected:
        raise ValueError(
            f"Contagem de jogos regular-season inesperada: {actual}; "
            f"esperado: {expected}. Rever calendário/dataset antes do treino."
        )
    return df


def make_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Generate features using state frozen at the start of each game day."""
    elo = defaultdict(lambda: ELO_BASE)
    history = defaultdict(lambda: deque(maxlen=10))
    last_played = {}
    rows = []

    for day, day_games in games.groupby("game_day", sort=True):
        # Snapshot all features before processing ANY result from this day.
        day_records = []
        for r in day_games.itertuples(index=False):
            h, a = r.home_team, r.away_team
            hh, ah = list(history[h]), list(history[a])

            def team_stats(hist, window):
                sample = hist[-window:]
                return {
                    f"win_rate_{window}": safe_mean([x["win"] for x in sample]),
                    f"margin_{window}": safe_mean([x["margin"] for x in sample]),
                    f"points_for_{window}": safe_mean([x["pf"] for x in sample]),
                    f"points_against_{window}": safe_mean([x["pa"] for x in sample]),
                }

            hs5, as5 = team_stats(hh, 5), team_stats(ah, 5)
            hs10, as10 = team_stats(hh, 10), team_stats(ah, 10)
            hrest = min((day - last_played[h]).days, 14) if h in last_played else np.nan
            arest = min((day - last_played[a]).days, 14) if a in last_played else np.nan
            rec = {
                "game_id": r.game_id, "season": r.season, "date": r.date,
                "game_day": day, "home_team": h, "away_team": a,
                "home_score": r.home_score, "away_score": r.away_score,
                "home_win": r.home_win,
                "elo_diff": elo[h] - elo[a], "home_court": 1.0,
                "rest_days_diff": hrest - arest if pd.notna(hrest) and pd.notna(arest) else np.nan,
            }
            for window, hs, ats in ((5, hs5, as5), (10, hs10, as10)):
                for stat in ("win_rate", "margin", "points_for", "points_against"):
                    rec[f"{stat}_{window}_diff"] = hs[f"{stat}_{window}"] - ats[f"{stat}_{window}"]
            rec["elo_home_pre"] = elo[h]
            rec["elo_away_pre"] = elo[a]
            day_records.append(rec)

        rows.extend(day_records)

        # Update Elo and rolling history only AFTER all same-day features were frozen.
        # If a team appears more than once on a date, apply its games in timestamp order.
        for r in day_games.sort_values(["date", "game_id"]).itertuples(index=False):
            h, a = r.home_team, r.away_team
            expected_home = elo_home_probability(elo[h], elo[a])
            actual_home = float(r.home_win)
            delta = ELO_K * (actual_home - expected_home)
            elo[h] += delta
            elo[a] -= delta
            margin = float(r.home_score - r.away_score)
            history[h].append({"win": actual_home, "margin": margin,
                               "pf": float(r.home_score), "pa": float(r.away_score)})
            history[a].append({"win": 1.0 - actual_home, "margin": -margin,
                               "pf": float(r.away_score), "pa": float(r.home_score)})
            last_played[h] = day
            last_played[a] = day

    return pd.DataFrame(rows).sort_values(["date", "game_id"]).reset_index(drop=True)


def evaluate_walk_forward(data: pd.DataFrame, min_train_seasons: int = 2):
    seasons = sorted(data["season"].unique())
    test_seasons = seasons[min_train_seasons:]
    if not test_seasons:
        raise ValueError(f"Não há épocas de teste. Épocas encontradas: {seasons}")

    predictions = []
    for season in test_seasons:
        train = data[data["season"] < season].copy()
        test = data[data["season"] == season].copy()
        if train.empty or test.empty:
            continue
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs")),
        ])
        model.fit(train[FEATURES], train["home_win"])
        p_m1 = model.predict_proba(test[FEATURES])[:, 1]
        p_elo = [elo_home_probability(h, a) for h, a in zip(test["elo_home_pre"], test["elo_away_pre"])]
        out = test[["game_id", "season", "date", "home_team", "away_team", "home_score", "away_score", "home_win"]].copy()
        out["p_home_m1"] = p_m1
        out["p_home_elo_k20"] = p_elo
        out["m1_pick_correct"] = ((p_m1 >= 0.5).astype(int) == out["home_win"]).astype(int)
        out["elo_pick_correct"] = ((np.array(p_elo) >= 0.5).astype(int) == out["home_win"]).astype(int)
        predictions.append(out)

    if not predictions:
        raise ValueError("Walk-forward não produziu previsões.")
    pred = pd.concat(predictions, ignore_index=True)
    y = pred["home_win"].to_numpy()
    metrics = {}
    for label, col in (("M1", "p_home_m1"), ("Elo_K20", "p_home_elo_k20")):
        p = np.clip(pred[col].to_numpy(), 1e-6, 1 - 1e-6)
        metrics[label] = {
            "games": int(len(pred)),
            "accuracy": float(accuracy_score(y, p >= 0.5)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
        }
    by_season = pred.groupby("season").apply(
        lambda g: pd.Series({
            "games": len(g),
            "m1_accuracy": accuracy_score(g["home_win"], g["p_home_m1"] >= 0.5),
            "elo_accuracy": accuracy_score(g["home_win"], g["p_home_elo_k20"] >= 0.5),
            "m1_brier": brier_score_loss(g["home_win"], g["p_home_m1"]),
            "elo_brier": brier_score_loss(g["home_win"], g["p_home_elo_k20"]),
            "m1_log_loss": log_loss(g["home_win"], np.clip(g["p_home_m1"], 1e-6, 1 - 1e-6), labels=[0, 1]),
            "elo_log_loss": log_loss(g["home_win"], np.clip(g["p_home_elo_k20"], 1e-6, 1 - 1e-6), labels=[0, 1]),
        }), include_groups=False
    ).reset_index()
    return pred, metrics, by_season


def main():
    parser = argparse.ArgumentParser(description="Treino e validação walk-forward do NBA Moneyline M1")
    parser.add_argument("--input", default="nba_games_history.csv", help="Caminho do CSV histórico")
    parser.add_argument("--output-dir", default="m1_output", help="Pasta para resultados")
    parser.add_argument("--min-train-seasons", type=int, default=2,
                        help="Número de épocas iniciais reservadas para treino antes do primeiro teste")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games = load_games(input_path)
    features = make_pregame_features(games)
    predictions, metrics, by_season = evaluate_walk_forward(features, args.min_train_seasons)

    features.to_csv(output_dir / "m1_pregame_features.csv", index=False)
    predictions.to_csv(output_dir / "m1_walkforward_predictions.csv", index=False)
    by_season.to_csv(output_dir / "m1_metrics_by_season.csv", index=False)
    with open(output_dir / "m1_metrics_overall.json", "w", encoding="utf-8") as f:
        json.dump({"input_games_with_final_score": int(len(games)),
                   "feature_rows": int(len(features)),
                   "test_seasons": sorted(int(x) for x in predictions["season"].unique()),
                   "metrics": metrics}, f, ensure_ascii=False, indent=2)

    print(f"Jogos válidos no histórico: {len(games):,}")
    print(f"Jogos avaliados fora da amostra: {len(predictions):,}")
    print("\nMétricas globais:")
    for name, values in metrics.items():
        print(f"{name:10s} | n={values['games']:5d} | Accuracy={values['accuracy']:.4%} "
              f"| Brier={values['brier']:.5f} | LogLoss={values['log_loss']:.5f}")
    print("\nMétricas por época:")
    print(by_season.to_string(index=False))
    print(f"\nFicheiros guardados em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
