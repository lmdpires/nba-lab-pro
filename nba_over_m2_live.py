"""NBA LAB PRO — frozen M2 HGB live inference engine.

Uses the frozen 102-feature HistGradientBoostingRegressor trained on 6,150
regular-season games through 2025, with a separate, append-only results file
for 2026-27 form. Does not retrain the model.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "nba_over_m2_hgb_candidate.joblib"
HISTORY_PATH = BASE / "nba_over_m2_history.csv"
METRICS_PATH = BASE / "nba_over_m2_metadata.json"

ALIASES = {
    "la clippers": "los angeles clippers", "la lakers": "los angeles lakers",
    "ny knicks": "new york knicks", "new york knicks": "new york knicks",
}

def _team_key(value: str) -> str:
    s = str(value or "").strip().lower().replace(".", "")
    return ALIASES.get(s, s)

def _stats(seq, n):
    sample = list(seq)[-n:]
    if not sample:
        return {k: np.nan for k in ("pf", "pa", "total", "pf_std", "pa_std", "total_std", "win")}
    pf = np.asarray([x["pf"] for x in sample], dtype=float)
    pa = np.asarray([x["pa"] for x in sample], dtype=float)
    total = pf + pa
    return {
        "pf": float(pf.mean()), "pa": float(pa.mean()), "total": float(total.mean()),
        "pf_std": float(pf.std(ddof=1)) if len(sample) > 1 else np.nan,
        "pa_std": float(pa.std(ddof=1)) if len(sample) > 1 else np.nan,
        "total_std": float(total.std(ddof=1)) if len(sample) > 1 else np.nan,
        "win": float(np.mean([x["win"] for x in sample])),
    }

def _prepare_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Não encontrei {path.name} ao lado da aplicação.")
    df = pd.read_csv(path)
    required = {"game_id", "season", "date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Histórico M2 sem colunas: " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "season", "home_team", "away_team", "home_score", "away_score"]).copy()
    df["season"] = df["season"].astype(int)
    df = df[(df.home_score.between(50, 200)) & (df.away_score.between(50, 200)) & (df.home_score != df.away_score)].copy()
    df["game_day"] = df["date"].dt.tz_convert("America/New_York").dt.date
    # Frozen training history: the exact five validated regular seasons.
    windows = {
        2021: ("2021-10-19", "2022-04-10"),
        2022: ("2022-10-18", "2023-04-09"),
        2023: ("2023-10-24", "2024-04-14"),
        2024: ("2024-10-22", "2025-04-13"),
        2025: ("2025-10-21", "2026-04-12"),
    }
    cup_finals = {(2023, "2023-12-09"), (2024, "2024-12-17"), (2025, "2025-12-16")}
    daystr = df.game_day.astype(str)
    valid = pd.Series(False, index=df.index)
    for season, (start, end) in windows.items():
        valid |= (df.season.eq(season) & daystr.between(start, end))
    cup = pd.Series([(int(s), str(d)) in cup_finals for s, d in zip(df.season, daystr)], index=df.index)
    df = df[valid & ~cup].copy()
    counts = df.groupby("season").size().to_dict()
    expected = {s: 1230 for s in windows}
    actual = {int(k): int(v) for k, v in counts.items()}
    if actual != expected:
        raise ValueError(f"Histórico M2 congelado não passou a validação: {actual}; esperado {expected}.")
    # Add only completed current-season games. No preseason/playoffs are accepted.
    current = pd.read_csv(path)
    current["date"] = pd.to_datetime(current["date"], utc=True, errors="coerce")
    current["season"] = pd.to_numeric(current["season"], errors="coerce")
    current["home_score"] = pd.to_numeric(current["home_score"], errors="coerce")
    current["away_score"] = pd.to_numeric(current["away_score"], errors="coerce")
    current = current[(current.season == 2026) & current.date.notna() & current.home_score.notna() & current.away_score.notna()].copy()
    current["game_day"] = current.date.dt.tz_convert("America/New_York").dt.date
    current = current[(current.game_day.astype(str) >= "2026-10-20") & (current.game_day.astype(str) <= "2027-04-11")]
    current = current[(current.home_score.between(50, 200)) & (current.away_score.between(50, 200)) & (current.home_score != current.away_score)]
    cols = ["game_id", "season", "date", "game_day", "home_team", "away_team", "home_score", "away_score"]
    hist = pd.concat([df[cols], current[cols]], ignore_index=True)
    hist["game_id"] = hist.game_id.astype(str)
    if hist.game_id.duplicated().any():
        hist = hist.drop_duplicates("game_id", keep="last")
    return hist.sort_values(["game_day", "date", "game_id"]).reset_index(drop=True)

@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {MODEL_PATH.name}.")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {METRICS_PATH.name}.")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if int(model.n_features_in_) != 102 or len(meta["feature_names"]) != 102:
        raise ValueError("O artefacto M2 não corresponde ao schema congelado de 102 features.")
    return model, meta

@lru_cache(maxsize=8)
def _history_state(target_day):
    hist = _prepare_history(HISTORY_PATH)
    # Match the research protocol: all games on the fixture ET date are held
    # out of the feature state, even if another same-day game has finished.
    hist = hist[hist.game_day < target_day].copy()
    teams = defaultdict(list)
    last = {}
    for day, batch in hist.groupby("game_day", sort=True):
        pending = []
        for g in batch.itertuples(index=False):
            h, a = str(g.home_team), str(g.away_team)
            pending.append((h, a,
                {"season": int(g.season), "pf": float(g.home_score), "pa": float(g.away_score), "win": int(g.home_score > g.away_score)},
                {"season": int(g.season), "pf": float(g.away_score), "pa": float(g.home_score), "win": int(g.away_score > g.home_score)}))
        for h, a, hu, au in pending:
            teams[h].append(hu); teams[a].append(au); last[h] = day; last[a] = day
    return teams, last, hist

def _fixture_row(home: str, away: str, kickoff) -> dict:
    day = pd.Timestamp(kickoff)
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    day = day.tz_convert("America/New_York").date()
    teams, last, hist = _history_state(day)
    # Fixture aliases map to the labels used by the current 2026-27 results feed.
    current_aliases = {"la clippers": "los angeles clippers", "la lakers": "los angeles lakers", "ny knicks": "new york knicks"}
    h, a = _team_key(home), _team_key(away)
    h, a = current_aliases.get(h, h), current_aliases.get(a, a)
    # Resolve case-insensitively to exact labels present in the current history.
    labels = {str(name).lower(): str(name) for name in teams.keys()}
    h, a = labels.get(h, h), labels.get(a, a)
    # Preserve raw historical team labels: the frozen feature builder treated
    # legacy labels and 2026-27 labels as separate keys.
    hh, aa = teams[h], teams[a]
    r = {"home_team": home, "away_team": away, "season": 2026}
    for n in (3, 5, 10, 20):
        for side, seq in (("home", hh), ("away", aa)):
            st = _stats(seq, n)
            for k, v in st.items(): r[f"{side}_{k}_{n}"] = v
    for side, seq in (("home", hh), ("away", aa)):
        ss = [x for x in seq if x["season"] == 2026]
        for k, fn in (("pf", lambda x: x["pf"]), ("pa", lambda x: x["pa"]), ("total", lambda x: x["pf"] + x["pa"])):
            r[f"{side}_season_{k}"] = float(np.mean([fn(x) for x in ss])) if ss else np.nan
        r[f"{side}_games_prior"] = len(ss)
    r["home_rest_days"] = min((day - last[h]).days, 14) if h in last else np.nan
    r["away_rest_days"] = min((day - last[a]).days, 14) if a in last else np.nan
    r["rest_sum"] = r["home_rest_days"] + r["away_rest_days"] if pd.notna(r["home_rest_days"]) and pd.notna(r["away_rest_days"]) else np.nan
    r["rest_diff"] = r["home_rest_days"] - r["away_rest_days"] if pd.notna(r["home_rest_days"]) and pd.notna(r["away_rest_days"]) else np.nan
    for n in (5, 10, 20):
        hp, ap, hpa, apa = r[f"home_pf_{n}"], r[f"away_pf_{n}"], r[f"home_pa_{n}"], r[f"away_pa_{n}"]
        r[f"sum_pf{n}"] = hp + ap if pd.notna(hp) and pd.notna(ap) else np.nan
        r[f"sum_pa{n}"] = hpa + apa if pd.notna(hpa) and pd.notna(apa) else np.nan
        r[f"sum_total{n}"] = r[f"sum_pf{n}"] + r[f"sum_pa{n}"] if pd.notna(r[f"sum_pf{n}"]) and pd.notna(r[f"sum_pa{n}"]) else np.nan
        r[f"pace_form{n}"] = r[f"sum_total{n}"]
        r[f"defense_matchup{n}"] = r[f"sum_pa{n}"]; r[f"offense_matchup{n}"] = r[f"sum_pf{n}"]
    r.update({
      "home_pf3":r["home_pf_3"],"home_pa3":r["home_pa_3"],"away_pf3":r["away_pf_3"],"away_pa3":r["away_pa_3"],
      "home_total3":r["home_total_3"],"away_total3":r["away_total_3"],
      "home_pf5":r["home_pf_5"],"home_pa5":r["home_pa_5"],"away_pf5":r["away_pf_5"],"away_pa5":r["away_pa_5"],
      "home_total5":r["home_total_5"],"away_total5":r["away_total_5"],
      "home_pf_std5":r["home_pf_std_5"],"away_pf_std5":r["away_pf_std_5"],"home_pa_std5":r["home_pa_std_5"],"away_pa_std5":r["away_pa_std_5"],
      "home_total_std5":r["home_total_std_5"],"away_total_std5":r["away_total_std_5"],
      "home_win5":r["home_win_5"],"away_win5":r["away_win_5"],
      "home_pf10":r["home_pf_10"],"home_pa10":r["home_pa_10"],"away_pf10":r["away_pf_10"],"away_pa10":r["away_pa_10"],
      "home_total10":r["home_total_10"],"away_total10":r["away_total_10"],"home_win10":r["home_win_10"],"away_win10":r["away_win_10"],
      "rest_home_days":r["home_rest_days"],"rest_away_days":r["away_rest_days"],
      "form_diff5":(r["home_pf_5"]-r["home_pa_5"])-(r["away_pf_5"]-r["away_pa_5"]),
      "form_diff10":(r["home_pf_10"]-r["home_pa_10"])-(r["away_pf_10"]-r["away_pa_10"]),
      "points_for_diff5":r["home_pf_5"]-r["away_pf_5"],"defence_diff5":r["away_pa_5"]-r["home_pa_5"]})
    return r

def _add_derived(row):
    for n in (5, 10, 20):
        row[f"combined_scoring_{n}"] = row[f"home_pf_{n}"] + row[f"away_pf_{n}"]
        row[f"combined_conceded_{n}"] = row[f"home_pa_{n}"] + row[f"away_pa_{n}"]
        row[f"combined_total_form_{n}"] = row[f"home_total_{n}"] + row[f"away_total_{n}"]
        row[f"attack_defence_matchup_{n}"] = (row[f"home_pf_{n}"] + row[f"away_pa_{n}"] + row[f"away_pf_{n}"] + row[f"home_pa_{n}"]) / 2
    return row

def predict_total(home: str, away: str, kickoff=None):
    """Return frozen M2 HGB total prediction and metadata; no fallback model."""
    try:
        model, meta = _model()
        if kickoff is None:
            raise ValueError("Data/hora de início do jogo não disponível.")
        row = _add_derived(_fixture_row(home, away, kickoff))
        cols = meta["feature_names"]
        X = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}], columns=cols)
        if X.shape[1] != model.n_features_in_:
            raise ValueError("Número de features incompatível com o modelo congelado.")
        value = float(model.predict(X)[0])
        if not np.isfinite(value):
            raise ValueError("O M2 devolveu uma previsão não finita.")
        return value, None
    except Exception as exc:
        return None, f"M2 indisponível: {exc}"

def probability_over(total_pred: float, line: float):
    """Frozen baseline: Normal residual distribution, sigma estimated OOT in 2024-25."""
    _, meta = _model()
    sigma = float(meta["sigma_from_2025_oot"])
    p_over = float(norm.sf((float(line) - float(total_pred)) / sigma))
    p_over = min(0.999999, max(0.000001, p_over))
    return p_over, 1.0 - p_over

def predict_total_and_over_prob(home: str, away: str, kickoff=None, line: float = 220.5):
    total, error = predict_total(home, away, kickoff)
    if total is None:
        return None, None, error
    try:
        p_over, _ = probability_over(total, line)
        return round(total, 2), p_over, None
    except Exception as exc:
        return None, None, f"M2 probabilidade indisponível: {exc}"
