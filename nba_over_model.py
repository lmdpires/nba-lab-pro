"""
NBA LAB PRO — modelo Over baseado no histórico.
Treina Ridge com features cronológicas e calibra probabilidades através de resíduos
de uma época de validação posterior ao treino. Não usa odds históricas.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATA_PATH = Path(__file__).with_name("nba_games_history.csv")
FEATURES = [
    "home_pf3", "home_pa3", "away_pf3", "away_pa3",
    "home_total3", "away_total3",
    "home_pf5", "home_pa5", "away_pf5", "away_pa5",
    "home_total5", "away_total5",
    "home_pf_std5", "away_pf_std5", "home_pa_std5", "away_pa_std5",
    "home_total_std5", "away_total_std5",
    "home_win5", "away_win5",
    "home_pf10", "home_pa10", "away_pf10", "away_pa10",
    "home_total10", "away_total10", "home_win10", "away_win10",
    "home_games_prior", "away_games_prior", "rest_home_days", "rest_away_days",
    "form_diff5", "form_diff10", "points_for_diff5", "defence_diff5",
]

def _team_stats(hist, n):
    recent = hist[-n:]
    if not recent:
        return {"pf": 110., "pa": 110., "win": .5, "total": 220.,
                "pf_std": 12., "pa_std": 12., "total_std": 20., "n": 0}
    totals = [x["pf"] + x["pa"] for x in recent]
    return {
        "pf": float(np.mean([x["pf"] for x in recent])),
        "pa": float(np.mean([x["pa"] for x in recent])),
        "win": float(np.mean([x["win"] for x in recent])),
        "total": float(np.mean(totals)),
        "pf_std": float(np.std([x["pf"] for x in recent], ddof=1)) if len(recent) > 1 else 12.,
        "pa_std": float(np.std([x["pa"] for x in recent], ddof=1)) if len(recent) > 1 else 12.,
        "total_std": float(np.std(totals, ddof=1)) if len(recent) > 1 else 20.,
        "n": len(recent),
    }

def _build_features(games):
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df[(df["home_score"] >= 50) & (df["away_score"] >= 50) &
            (df["home_score"] <= 200) & (df["away_score"] <= 200)]
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    history, rows = {}, []
    for date, batch in df.groupby("date", sort=True):
        pending = []
        for _, g in batch.iterrows():
            hid, aid = str(g["home_id"]), str(g["away_id"])
            hh, ah = history.get(hid, []), history.get(aid, [])
            h3, a3 = _team_stats(hh, 3), _team_stats(ah, 3)
            h5, a5 = _team_stats(hh, 5), _team_stats(ah, 5)
            h10, a10 = _team_stats(hh, 10), _team_stats(ah, 10)
            rest_h = min(10., (date-hh[-1]["date"]).total_seconds()/86400) if hh else 3.
            rest_a = min(10., (date-ah[-1]["date"]).total_seconds()/86400) if ah else 3.
            home_score, away_score = float(g["home_score"]), float(g["away_score"])
            home_win = int(home_score > away_score)
            rows.append({
                "game_id": str(g["game_id"]), "date": date, "season": str(g["season"]),
                "home_team": str(g["home_team"]), "away_team": str(g["away_team"]),
                "total_points": home_score + away_score,
                "home_pf3":h3["pf"],"home_pa3":h3["pa"],"away_pf3":a3["pf"],"away_pa3":a3["pa"],
                "home_total3":h3["total"],"away_total3":a3["total"],
                "home_pf5":h5["pf"],"home_pa5":h5["pa"],"away_pf5":a5["pf"],"away_pa5":a5["pa"],
                "home_total5":h5["total"],"away_total5":a5["total"],
                "home_pf_std5":h5["pf_std"],"away_pf_std5":a5["pf_std"],
                "home_pa_std5":h5["pa_std"],"away_pa_std5":a5["pa_std"],
                "home_total_std5":h5["total_std"],"away_total_std5":a5["total_std"],
                "home_win5":h5["win"],"away_win5":a5["win"],
                "home_pf10":h10["pf"],"home_pa10":h10["pa"],"away_pf10":a10["pf"],"away_pa10":a10["pa"],
                "home_total10":h10["total"],"away_total10":a10["total"],
                "home_win10":h10["win"],"away_win10":a10["win"],
                "home_games_prior":h10["n"],"away_games_prior":a10["n"],
                "rest_home_days":rest_h,"rest_away_days":rest_a,
                "form_diff5":(h5["pf"]-h5["pa"])-(a5["pf"]-a5["pa"]),
                "form_diff10":(h10["pf"]-h10["pa"])-(a10["pf"]-a10["pa"]),
                "points_for_diff5":h5["pf"]-a5["pf"],"defence_diff5":a5["pa"]-h5["pa"],
            })
            pending.append((hid, aid,
                {"date":date,"pf":home_score,"pa":away_score,"win":home_win},
                {"date":date,"pf":away_score,"pa":home_score,"win":int(away_score>home_score)}))
        for hid, aid, hu, au in pending:
            history.setdefault(hid, []).append(hu)
            history.setdefault(aid, []).append(au)
    return pd.DataFrame(rows)

def _normalize_team(value):
    value = str(value or "").strip().lower()
    aliases = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
    }
    return aliases.get(value, value)

@st.cache_resource(show_spinner=False)
def _load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {DATA_PATH.name} ao lado da app.py.")
    games = pd.read_csv(DATA_PATH)
    required = {"game_id","season","date","home_id","home_team","away_id","away_team","home_score","away_score"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError("Colunas em falta no histórico: " + ", ".join(sorted(missing)))
    games["season"] = games["season"].astype(str)
    feat = _build_features(games)
    feat = feat.dropna(subset=["total_points"]).sort_values("date").reset_index(drop=True)
    if len(feat) < 500:
        raise ValueError(f"Histórico insuficiente para treinar o modelo ({len(feat)} jogos).")
    seasons = sorted(feat["season"].dropna().unique())
    if len(seasons) >= 2:
        calibration_season = seasons[-1]
        train_cal = feat[feat["season"] < calibration_season]
        valid_cal = feat[feat["season"] == calibration_season]
        if len(train_cal) >= 300 and len(valid_cal) >= 100:
            cal_model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
            cal_model.fit(train_cal[FEATURES].replace([np.inf,-np.inf],np.nan), train_cal["total_points"])
            residuals = valid_cal["total_points"].to_numpy(dtype=float) - cal_model.predict(
                valid_cal[FEATURES].replace([np.inf,-np.inf],np.nan))
        else:
            residuals = np.array([])
    else:
        residuals = np.array([])
    if len(residuals) < 100:
        # Fallback is still based on chronological out-of-sample-like residuals where possible.
        split = int(len(feat) * .8)
        temp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
        temp.fit(feat.iloc[:split][FEATURES].replace([np.inf,-np.inf],np.nan), feat.iloc[:split]["total_points"])
        residuals = feat.iloc[split:]["total_points"].to_numpy(dtype=float) - temp.predict(
            feat.iloc[split:][FEATURES].replace([np.inf,-np.inf],np.nan))
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    model.fit(feat[FEATURES].replace([np.inf,-np.inf],np.nan), feat["total_points"])
    # Histories by API team name and team ID, for features at the requested kickoff time.
    history = {}
    raw = games.copy()
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date","home_score","away_score"]).sort_values("date")
    name_to_id = {}
    for _, g in raw.iterrows():
        hid, aid = str(g["home_id"]), str(g["away_id"])
        name_to_id[_normalize_team(g["home_team"])] = hid
        name_to_id[_normalize_team(g["away_team"])] = aid
        dt = g["date"]
        hs, aws = float(g["home_score"]), float(g["away_score"])
        history.setdefault(hid, []).append({"date":dt,"pf":hs,"pa":aws,"win":int(hs>aws)})
        history.setdefault(aid, []).append({"date":dt,"pf":aws,"pa":hs,"win":int(aws>hs)})
    return model, residuals, history, name_to_id, feat["date"].max()

def predict_total_and_over_prob(home_team, away_team, kickoff=None, line=220.5):
    """Return (predicted total, P(total > line), note)."""
    try:
        model, residuals, history, name_to_id, last_history_date = _load_model()
        if kickoff is None:
            kickoff = pd.Timestamp.now(tz="UTC")
        else:
            kickoff = pd.to_datetime(kickoff, utc=True, errors="coerce")
            if pd.isna(kickoff):
                kickoff = pd.Timestamp.now(tz="UTC")
        hid = name_to_id.get(_normalize_team(home_team))
        aid = name_to_id.get(_normalize_team(away_team))
        if hid is None or aid is None:
            return None, None, f"Não encontrei a equipa no histórico: {home_team if hid is None else away_team}."
        hh = [x for x in history.get(hid, []) if x["date"] < kickoff]
        ah = [x for x in history.get(aid, []) if x["date"] < kickoff]
        h3,a3 = _team_stats(hh,3),_team_stats(ah,3)
        h5,a5 = _team_stats(hh,5),_team_stats(ah,5)
        h10,a10 = _team_stats(hh,10),_team_stats(ah,10)
        rest_h = min(10.,(kickoff-hh[-1]["date"]).total_seconds()/86400) if hh else 3.
        rest_a = min(10.,(kickoff-ah[-1]["date"]).total_seconds()/86400) if ah else 3.
        row = {
            "home_pf3":h3["pf"],"home_pa3":h3["pa"],"away_pf3":a3["pf"],"away_pa3":a3["pa"],
            "home_total3":h3["total"],"away_total3":a3["total"],
            "home_pf5":h5["pf"],"home_pa5":h5["pa"],"away_pf5":a5["pf"],"away_pa5":a5["pa"],
            "home_total5":h5["total"],"away_total5":a5["total"],
            "home_pf_std5":h5["pf_std"],"away_pf_std5":a5["pf_std"],
            "home_pa_std5":h5["pa_std"],"away_pa_std5":a5["pa_std"],
            "home_total_std5":h5["total_std"],"away_total_std5":a5["total_std"],
            "home_win5":h5["win"],"away_win5":a5["win"],
            "home_pf10":h10["pf"],"home_pa10":h10["pa"],"away_pf10":a10["pf"],"away_pa10":a10["pa"],
            "home_total10":h10["total"],"away_total10":a10["total"],
            "home_win10":h10["win"],"away_win10":a10["win"],
            "home_games_prior":h10["n"],"away_games_prior":a10["n"],
            "rest_home_days":rest_h,"rest_away_days":rest_a,
            "form_diff5":(h5["pf"]-h5["pa"])-(a5["pf"]-a5["pa"]),
            "form_diff10":(h10["pf"]-h10["pa"])-(a10["pf"]-a10["pa"]),
            "points_for_diff5":h5["pf"]-a5["pf"],"defence_diff5":a5["pa"]-h5["pa"],
        }
        X = pd.DataFrame([{c:row.get(c,np.nan) for c in FEATURES}])
        predicted = float(model.predict(X)[0])
        if len(residuals):
            p_over = float(np.mean(predicted + residuals > float(line)))
            p_over = min(.99, max(.01, p_over))
        else:
            p_over = None
        return round(predicted,1), (round(p_over,4) if p_over is not None else None), None
    except Exception as exc:
        return None, None, f"Modelo Over indisponível: {exc}"
