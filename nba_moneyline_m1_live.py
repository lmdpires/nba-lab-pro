#!/usr/bin/env python3
"""NBA LAB PRO — M1 v1.1 live-state helper for Moneyline production integration.

The classifier is fitted once on the validated 2021-2025 regular-season history.
Current-season completed results update team state only; they never refit the model.
All state transitions use America/New_York calendar days and same-day snapshots.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nba_moneyline_m1_v1_1 as m1

ET = "America/New_York"

@dataclass(frozen=True)
class Game:
    game_id: str
    date: pd.Timestamp
    home_team: str
    away_team: str
    home_score: float | None = None
    away_score: float | None = None
    status: str = "scheduled"

class M1Live:
    def __init__(self, historical_csv: str | Path):
        self.historical = m1.load_games(Path(historical_csv))
        self.historical_features = m1.make_pregame_features(self.historical)
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs")),
        ])
        self.model.fit(self.historical_features[m1.FEATURES], self.historical_features["home_win"])
        self._team_ids = {}
        self._id_to_team = {}
        for row in self.historical.itertuples(index=False):
            for team, tid in ((row.home_team, getattr(row, "home_id", None)), (row.away_team, getattr(row, "away_id", None))):
                if tid is not None and pd.notna(tid):
                    tid = str(tid)
                    if team in self._team_ids and self._team_ids[team] != tid:
                        raise ValueError(f"ID inconsistente para {team}")
                    if tid in self._id_to_team and self._id_to_team[tid] != team:
                        raise ValueError(f"ID {tid} associado a duas equipas")
                    self._team_ids[team] = tid
                    self._id_to_team[tid] = team
        self.reset_state()
        self._replay(self.historical)

    def reset_state(self):
        self.elo = defaultdict(lambda: m1.ELO_BASE)
        self.history = defaultdict(lambda: deque(maxlen=10))
        self.last_played = {}
        self.applied_game_ids = set()

    @staticmethod
    def _day(value):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            raise ValueError("A data do jogo tem de incluir timezone")
        return ts.tz_convert(ET).date()

    def _features(self, home: str, away: str, day):
        if home not in self._team_ids or away not in self._team_ids:
            missing = [x for x in (home, away) if x not in self._team_ids]
            raise ValueError(f"Equipa(s) sem correspondência explícita no histórico: {missing}")
        def stats(team, window):
            hist = list(self.history[team])[-window:]
            return {k: m1.safe_mean([x[k] for x in hist]) for k in ("win", "margin", "pf", "pa")}
        h5, a5, h10, a10 = stats(home, 5), stats(away, 5), stats(home, 10), stats(away, 10)
        hr = min((day-self.last_played[home]).days, 14) if home in self.last_played else np.nan
        ar = min((day-self.last_played[away]).days, 14) if away in self.last_played else np.nan
        row = {"elo_diff": self.elo[home]-self.elo[away], "home_court": 1.0,
               "rest_days_diff": hr-ar if pd.notna(hr) and pd.notna(ar) else np.nan}
        for w, hs, ats in ((5,h5,a5),(10,h10,a10)):
            for source, key in (("win_rate","win"),("margin","margin"),("points_for","pf"),("points_against","pa")):
                row[f"{source}_{w}_diff"] = hs[key]-ats[key]
        return row

    def _apply_result(self, home, away, day, hs, aws):
        if hs is None or aws is None or not np.isfinite(hs) or not np.isfinite(aws) or hs == aws:
            raise ValueError("Resultado inválido: só jogos finalizados sem empate atualizam o estado")
        win = float(hs > aws)
        delta = m1.ELO_K * (win-m1.elo_home_probability(self.elo[home], self.elo[away]))
        self.elo[home] += delta; self.elo[away] -= delta
        margin = float(hs-aws)
        self.history[home].append({"win":win,"margin":margin,"pf":float(hs),"pa":float(aws)})
        self.history[away].append({"win":1-win,"margin":-margin,"pf":float(aws),"pa":float(hs)})
        self.last_played[home] = day; self.last_played[away] = day

    def _replay(self, games: pd.DataFrame):
        for day, day_games in games.groupby("game_day", sort=True):
            # The historical feature generation freezes the full day's state first.
            for r in day_games.sort_values(["date","game_id"]).itertuples(index=False):
                self._apply_result(r.home_team,r.away_team,day,float(r.home_score),float(r.away_score))

    def apply_completed_games(self, games: Iterable[Game]):
        """Apply current-season finals in ET-day batches; scheduled/live games ignored."""
        parsed=[]
        for g in games:
            status=str(g.status).strip().lower()
            if status not in {"finished","final","completed","3"}:
                continue
            if str(g.game_id) in self.applied_game_ids:
                continue
            day=self._day(g.date)
            # Explicit mapping only; no fuzzy team-name matching.
            if g.home_team not in self._team_ids or g.away_team not in self._team_ids:
                raise ValueError(f"Equipa desconhecida: {g.home_team} / {g.away_team}")
            parsed.append((day,pd.Timestamp(g.date),str(g.game_id),g))
        parsed.sort(key=lambda x:(x[0],x[1],x[2]))
        # Validate the complete incoming batch before mutating any team state.
        seen_ids = set()
        for day, _, gid, g in parsed:
            if gid in seen_ids:
                raise ValueError(f"Jogo duplicado na resposta API: {gid}")
            seen_ids.add(gid)
            if g.home_score is None or g.away_score is None or not np.isfinite(g.home_score) or not np.isfinite(g.away_score) or g.home_score==g.away_score:
                raise ValueError(f"Resultado inválido no jogo {g.game_id}")
        for day in sorted({x[0] for x in parsed}):
            batch=[x[3] for x in parsed if x[0]==day]
            for g in sorted(batch,key=lambda x:(pd.Timestamp(x.date),str(x.game_id))):
                self._apply_result(g.home_team,g.away_team,day,g.home_score,g.away_score)
                self.applied_game_ids.add(str(g.game_id))

    def predict_day(self, games: Iterable[Game], apply_finals_after: bool = False):
        """Predict every fixture on one ET date from one frozen state snapshot.

        If apply_finals_after=True, completed results from that date are applied
        only after every fixture has received its pre-day prediction.
        """
        items=list(games)
        if not items:
            return []
        days={self._day(g.date) for g in items}
        if len(days)!=1:
            raise ValueError("predict_day aceita jogos de uma única data ET")
        day=next(iter(days))
        results=[]
        for g in items:
            results.append({"game_id":g.game_id,"date":g.date,
                            **self.predict(g.date,g.home_team,g.away_team)})
        if apply_finals_after:
            self.apply_completed_games(items)
        return results

    def predict(self, game_date, home_team: str, away_team: str):
        day=self._day(game_date)
        row=self._features(home_team,away_team,day)
        X=pd.DataFrame([[row[f] for f in m1.FEATURES]],columns=m1.FEATURES)
        p=float(self.model.predict_proba(X)[0,1])
        if not 0.0 <= p <= 1.0:
            raise RuntimeError("Probabilidade fora do intervalo [0,1]")
        return {"p_home_m1":p,"p_away_m1":1-p,"features":row}


def _api_team_name(team_obj, engine: M1Live):
    """Resolve API-Sports team by historical provider ID; name fallback is exact only."""
    if not isinstance(team_obj, dict):
        raise ValueError("Objeto de equipa API-Sports inválido")
    team_id = team_obj.get("id")
    if team_id is not None:
        mapped = engine._id_to_team.get(str(team_id))
        if mapped:
            return mapped
    name = str(team_obj.get("name") or "").strip()
    if name in engine._team_ids:
        return name
    raise ValueError(f"Equipa API-Sports sem correspondência explícita: id={team_id!r}, nome={name!r}")


def parse_api_sports_game(item, engine: M1Live):
    """Parse one API-Sports v2 Basketball game object into a strict M1 Game."""
    if not isinstance(item, dict):
        raise ValueError("Jogo API-Sports inválido")
    teams = item.get("teams") or {}
    home_obj = teams.get("home") or {}
    away_obj = teams.get("visitors") or teams.get("away") or {}
    date_obj = item.get("date") or {}
    kickoff = date_obj.get("start") or date_obj.get("timestamp")
    if not kickoff:
        raise ValueError(f"Jogo API-Sports {item.get('id')} sem data de início")
    status_obj = item.get("status") or {}
    short = str(status_obj.get("short") or "").strip().upper()
    long_status = str(status_obj.get("long") or "").strip().lower()
    if short in {"FT", "AOT", "FTOT", "FINAL", "FINISHED", "ENDED", "3"} or any(w in long_status for w in ("finished", "final", "game over", "ended", "terminé", "terminado")):
        status = "final"
    elif short in {"Q1", "Q2", "Q3", "Q4", "OT", "HT", "LIVE", "IN", "2"} or any(w in long_status for w in ("in play", "live", "quarter", "halftime", "overtime", "en cours")):
        status = "live"
    else:
        status = "scheduled"
    scores = item.get("scores") or {}
    def score(value):
        if isinstance(value, dict):
            value = value.get("points", value.get("total", value.get("score")))
        try:
            return None if value is None or value == "" else float(value)
        except (TypeError, ValueError):
            return None
    return Game(game_id=str(item.get("id") or ""), date=pd.Timestamp(kickoff),
                home_team=_api_team_name(home_obj, engine),
                away_team=_api_team_name(away_obj, engine),
                home_score=score(scores.get("home")),
                away_score=score(scores.get("visitors") or scores.get("away")), status=status)


def parse_api_sports_payload(payload, engine: M1Live):
    """Parse API-Sports response envelope; reject API errors and malformed payloads."""
    if not isinstance(payload, dict):
        raise ValueError("Resposta API-Sports inválida")
    if payload.get("errors"):
        raise RuntimeError(f"API-Sports devolveu erros: {payload['errors']}")
    records = payload.get("response")
    if not isinstance(records, list):
        raise ValueError("Resposta API-Sports sem lista 'response'")
    return [parse_api_sports_game(item, engine) for item in records]


def api_season_query(season_start_year: int):
    """Query the full NBA season so M1 can replay every completed result to date.

    API-Sports /games supports season + league without date. The response must
    still be filtered locally to regular-season games and ET dates <= target day.
    """
    return {"season": str(int(season_start_year)), "league": "standard"}


def api_date_queries(selected_date):
    """Build app-compatible date queries: previous day plus selected day."""
    day = pd.Timestamp(selected_date).date()
    previous = day - pd.Timedelta(days=1)
    def season_for(d):
        return str(d.year if d.month >= 10 else d.year - 1)
    return [{"date": previous.isoformat(), "season": season_for(previous), "league": "standard"},
            {"date": day.isoformat(), "season": season_for(day), "league": "standard"}]


def load_api_games(records):
    """Strict adapter for already normalized records (legacy helper)."""
    games=[]
    for r in records:
        games.append(Game(game_id=str(r["game_id"]),date=pd.Timestamp(r["date"]),
                          home_team=str(r["home_team"]),away_team=str(r["away_team"]),
                          home_score=None if pd.isna(r.get("home_score")) else float(r["home_score"]),
                          away_score=None if pd.isna(r.get("away_score")) else float(r["away_score"]),
                          status=str(r.get("status","scheduled"))))
    return games
