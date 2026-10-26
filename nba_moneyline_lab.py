from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, confusion_matrix

st.set_page_config(page_title="NBA Moneyline Lab", page_icon="🏀", layout="wide")
st.title("🏀 NBA LAB PRO — Moneyline Lab")
st.caption("Laboratório separado para testar probabilidades de vitória com validação cronológica. Não altera a app principal nem as outras Labs.")

DATA_PATH = Path(__file__).with_name("nba_games_history.csv")
WINDOWS = (5, 10)
FEATURES = [
    "home_win_rate_5", "home_win_rate_10", "away_win_rate_5", "away_win_rate_10",
    "home_margin_5", "home_margin_10", "away_margin_5", "away_margin_10",
    "home_points_for_5", "home_points_for_10", "away_points_for_5", "away_points_for_10",
    "home_points_against_5", "home_points_against_10", "away_points_against_5", "away_points_against_10",
    "home_games", "away_games", "home_rest_days", "away_rest_days", "home_court"
]

def load_games(path):
    df = pd.read_csv(path)
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Faltam colunas no CSV: " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    if "status" in df.columns:
        status = df["status"].astype(str).str.strip().str.upper()
        final_words = {"3", "FT", "AOT", "FTOT", "FINAL", "FINISHED", "ENDED"}
        is_final = status.isin(final_words) | status.str.contains("FINAL|FINISHED|ENDED", na=False)
        # Some historical files use numeric status 3; when score columns are present, accept these final rows.
        df = df[is_final].copy()
    df = df[(df["home_score"] >= 50) & (df["away_score"] >= 50)]
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df.sort_values(["date", "game_id"] if "game_id" in df.columns else ["date"]).reset_index(drop=True)

def stats(history, today):
    out = {}
    for n in WINDOWS:
        recent = history[-n:]
        if not recent:
            out[f"win_rate_{n}"] = np.nan
            out[f"margin_{n}"] = np.nan
            out[f"points_for_{n}"] = np.nan
            out[f"points_against_{n}"] = np.nan
        else:
            out[f"win_rate_{n}"] = float(np.mean([x["win"] for x in recent]))
            out[f"margin_{n}"] = float(np.mean([x["pf"] - x["pa"] for x in recent]))
            out[f"points_for_{n}"] = float(np.mean([x["pf"] for x in recent]))
            out[f"points_against_{n}"] = float(np.mean([x["pa"] for x in recent]))
    out["games"] = len(history)
    out["rest_days"] = min(10, max(0, (today - history[-1]["date"]).total_seconds() / 86400)) if history else np.nan
    return out

def make_dataset(games):
    histories = {}
    rows = []
    # Process each calendar date as a batch so no game can leak another result from the same date.
    for date, batch in games.groupby(games["date"].dt.floor("D"), sort=True):
        batch_rows = []
        for _, g in batch.iterrows():
            home, away = str(g["home_team"]), str(g["away_team"])
            hs = stats(histories.get(home, []), g["date"])
            aw = stats(histories.get(away, []), g["date"])
            row = {"date": g["date"], "home_team": home, "away_team": away, "home_win": int(g["home_win"]),
                   "home_court": 1.0}
            for key, value in hs.items(): row["home_" + key] = value
            for key, value in aw.items(): row["away_" + key] = value
            # Rename generated fields to the feature names used by the model.
            for n in WINDOWS:
                row[f"home_win_rate_{n}"] = row.pop(f"home_win_rate_{n}", row.get(f"home_win_rate_{n}"))
                row[f"away_win_rate_{n}"] = row.pop(f"away_win_rate_{n}", row.get(f"away_win_rate_{n}"))
            rows.append(row)
            batch_rows.append((g, home, away))
        for g, home, away in batch_rows:
            histories.setdefault(home, []).append({"date": g["date"], "win": int(g["home_win"]), "pf": float(g["home_score"]), "pa": float(g["away_score"])})
            histories.setdefault(away, []).append({"date": g["date"], "win": int(g["away_score"] > g["home_score"]), "pf": float(g["away_score"]), "pa": float(g["home_score"])})
    result = pd.DataFrame(rows)
    # Explicit mapping from the generated history stats to model feature names.
    for side in ("home", "away"):
        for n in WINDOWS:
            for metric in ("win_rate", "margin", "points_for", "points_against"):
                src = f"{side}_{metric}_{n}"
                if src not in result.columns:
                    result[src] = np.nan
        result[f"{side}_games"] = result.get(f"{side}_games", np.nan)
        result[f"{side}_rest_days"] = result.get(f"{side}_rest_days", np.nan)
    # Stats keys are produced as win_rate_5, margin_5 etc.; above names match those keys.
    result = result.dropna(subset=["date", "home_win"]).sort_values("date").reset_index(drop=True)
    return result

@st.cache_data(show_spinner="A preparar o histórico e as variáveis pré-jogo…")
def cached_data(path, modified):
    games = load_games(path)
    data = make_dataset(games)
    return games, data

if not DATA_PATH.exists():
    st.error("Não encontrei nba_games_history.csv na mesma pasta da aplicação. Coloca o CSV na raiz do repositório GitHub.")
    st.stop()

try:
    games, data = cached_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Não foi possível preparar o histórico: {exc}")
    st.stop()

# Require enough prior history for both teams; this prevents cold-start rows dominating the test.
usable = data[(data["home_games"] >= 3) & (data["away_games"] >= 3)].copy()
if len(usable) < 300:
    st.error(f"Histórico insuficiente após preparar as variáveis: {len(usable)} jogos utilizáveis.")
    st.stop()

split = int(len(usable) * 0.80)
train, test = usable.iloc[:split].copy(), usable.iloc[split:].copy()
if train["home_win"].nunique() < 2 or test["home_win"].nunique() < 2:
    st.error("Não há variedade suficiente de resultados para treinar/avaliar o modelo.")
    st.stop()
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
model.fit(train[FEATURES], train["home_win"])
prob = model.predict_proba(test[FEATURES])[:, 1]
pred = (prob >= 0.5).astype(int)
base_prob = np.full(len(test), train["home_win"].mean())

st.subheader("Validação cronológica fora da amostra")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jogos de teste", f"{len(test):,}")
c2.metric("Acerto", f"{accuracy_score(test['home_win'], pred)*100:.1f}%")
c3.metric("Brier Score", f"{brier_score_loss(test['home_win'], prob):.4f}", help="Mais baixo é melhor.")
c4.metric("Log Loss", f"{log_loss(test['home_win'], prob, labels=[0,1]):.4f}", help="Mais baixo é melhor.")

st.markdown("**Comparação com uma referência simples**")
b1, b2, b3 = st.columns(3)
b1.metric("Brier do modelo", f"{brier_score_loss(test['home_win'], prob):.4f}")
b2.metric("Brier da referência", f"{brier_score_loss(test['home_win'], base_prob):.4f}")
b3.metric("Taxa de vitória da casa no teste", f"{test['home_win'].mean()*100:.1f}%")
if brier_score_loss(test['home_win'], prob) < brier_score_loss(test['home_win'], base_prob):
    st.success("O modelo melhora o Brier Score face à referência simples nesta janela de teste. Isto não demonstra rentabilidade.")
else:
    st.warning("O modelo não melhora o Brier Score face à referência simples nesta janela. Não o ligues ainda à app principal.")

st.caption(f"Histórico final utilizado: {len(games):,} jogos · Dataset com variáveis pré-jogo: {len(usable):,} · Treino: {train['date'].min().date()} a {train['date'].max().date()} · Teste: {test['date'].min().date()} a {test['date'].max().date()}.")

st.subheader("Últimos jogos do período de teste")
view = test[["date", "home_team", "away_team", "home_win"]].copy()
view["P vitória casa"] = prob
view["Previsão"] = np.where(prob >= 0.5, "Casa", "Visitante")
view["Resultado"] = np.where(view["home_win"] == 1, "Casa venceu", "Visitante venceu")
view["Acertou"] = np.where((prob >= 0.5) == (view["home_win"].to_numpy() == 1), "Sim", "Não")
view["date"] = view["date"].dt.strftime("%d/%m/%Y")
view["P vitória casa"] = (view["P vitória casa"] * 100).round(1).astype(str) + "%"
st.dataframe(view.tail(100).iloc[::-1].rename(columns={"date":"Data", "home_team":"Casa", "away_team":"Visitante"}), use_container_width=True, hide_index=True)

st.subheader("Probabilidade para um jogo")
team_names = sorted(set(games["home_team"].astype(str)) | set(games["away_team"].astype(str)))
left, right = st.columns(2)
with left: home_pick = st.selectbox("Equipa da casa", team_names, index=team_names.index("Boston Celtics") if "Boston Celtics" in team_names else 0)
with right: away_pick = st.selectbox("Equipa visitante", [x for x in team_names if x != home_pick], index=0)
st.info("A previsão para um jogo futuro ainda não está ligada à API de calendário; esta primeira versão serve para validar o modelo no histórico.")

st.download_button("Descarregar resultados do teste CSV", data=view.to_csv(index=False).encode("utf-8-sig"), file_name="nba_moneyline_backtest.csv", mime="text/csv")
st.caption("Importante: o modelo não utiliza odds de bookmaker. Brier/Log Loss e acerto avaliam probabilidades; não provam que exista vantagem financeira.")
