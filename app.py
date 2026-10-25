import datetime
import uuid
from pathlib import Path
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")

# Registo CSV local; descarregar backups regularmente se a app estiver na Streamlit Cloud.
LOG_PATH = Path(__file__).with_name("nba_prediction_log.csv")
LOG_COLUMNS = [
    "record_id", "saved_at", "game_date", "game_id", "start_time",
    "freeze_at", "away_team", "home_team", "model_label", "capture_mode",
    "moneyline_pick", "p_home", "p_away", "fair_home_odds", "fair_away_odds",
    "home_market_odds", "away_market_odds", "moneyline_edge_pct", "away_moneyline_edge_pct",
    "total_reference", "total_line", "total_gap", "total_pick",
    "home_score", "away_score", "actual_total", "game_status", "result_updated_at",
    "moneyline_result", "total_result"
]

def load_log():
    if LOG_PATH.exists():
        try:
            df = pd.read_csv(LOG_PATH)
            for col in LOG_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[LOG_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLUMNS)

def save_log(df):
    df.to_csv(LOG_PATH, index=False)

def model_snapshot(home, away):
    """Same transparent demo heuristics as the UI; not a trained prediction model."""
    factor = (len(home) + len(away) * 3) % 10
    p_home = min(0.90, max(0.10, 0.54 + factor * 0.015))
    p_away = 1 - p_home
    total_reference = MEDIAS_REAIS.get(home, 114.5) + MEDIAS_REAIS.get(away, 112.2)
    return p_home, p_away, total_reference

def evaluate_record(row, home_score, away_score):
    """Evaluate saved Moneyline and total picks against the official final score."""
    total_score = int(home_score) + int(away_score)
    pick_ml = str(row.get("moneyline_pick", ""))
    if int(home_score) == int(away_score):
        ml_result = "Push"
    else:
        home_won = int(home_score) > int(away_score)
        ml_result = "Acertou" if ((pick_ml == "Casa" and home_won) or
                                  (pick_ml == "Fora" and not home_won)) else "Falhou"

    try:
        line = float(row.get("total_line"))
    except (TypeError, ValueError):
        line = None
    pick_total = str(row.get("total_pick", ""))
    if line is None or pick_total in ("", "Sem seleção", "nan"):
        total_result = "Sem linha/seleção"
    elif total_score == line:
        total_result = "Push"
    else:
        won = (pick_total == "Over" and total_score > line) or (pick_total == "Under" and total_score < line)
        total_result = "Acertou" if won else "Falhou"
    return total_score, ml_result, total_result

def upsert_api_results(games, log_df):
    """Apply API final scores to previously frozen predictions."""
    if log_df.empty:
        return log_df, 0

    # CSV inference can turn columns containing mostly blanks into float columns.
    # Pandas 3 / NumPy then reject string assignments (e.g. result_updated_at).
    # Normalize mutable columns to object before writing mixed text/numeric results.
    mutable_columns = [
        "home_score", "away_score", "actual_total", "game_status",
        "result_updated_at", "moneyline_result", "total_result"
    ]
    for column in mutable_columns:
        if column not in log_df.columns:
            log_df[column] = None
        log_df[column] = log_df[column].astype("object")

    changed = 0
    by_id = {normalize_game_id(g.get("id")): g for g in games}
    for idx, row in log_df.iterrows():
        game = by_id.get(normalize_game_id(row.get("game_id")))
        if not game or game.get("status") != "FT":
            continue
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if home_score is None or away_score is None:
            continue
        if str(row.get("game_status", "")) == "Final" and pd.notna(row.get("actual_total")):
            continue
        total_score, ml_result, total_result = evaluate_record(row, home_score, away_score)
        log_df.at[idx, "home_score"] = int(home_score)
        log_df.at[idx, "away_score"] = int(away_score)
        log_df.at[idx, "actual_total"] = int(total_score)
        log_df.at[idx, "game_status"] = "Final"
        log_df.at[idx, "result_updated_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        log_df.at[idx, "moneyline_result"] = ml_result
        log_df.at[idx, "total_result"] = total_result
        changed += 1
    return log_df, changed

def auto_freeze_predictions(games, log_df, now=None):
    """Freeze a demo snapshot at T-5 when the app is running / refreshed."""
    now = now or datetime.datetime.now().astimezone()
    if log_df.empty:
        existing_ids = set()
    else:
        existing_ids = set(log_df["game_id"].astype(str))
    records = []
    for game in games:
        try:
            kickoff = datetime.datetime.fromtimestamp(float(game["timestamp"]), tz=now.tzinfo)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        freeze_at = kickoff - datetime.timedelta(minutes=5)
        if now < freeze_at or str(game["id"]) in existing_ids:
            continue
        if game.get("status") in ("FT", "LIVE", "Q1", "Q2", "Q3", "Q4", "HT", "OT"):
            # Do not create a new "pre-game" prediction once play has started.
            continue
        home, away = game["home"], game["away"]
        p_home, p_away, total_reference = model_snapshot(home, away)
        # No bookmaker odds feed is assumed here. Odds are deliberately left blank.
        # The total line is the app's clearly labelled fallback until a market line is entered.
        total_line = 220.5
        records.append({
            "record_id": str(uuid.uuid4())[:8],
            "saved_at": now.isoformat(timespec="seconds"),
            "game_date": kickoff.strftime("%Y-%m-%d"),
            "game_id": str(game["id"]),
            "start_time": kickoff.strftime("%H:%M"),
            "freeze_at": freeze_at.isoformat(timespec="seconds"),
            "away_team": away, "home_team": home,
            "model_label": "Demonstrativo — não calibrado",
            "capture_mode": "Automático T-5",
            "moneyline_pick": "Casa" if p_home >= p_away else "Fora",
            "p_home": round(float(p_home), 4), "p_away": round(float(p_away), 4),
            "fair_home_odds": round(1 / p_home, 3), "fair_away_odds": round(1 / p_away, 3),
            "home_market_odds": None, "away_market_odds": None,
            "moneyline_edge_pct": None, "away_moneyline_edge_pct": None,
            "total_reference": round(float(total_reference), 1),
            "total_line": total_line,
            "total_gap": round(float(total_reference) - total_line, 1),
            "total_pick": "Over" if total_reference > total_line else ("Under" if total_reference < total_line else "Sem seleção"),
            "home_score": None, "away_score": None, "actual_total": None,
            "game_status": "Pendente", "result_updated_at": None,
            "moneyline_result": "Pendente", "total_result": "Pendente"
        })
        existing_ids.add(str(game["id"]))
    if records:
        log_df = pd.concat([log_df, pd.DataFrame(records)], ignore_index=True)
    return log_df, len(records)

# Compact, two-column workspace: fixtures on the left, both markets on the right.
st.markdown("""
<style>
:root { --bg:#0a1018; --panel:#111b27; --line:#26384b; --muted:#8fa4ba; --white:#f2f6fb; --blue:#68c4ff; --green:#35d49a; }
.stApp {background:radial-gradient(ellipse at 10% 0%,rgba(27,91,133,.13),transparent 32%),var(--bg);color:var(--white)}
.block-container {max-width:1720px;padding:2.25rem 1.1rem .9rem!important}
header[data-testid="stHeader"] {background:#0a1018!important;height:2rem!important;min-height:2rem!important;border-bottom:1px solid #172432!important}
header[data-testid="stHeader"] [data-testid="stToolbar"] {background:transparent!important;height:2rem!important;min-height:2rem!important}
h1,h2,h3,p,label {margin-top:0}
h1 {font-size:1.42rem!important;line-height:1.4!important;letter-spacing:-.025em;font-weight:900!important;margin:0!important;padding:0!important;overflow:visible!important}
h2 {font-size:1.05rem!important;font-weight:800!important}
h3 {font-size:.95rem!important;font-weight:800!important}
p,label,[data-testid="stCaptionContainer"] p {color:var(--muted)}
hr {margin:.6rem 0!important;border-color:var(--line)!important}
.topbar {display:flex;align-items:center;gap:13px;padding:10px 16px;margin-bottom:11px;min-height:66px;box-sizing:border-box;overflow:visible;border:1px solid #2c4559;border-radius:10px;background:linear-gradient(100deg,#14283a,#111b27 70%,#12251f);box-shadow:0 5px 18px rgba(0,0,0,.12)}
.ball {font-size:1.55rem;line-height:1;flex:0 0 auto}
.brand-sub {font-size:.74rem;color:#b0c2d4;margin-top:3px;line-height:1.4}
.status {margin-left:auto;font-size:.66rem;color:#53dca9;font-weight:800;letter-spacing:.09em;white-space:nowrap;padding:6px 9px;border:1px solid #245344;border-radius:999px;background:rgba(16,76,59,.22)}
.section-label {font-size:.67rem;text-transform:uppercase;letter-spacing:.12em;font-weight:850;color:#83a1bc;margin:0 0 6px}
.left-panel,.right-panel {background:rgba(14,22,32,.7);border:1px solid #243447;border-radius:12px;padding:12px}
.fixture-help {font-size:.72rem;color:#8fa4ba;margin-bottom:8px}
.game-button div[data-testid="stButton"] > button {text-align:center!important;justify-content:center!important;white-space:normal!important;min-height:0!important;height:auto!important;padding:9px 10px!important;line-height:1.3!important;border-radius:8px!important;margin-bottom:3px!important;font-size:.78rem!important;font-weight:650!important;letter-spacing:.005em!important}\n.game-button div[data-testid="stButton"] > button[kind="primary"] {background:linear-gradient(105deg,#0e6b55,#125844)!important;border:1px solid #35d9a2!important;border-left:4px solid #64f3c0!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;font-weight:850!important;font-size:.82rem!important;letter-spacing:.01em!important;box-shadow:0 0 0 1px rgba(53,217,162,.12),0 4px 12px rgba(0,0,0,.18)!important}\n.game-button div[data-testid="stButton"] > button[kind="primary"] p {color:#ffffff!important;font-weight:850!important}\n.game-button div[data-testid="stButton"] > button[kind="primary"]:hover {background:linear-gradient(105deg,#11785f,#17694f)!important;border-color:#73f4c5!important}
div[data-testid="stButton"] > button {border:1px solid #2a3d51;background:#121e2b;color:#eaf2fb;font-weight:650;justify-content:flex-start!important;text-align:left!important;transition:background .15s,border-color .15s}
div[data-testid="stButton"] > button:hover {border-color:#68c4ff;background:#182a3c;color:#fff}
div[data-testid="stButton"] > button[kind="primary"] {background:#105b49;border-color:#35d49a;color:#fff!important;-webkit-text-fill-color:#fff!important;font-weight:800;box-shadow:inset 3px 0 0 #72f0c4,0 2px 8px rgba(0,0,0,.12)}
.st-key-save_prediction_area {background:linear-gradient(100deg,rgba(17,53,52,.7),rgba(16,27,40,.98) 72%);border:1px solid #2b665c;border-left:4px solid #35d9a2;border-radius:10px;padding:12px 14px 13px;margin-top:10px;margin-bottom:12px}
.st-key-save_prediction_area [data-testid="stMarkdownContainer"] p {margin-bottom:4px}
.st-key-save_prediction_area div[data-testid="stButton"] > button {background:linear-gradient(100deg,#147b63,#176b60)!important;border:1px solid #45d8ad!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;font-size:.9rem!important;font-weight:800!important;letter-spacing:.005em!important;min-height:46px!important;height:46px!important;padding:10px 14px!important;border-radius:8px!important;justify-content:center!important;text-align:center!important;box-shadow:0 3px 10px rgba(0,0,0,.16)!important}
.st-key-save_prediction_area div[data-testid="stButton"] > button:hover {background:linear-gradient(100deg,#199575,#1b8171)!important;border-color:#83f2d0!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;transform:translateY(-1px)}

div[data-testid="stDateInput"] input {background:#eaf1f8!important;color:#182331!important;border-radius:7px!important;min-height:36px}
div[data-testid="stDateInput"] label {font-size:.75rem!important;font-weight:700!important}
div[data-testid="stMetric"] {background:#0c1520;border:1px solid #26384b;border-radius:8px;padding:7px 9px;min-height:88px;box-sizing:border-box}
div[data-testid="stMetricLabel"] {font-size:.72rem!important;color:#9db0c4!important}
div[data-testid="stMetricValue"] {font-size:1.2rem!important;color:#f5f9ff!important;font-weight:850}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {margin-top:3px}
.market-card {border:1px solid #285743;border-radius:8px;padding:9px 10px;background:linear-gradient(140deg,#14251f,#101923);margin-bottom:5px;min-height:78px;box-sizing:border-box}
.market-card.total {border-color:#2a5277;background:linear-gradient(140deg,#142337,#101923)}
.market-kicker {font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;font-weight:850;color:#73dcb5}
.market-card.total .market-kicker {color:#79caff}
.market-title {font-size:1rem;font-weight:850;color:#fff;margin-top:3px}
.market-desc {font-size:.72rem;color:#91a5ba;margin-top:2px}
.match-header {border:1px solid #3b607a;border-left:4px solid #68c4ff;border-radius:10px;padding:13px 15px;background:linear-gradient(105deg,#172c40 0%,#122131 62%,#101a26 100%);margin:6px 0 10px;box-shadow:0 5px 18px rgba(0,0,0,.13)}
.match-title {font-size:1.35rem;line-height:1.35;font-weight:900;letter-spacing:-.025em;color:#fff}
.match-meta {font-size:.76rem;color:#b5c8d9;margin-top:6px;line-height:1.4}
.small-note {font-size:.69rem;color:#8195a9;line-height:1.4}
[data-testid="stAlert"] {padding:.55rem .75rem!important;border-radius:8px!important}
[data-testid="stNumberInput"] button {background:#eaf1f8!important;color:#182331!important;border:0!important}
.empty-state {padding:24px 14px;text-align:center;border:1px dashed #33485c;border-radius:10px;color:#9aafc4;background:#101923}
div[data-testid="stNumberInput"] input {font-size:.85rem}
[data-testid="stVerticalBlock"] {gap:.35rem}
[data-testid="stHorizontalBlock"] {gap:.6rem}
.stColumn:has(.market-card) {background:linear-gradient(180deg,rgba(17,29,42,.82),rgba(12,21,31,.78));border:1px solid #26384b;border-radius:11px;padding:9px 10px 8px;box-sizing:border-box}
.stColumn:has(.market-card) .market-card {margin-top:0;margin-bottom:7px;min-height:76px}
.stColumn:has(.market-card) [data-testid="stMetric"] {height:100%;box-sizing:border-box}
.stColumn:has(.market-card) label {font-size:.74rem!important}
.edge-box {padding:9px 11px;border-radius:8px;font-size:.82rem;margin:4px 0 7px;border:1px solid #25473e;background:#0c241f;color:#9de8ca}
.edge-box.neutral {border-color:#344454;background:#111c28;color:#b8c7d7}
.outcome-box {padding:8px 9px;border-radius:8px;border:1px solid #285743;background:#10231f;min-height:58px;box-sizing:border-box}
.outcome-box.under {border-color:#2a5277;background:#111f2d}
.outcome-title {font-size:.78rem;font-weight:850;letter-spacing:.02em}
.outcome-sub {font-size:.68rem;color:#91a5ba;margin-top:3px}
.result-line {font-size:.76rem;padding:7px 9px;margin-top:4px;border-radius:7px;background:#0c241f;color:#9de8ca}
.result-line.neutral {background:#111c28;color:#9aafc4}
.final-score-panel {
  border:1px solid #2c5878; border-left:5px solid #55ddb0; border-radius:12px;
  padding:14px 18px; margin:8px 0 12px; background:linear-gradient(110deg,#10291f,#101d2c 75%);
}
.final-score-kicker {font-size:.72rem;letter-spacing:.12em;font-weight:900;color:#6ee7b7;margin-bottom:10px}
.final-score-teams {display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:14px}
.final-team {display:flex;align-items:center;justify-content:space-between;gap:10px;color:#dce8f4;font-weight:750}
.final-team span {font-size:1rem}
.final-team strong {font-size:2rem;line-height:1;color:#fff;font-weight:900}
.final-vs {font-size:1.5rem;color:#7790a8;font-weight:800}
.final-total {border-top:1px solid #294456;margin-top:12px;padding-top:9px;display:flex;justify-content:space-between;color:#9eb4c9;font-size:.72rem;letter-spacing:.06em;font-weight:800}
.final-total strong {color:#fff;font-size:.95rem}
.market-outcome {border:1px solid #2b4054;border-radius:10px;padding:12px 14px;margin:3px 0 12px;min-height:124px;background:#101c29}
.outcome-heading {font-size:.72rem;letter-spacing:.08em;font-weight:900;color:#9ab0c6}
.outcome-pick {font-size:.95rem;font-weight:750;color:#eef5fb;margin:7px 0}
.outcome-result {font-size:1rem;font-weight:950;letter-spacing:.03em;padding:7px 10px;border-radius:7px;display:inline-block;margin:2px 0 7px}
.outcome-detail {font-size:.76rem;color:#a8bacb}
.market-outcome.won {border-color:#239b72;background:linear-gradient(130deg,#103b30,#10241f)}
.market-outcome.won .outcome-result {color:#09291f;background:#66e0ae}
.market-outcome.lost {border-color:#b84e59;background:linear-gradient(130deg,#401e29,#241922)}
.market-outcome.lost .outcome-result {color:#fff;background:#c84f5b}
.market-outcome.push {border-color:#b58a35;background:linear-gradient(130deg,#3b3018,#211f1a)}
.market-outcome.push .outcome-result {color:#2b2109;background:#e7c36b}
.market-outcome.pending .outcome-result {color:#d8e5f1;background:#304356}
div[data-testid="stAlert"] {margin:.25rem 0!important}

@media(max-width:900px){.block-container{padding:.65rem}.status{display:none}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="ball">🏀</div>
  <div><h1>NBA LAB <span style="color:#68c4ff">PRO</span></h1>
  <div class="brand-sub">Calendário e análise de mercados · espaço de trabalho compacto</div></div>
  <div class="status">● PRÉ-JOGO · NBA</div>
</div>
""", unsafe_allow_html=True)

try:
    API_KEY = st.secrets["API_SPORTS_NBA_KEY"]
except Exception:
    API_KEY = ""

MEDIAS_REAIS = {
    "Atlanta Hawks":116.5,"Boston Celtics":120.8,"Brooklyn Nets":111.2,"Charlotte Hornets":108.5,
    "Chicago Bulls":112.3,"Cleveland Cavaliers":114.8,"Dallas Mavericks":117.9,"Denver Nuggets":115.4,
    "Detroit Pistons":110.1,"Golden State Warriors":118.2,"Houston Rockets":113.5,"Indiana Pacers":121.2,
    "LA Clippers":115.1,"Los Angeles Lakers":116.8,"Memphis Grizzlies":113.8,"Miami Heat":110.5,
    "Milwaukee Bucks":119.4,"Minnesota Timberwolves":113.2,"New Orleans Pelicans":115.0,
    "New York Knicks":112.8,"Oklahoma City Thunder":119.5,"Orlando Magic":110.4,
    "Philadelphia 76ers":114.2,"Phoenix Suns":116.2,"Portland Trail Blazers":106.8,
    "Sacramento Kings":116.6,"San Antonio Spurs":112.1,"Toronto Raptors":112.5,
    "Utah Jazz":115.7,"Washington Wizards":113.1
}

# Main workspace: date and fixtures on the left, both market panels on the right.
left, right = st.columns([0.9, 2.1], gap="medium")
with left:
    st.markdown('<div class="section-label">01 · Jogos</div>', unsafe_allow_html=True)
    selected_date = st.date_input("Data dos jogos", value=datetime.date.today(), key="nba_matchday", format="DD/MM/YYYY")
    date_str = selected_date.strftime("%Y-%m-%d")

def normalize_game_status(status_obj):
    """Normalize API-Sports status variants into a small stable set."""
    status_obj = status_obj or {}
    short = str(status_obj.get("short") or "").strip().upper()
    long_status = str(status_obj.get("long") or "").strip().lower()
    if short in {"FT", "AOT", "FTOT", "FINAL", "FINISHED", "ENDED"} or any(
        word in long_status for word in ("finished", "final", "game over", "ended", "terminé", "terminado")
    ):
        return "FT"
    if short in {"Q1", "Q2", "Q3", "Q4", "OT", "HT", "LIVE", "IN"} or any(
        word in long_status for word in ("in play", "live", "quarter", "halftime", "overtime", "en cours")
    ):
        return short if short in {"Q1", "Q2", "Q3", "Q4", "OT", "HT"} else "LIVE"
    if short in {"NS", "TBD", "SCHED", "PST"} or any(
        word in long_status for word in ("not started", "scheduled", "postponed", "time to be defined")
    ):
        return "NS"
    return short or ("FT" if "final" in long_status else "NS")

def normalize_score(value):
    """Handle both numeric scores and nested API-Sports score objects."""
    if isinstance(value, dict):
        for key in ("points", "total", "score"):
            if value.get(key) is not None:
                return normalize_score(value[key])
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def normalize_game_id(value):
    if value is None:
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()

# Load fixtures for the chosen date.
games = []
if API_KEY:
    season = str(selected_date.year if selected_date.month >= 10 else selected_date.year - 1)
    try:
        response = requests.get(
            "https://v2.nba.api-sports.io/games",
            headers={"x-apisports-key": API_KEY},
            params={"date": date_str, "season": season, "league": "standard"},
            timeout=20
        )
        if response.status_code != 200:
            st.error(f"Erro HTTP {response.status_code}: {response.text[:250]}")
        else:
            payload = response.json()
            if payload.get("errors"):
                st.error(f"Erro da API-NBA: {payload['errors']}")
            else:
                for item in payload.get("response", []):
                    teams = item.get("teams") or {}
                    home = teams.get("home") or {}
                    away = teams.get("visitors") or teams.get("away") or {}
                    start_time = (item.get("date") or {}).get("start")
                    if not home.get("name") or not away.get("name") or not start_time:
                        continue
                    local_dt = pd.to_datetime(start_time, utc=True).tz_convert("Europe/Lisbon")
                    status_obj = item.get("status") or {}
                    scores_obj = item.get("scores") or {}
                    home_score = normalize_score(scores_obj.get("home"))
                    away_score = normalize_score(scores_obj.get("visitors") or scores_obj.get("away"))
                    games.append({
                        "id": str(item.get("id", f"{home['name']}-{away['name']}-{start_time}")),
                        "home": home["name"], "away": away["name"],
                        "time": local_dt.strftime("%H:%M"), "timestamp": local_dt.timestamp(),
                        "status": normalize_game_status(status_obj),
                        "status_long": str(status_obj.get("long") or ""),
                        "home_score": home_score, "away_score": away_score
                    })
    except requests.RequestException as exc:
        st.error(f"Erro de ligação à API-NBA: {exc}")
    except ValueError as exc:
        st.error(f"A resposta da API não é JSON válido: {exc}")

games.sort(key=lambda g: g["timestamp"])
deduped, seen = [], set()
for game in games:
    key = (game["home"], game["away"], game["time"])
    if key not in seen:
        seen.add(key)
        deduped.append(game)
games = deduped

def parse_api_game(item):
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("visitors") or teams.get("away") or {}
    start_time = (item.get("date") or {}).get("start")
    if not home.get("name") or not away.get("name") or not start_time:
        return None
    local_dt = pd.to_datetime(start_time, utc=True).tz_convert("Europe/Lisbon")
    status_obj = item.get("status") or {}
    scores_obj = item.get("scores") or {}
    home_score = normalize_score(scores_obj.get("home"))
    away_score = normalize_score(scores_obj.get("visitors") or scores_obj.get("away"))
    return {
        "id": str(item.get("id", f"{home['name']}-{away['name']}-{start_time}")),
        "home": home["name"], "away": away["name"],
        "time": local_dt.strftime("%H:%M"), "timestamp": local_dt.timestamp(),
        "status": normalize_game_status(status_obj),
        "status_long": str(status_obj.get("long") or ""),
        "home_score": home_score, "away_score": away_score
    }

# Load the persistent prediction history and fetch any past dates that still need results.
log_df = load_log()
result_games_by_id = {str(g["id"]): g for g in games}
if API_KEY and not log_df.empty:
    unresolved_dates = sorted({
        str(d) for d in log_df.loc[
            ~log_df["game_status"].astype(str).str.lower().eq("final"), "game_date"
        ].dropna().tolist()
        if str(d) and str(d) != date_str
    })
    # Query only dates represented in the log, rather than re-fetching the entire season.
    for historical_date in unresolved_dates:
        try:
            hist_dt = datetime.date.fromisoformat(historical_date)
            hist_season = str(hist_dt.year if hist_dt.month >= 10 else hist_dt.year - 1)
            hist_response = requests.get(
                "https://v2.nba.api-sports.io/games",
                headers={"x-apisports-key": API_KEY},
                params={"date": historical_date, "season": hist_season, "league": "standard"},
                timeout=15
            )
            if hist_response.status_code != 200:
                continue
            hist_payload = hist_response.json()
            for item in hist_payload.get("response", []):
                parsed = parse_api_game(item)
                if parsed:
                    result_games_by_id[str(parsed["id"])] = parsed
        except (requests.RequestException, ValueError, TypeError):
            continue

# Automatic prediction freeze and result reconciliation. Runs whenever the app reruns.
log_df, new_snapshots = auto_freeze_predictions(games, log_df)
log_df, updated_results = upsert_api_results(list(result_games_by_id.values()), log_df)
if new_snapshots or updated_results:
    save_log(log_df)

current_ids = {g["id"] for g in games}
if st.session_state.get("nba_selection_date") != date_str:
    st.session_state["nba_selection_date"] = date_str
    st.session_state["nba_selected_game"] = games[0]["id"] if games else None
elif st.session_state.get("nba_selected_game") not in current_ids:
    st.session_state["nba_selected_game"] = games[0]["id"] if games else None

selected_game = next((g for g in games if g["id"] == st.session_state.get("nba_selected_game")), None)

with left:
    st.markdown(f'<div class="fixture-help">{selected_date.strftime("%A, %d %B %Y")} · {len(games)} jogo(s)</div>', unsafe_allow_html=True)
    if not API_KEY:
        st.warning("Configura `API_SPORTS_NBA_KEY` nas Secrets.")
    elif not games:
        st.markdown('<div class="empty-state">Não há jogos nesta data.<br>Experimenta outro dia.</div>', unsafe_allow_html=True)
    else:
        # Compact rows; each button is one line when possible.
        with st.container(height=455, border=False):
            for game in games:
                is_selected = st.session_state.get("nba_selected_game") == game["id"]
                time_col, match_col = st.columns([0.23, 0.77], gap="small", vertical_alignment="center")
                with time_col:
                    st.markdown(
                        f'<div style="font-size:.72rem;font-weight:750;color:{"#55ddb0" if is_selected else "#8fa4ba"};text-align:center;padding:7px 0">{game["time"]}</div>',
                        unsafe_allow_html=True
                    )
                with match_col:
                    st.markdown('<div class="game-button">', unsafe_allow_html=True)
                    status = game.get("status", "NS")
                    if status == "FT" and game.get("home_score") is not None and game.get("away_score") is not None:
                        label = f"{game['home']} {game['home_score']}–{game['away_score']} {game['away']} · FINAL"
                    elif status in ("LIVE", "Q1", "Q2", "Q3", "Q4", "HT", "OT"):
                        label = f"🔴 {game['home']} vs {game['away']} · EM DIRETO"
                    else:
                        label = f"▸ **{game['home']} vs {game['away']}**" if is_selected else f"{game['home']} vs {game['away']}"
                    if st.button(label, key=f"choose_{date_str}_{game['id']}",
                                 type="primary" if is_selected else "secondary", use_container_width=True):
                        st.session_state["nba_selected_game"] = game["id"]
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Seleciona qualquer jogo para atualizar os mercados à direita.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">02 · Análise dos mercados</div>', unsafe_allow_html=True)
    if not selected_game:
        st.markdown('<div class="empty-state">Escolhe um jogo na lista à esquerda.<br>Os dois mercados aparecem aqui, lado a lado.</div>', unsafe_allow_html=True)
    else:
        home, away = selected_game["home"], selected_game["away"]
        status_code = selected_game.get("status", "NS")
        if status_code == "FT" and selected_game.get("home_score") is not None and selected_game.get("away_score") is not None:
            status_label = f"FINAL · {selected_game['home_score']}–{selected_game['away_score']}"
        elif status_code in ("LIVE", "Q1", "Q2", "Q3", "Q4", "HT", "OT"):
            status_label = "EM DIRETO"
        else:
            status_label = "PRÉ-JOGO"
        st.markdown(f"""
        <div class="match-header">
          <div class="match-title">{home} <span style="color:#6c849b;font-weight:500">vs</span> {away}</div>
          <div class="match-meta">📅 {selected_date.strftime("%d/%m/%Y")} &nbsp; · &nbsp; 🕒 {selected_game["time"]} (Portugal) &nbsp; · &nbsp; <span style="color:#68c4ff;font-weight:750">CASA → VISITANTE</span> &nbsp; · &nbsp; <b>{status_label}</b></div>
        </div>
        """, unsafe_allow_html=True)

        if status_code == "FT" and selected_game.get("home_score") is not None and selected_game.get("away_score") is not None:
            final_home = int(selected_game["home_score"])
            final_away = int(selected_game["away_score"])
            final_total = final_home + final_away
            st.markdown(f"""
            <div class="final-score-panel">
              <div class="final-score-kicker">🏁 RESULTADO FINAL</div>
              <div class="final-score-teams">
                <div class="final-team"><span>{home}</span><strong>{final_home}</strong></div>
                <div class="final-vs">–</div>
                <div class="final-team"><span>{away}</span><strong>{final_away}</strong></div>
              </div>
              <div class="final-total">TOTAL DE PONTOS <strong>{final_total}</strong></div>
            </div>
            """, unsafe_allow_html=True)
            saved_row = log_df[
                log_df["game_id"].map(normalize_game_id) == normalize_game_id(selected_game["id"])
            ] if not log_df.empty else pd.DataFrame()
            if not saved_row.empty:
                saved = saved_row.iloc[-1]

                def outcome_card(title, pick, result, detail=""):
                    result_text = str(result or "Pendente")
                    if result_text == "Acertou":
                        css_class, icon_text = "won", "✓ ACERTOU"
                    elif result_text == "Falhou":
                        css_class, icon_text = "lost", "✕ FALHOU"
                    elif result_text == "Push":
                        css_class, icon_text = "push", "＝ PUSH"
                    else:
                        css_class, icon_text = "pending", "… PENDENTE"
                    return f"""
                    <div class="market-outcome {css_class}">
                      <div class="outcome-heading">{title}</div>
                      <div class="outcome-pick">{pick}</div>
                      <div class="outcome-result">{icon_text}</div>
                      <div class="outcome-detail">{detail}</div>
                    </div>"""

                ml_pick = str(saved.get("moneyline_pick", "—"))
                ml_detail = f"Vencedor real: {home if final_home > final_away else away if final_away > final_home else 'Empate'}"
                total_pick = str(saved.get("total_pick", "—"))
                total_line_value = saved.get("total_line", "—")
                total_detail = f"Total real: {final_total} pts · Linha: {total_line_value}"
                outcome1 = outcome_card("MONEYLINE", f"Seleção: {ml_pick}", saved.get("moneyline_result", "Pendente"), ml_detail)
                outcome2 = outcome_card("OVER / UNDER", f"Seleção: {total_pick}", saved.get("total_result", "Pendente"), total_detail)
                out1, out2 = st.columns(2, gap="small")
                with out1:
                    st.markdown(outcome1, unsafe_allow_html=True)
                with out2:
                    st.markdown(outcome2, unsafe_allow_html=True)
            else:
                st.info("O resultado final está disponível, mas não existe uma previsão congelada para este jogo. Não é possível classificar um acerto retroativamente.")

        # MARKET 1: Moneyline
        market1, market2 = st.columns(2, gap="small")
        with market1:
            st.markdown("""
            <div class="market-card">
              <div class="market-kicker">Mercado 01 · Moneyline</div>
              <div class="market-title">🏆 Vencedor do jogo</div>
              <div class="market-desc">Probabilidade e odd teórica por equipa.</div>
            </div>
            """, unsafe_allow_html=True)
            factor = (len(home) + len(away) * 3) % 10
            p_home = min(0.90, max(0.10, 0.54 + factor * 0.015))
            p_away = 1 - p_home
            fair_home, fair_away = 1 / p_home, 1 / p_away
            a, b = st.columns(2, gap="small")
            with a:
                st.metric(f"Casa · {home}", f"{p_home*100:.1f}%", f"Odd justa {fair_home:.2f}")
            with b:
                st.metric(f"Visitante · {away}", f"{p_away*100:.1f}%", f"Odd justa {fair_away:.2f}")
            # Use native Streamlit labels so both inputs align and never overlap.
            odds_home_col, odds_away_col = st.columns(2, gap="small")
            with odds_home_col:
                market_odd_home = st.number_input(
                    "Odd Casa", min_value=1.01, value=1.95, step=0.05,
                    help=f"Odd de mercado para {home}",
                    key=f"market_odd_home_{selected_game['id']}"
                )
            with odds_away_col:
                market_odd_away = st.number_input(
                    "Odd Visitante", min_value=1.01, value=1.95, step=0.05,
                    help=f"Odd de mercado para {away}",
                    key=f"market_odd_away_{selected_game['id']}"
                )
            edge_home = (market_odd_home / fair_home - 1) * 100
            edge_away = (market_odd_away / fair_away - 1) * 100
            edge_col1, edge_col2 = st.columns(2, gap="small")
            with edge_col1:
                if edge_home > 0:
                    st.markdown(f'<div class="edge-box">EDGE · CASA <b style="float:right">+{edge_home:.1f}%</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="edge-box neutral">EDGE · CASA <b style="float:right">Sem vantagem</b></div>', unsafe_allow_html=True)
            with edge_col2:
                if edge_away > 0:
                    st.markdown(f'<div class="edge-box">EDGE · VISITANTE <b style="float:right">+{edge_away:.1f}%</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="edge-box neutral">EDGE · VISITANTE <b style="float:right">Sem vantagem</b></div>', unsafe_allow_html=True)
            st.caption("Introduz cada odd no campo da equipa correspondente. Estimativas demonstrativas, não validadas com dados históricos.")

        # MARKET 2: Total points / Over Under
        with market2:
            st.markdown("""
            <div class="market-card total">
              <div class="market-kicker">Mercado 02 · Total de pontos</div>
              <div class="market-title">🏀 Over / Under</div>
              <div class="market-desc">Referência total de pontos vs linha do mercado.</div>
            </div>
            """, unsafe_allow_html=True)
            total_reference = MEDIAS_REAIS.get(home, 114.5) + MEDIAS_REAIS.get(away, 112.2)
            st.metric("Total de referência", f"{total_reference:.1f} pts")
            over_line = st.number_input("Linha de mercado", min_value=150.0, max_value=300.0, value=220.5, step=0.5,
                                        key=f"total_line_{selected_game['id']}")
            a, b = st.columns(2, gap="small")
            with a:
                st.markdown('<div class="outcome-box"><div class="outcome-title" style="color:#73dcb5">📈 OVER</div><div class="outcome-sub">Mais pontos</div></div>', unsafe_allow_html=True)
                if over_line < total_reference:
                    st.markdown(f'<div class="result-line">Referência <b>+{total_reference-over_line:.1f} pts</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-line neutral">Sem vantagem pela referência</div>', unsafe_allow_html=True)
            with b:
                st.markdown('<div class="outcome-box under"><div class="outcome-title" style="color:#79caff">📉 UNDER</div><div class="outcome-sub">Menos pontos</div></div>', unsafe_allow_html=True)
                if over_line > total_reference:
                    st.markdown(f'<div class="result-line">Referência <b>−{over_line-total_reference:.1f} pts</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-line neutral">Sem vantagem pela referência</div>', unsafe_allow_html=True)
            st.caption("Médias fixas por equipa; não é uma previsão calibrada.")

        st.info("A previsão é congelada automaticamente quando a aplicação está ativa/é atualizada a partir dos 5 minutos antes do início. As odds de bookmaker ainda não são fornecidas automaticamente nesta versão.")

st.markdown('<hr><div class="small-note">NBA LAB PRO · Painel de análise As previsões atuais são demonstrativas e não resultam de um modelo estatístico treinado. Validar com dados históricos, odds e testes fora da amostra antes de decisões reais.</div>', unsafe_allow_html=True)


st.markdown("## Resultados e avaliação automática")
st.caption("Os resultados finais vêm da API-Sports NBA. As previsões são congeladas automaticamente a partir de T−5 quando a aplicação executa uma atualização.")

log_df = load_log()
if log_df.empty:
    st.info("Ainda não existem previsões congeladas. Mantém a aplicação aberta ou atualiza-a perto do início dos jogos.")
else:
    finished_ml = log_df[log_df["moneyline_result"].isin(["Acertou", "Falhou"])]
    finished_total = log_df[log_df["total_result"].isin(["Acertou", "Falhou"])]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rate = (finished_ml["moneyline_result"] == "Acertou").mean() * 100 if not finished_ml.empty else 0
        st.metric("Acerto Moneyline", f"{rate:.1f}%", f"{len(finished_ml)} avaliados")
    with c2:
        rate = (finished_total["total_result"] == "Acertou").mean() * 100 if not finished_total.empty else 0
        st.metric("Acerto Over/Under", f"{rate:.1f}%", f"{len(finished_total)} avaliados")
    with c3:
        st.metric("Previsões congeladas", int(len(log_df)))
    with c4:
        st.metric("Resultados finais", int(log_df["game_status"].eq("Final").sum()))

    history = log_df.copy()
    history["Resultado"] = history.apply(
        lambda r: f"{int(r['home_score'])}–{int(r['away_score'])}" if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")) else "—",
        axis=1
    )
    history["Total real"] = pd.to_numeric(history["actual_total"], errors="coerce")
    history["Linha"] = pd.to_numeric(history["total_line"], errors="coerce")
    history["Prob. casa"] = pd.to_numeric(history["p_home"], errors="coerce").map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    history["Prob. visitante"] = pd.to_numeric(history["p_away"], errors="coerce").map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    display_cols = [
        "game_date", "home_team", "away_team", "Resultado",
        "moneyline_pick", "moneyline_result", "Prob. casa", "Prob. visitante",
        "total_reference", "Linha", "total_pick", "Total real", "total_result", "capture_mode"
    ]
    st.dataframe(
        history[display_cols].rename(columns={
            "game_date": "Data", "home_team": "Casa", "away_team": "Visitante",
            "moneyline_pick": "Moneyline previsto", "moneyline_result": "Moneyline",
            "total_reference": "Total referência", "total_pick": "Over/Under previsto",
            "total_result": "Over/Under", "capture_mode": "Registo"
        }),
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "Descarregar backup CSV",
        data=log_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="nba_prediction_log_backup.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.warning("Limitação importante: o registo automático só acontece quando o Streamlit executa uma atualização. Para garantir o congelamento exato em T−5, é necessário um agendador externo ou manter uma atualização periódica ativa. A linha total usada automaticamente é uma referência demonstrativa de 220,5, não uma linha de bookmaker. As odds de mercado ficam vazias nesta versão.")
    pending_final = log_df[~log_df["game_status"].astype(str).str.lower().eq("final")]
    if not pending_final.empty:
        st.caption(f"Há {len(pending_final)} previsões ainda sem resultado final confirmado pela API. Os resultados são reconciliados pelas datas guardadas no histórico em cada atualização.")
