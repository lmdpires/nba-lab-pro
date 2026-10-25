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
    "away_team", "home_team", "model_label",
    "moneyline_pick", "p_home", "p_away", "fair_home_odds", "fair_away_odds",
    "home_market_odds", "moneyline_edge_pct",
    "total_reference", "total_line", "total_gap", "total_pick",
    "home_score", "away_score", "actual_total",
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
.game-button div[data-testid="stButton"] > button {text-align:left!important;justify-content:flex-start!important;white-space:normal!important;min-height:0!important;height:auto!important;padding:7px 9px!important;line-height:1.25!important;border-radius:7px!important;margin-bottom:2px!important;font-size:.76rem!important}
div[data-testid="stButton"] > button {border:1px solid #2a3d51;background:#121e2b;color:#eaf2fb;font-weight:650;justify-content:flex-start!important;text-align:left!important;transition:background .15s,border-color .15s}
div[data-testid="stButton"] > button:hover {border-color:#68c4ff;background:#182a3c;color:#fff}
div[data-testid="stButton"] > button[kind="primary"] {background:#105b49;border-color:#35d49a;color:#fff;font-weight:800;box-shadow:inset 3px 0 0 #72f0c4,0 2px 8px rgba(0,0,0,.12)}
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
                    games.append({
                        "id": str(item.get("id", f"{home['name']}-{away['name']}-{start_time}")),
                        "home": home["name"], "away": away["name"],
                        "time": local_dt.strftime("%H:%M"), "timestamp": local_dt.timestamp()
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
                    label = f"{'▸ ' if is_selected else ''}{game['away']} @ {game['home']}"
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
        st.markdown(f"""
        <div class="match-header">
          <div class="match-title">{away} <span style="color:#6c849b;font-weight:500">at</span> {home}</div>
          <div class="match-meta">📅 {selected_date.strftime("%d/%m/%Y")} &nbsp; · &nbsp; 🕒 {selected_game["time"]} (Portugal) &nbsp; · &nbsp; <span style="color:#68c4ff;font-weight:750">VISITANTE → CASA</span></div>
        </div>
        """, unsafe_allow_html=True)

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
                st.metric("Casa", f"{p_home*100:.1f}%", f"Odd {fair_home:.2f}")
            with b:
                st.metric("Fora", f"{p_away*100:.1f}%", f"Odd {fair_away:.2f}")
            market_odd = st.number_input("Odd de mercado · Casa", min_value=1.01, value=1.95, step=0.05,
                                         key=f"market_odd_{selected_game['id']}")
            if market_odd > fair_home:
                st.markdown(f'<div class="edge-box">EDGE INDICATIVO <b style="float:right">+{(market_odd / fair_home - 1) * 100:.1f}%</b></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="edge-box neutral">EDGE INDICATIVO <b style="float:right">Sem vantagem aparente</b></div>', unsafe_allow_html=True)
            st.caption("Estimativa demonstrativa; não validada com dados históricos.")

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

        st.markdown("---")
        st.markdown("**Guardar esta análise**")
        st.caption("Guarda uma fotografia das previsões atuais para as avaliar depois do jogo.")
        if st.button("＋ Guardar previsão para avaliação", key=f"save_prediction_{selected_game['id']}", use_container_width=True):
            log_df = load_log()
            record = {
                "record_id": str(uuid.uuid4())[:8],
                "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "game_date": selected_date.strftime("%Y-%m-%d"),
                "game_id": selected_game["id"],
                "start_time": selected_game["time"],
                "away_team": away,
                "home_team": home,
                "model_label": "Demonstrativo — não calibrado",
                "moneyline_pick": "Casa" if p_home >= p_away else "Fora",
                "p_home": round(float(p_home), 4),
                "p_away": round(float(p_away), 4),
                "fair_home_odds": round(float(fair_home), 3),
                "fair_away_odds": round(float(fair_away), 3),
                "home_market_odds": round(float(market_odd), 2),
                "moneyline_edge_pct": round((float(market_odd) / float(fair_home) - 1) * 100, 2),
                "total_reference": round(float(total_reference), 1),
                "total_line": round(float(over_line), 1),
                "total_gap": round(float(total_reference) - float(over_line), 1),
                "total_pick": "Over" if total_reference > over_line else ("Under" if total_reference < over_line else "Sem seleção"),
                "home_score": None,
                "away_score": None,
                "actual_total": None,
                "moneyline_result": "Pendente",
                "total_result": "Pendente"
            }
            log_df = pd.concat([log_df, pd.DataFrame([record])], ignore_index=True)
            save_log(log_df)
            st.success("Previsão guardada. Regista o resultado final quando o jogo terminar.")

st.markdown('<hr><div class="small-note">NBA LAB PRO · Painel de análise As previsões atuais são demonstrativas e não resultam de um modelo estatístico treinado. Validar com dados históricos, odds e testes fora da amostra antes de decisões reais.</div>', unsafe_allow_html=True)


st.markdown("## Registo de previsões e resultados")
st.caption("Guarda a previsão antes do jogo e regista o resultado final para medir a taxa de acerto.")

log_df = load_log()
if log_df.empty:
    st.info("Ainda não há previsões guardadas. Seleciona um jogo e utiliza «Guardar previsão para avaliação».")
else:
    pending = log_df[log_df["home_score"].isna() | log_df["away_score"].isna()].copy()
    if not pending.empty:
        pending["display"] = pending.apply(
            lambda r: f"{r['game_date']} · {r['away_team']} @ {r['home_team']} · registo {r['record_id']}", axis=1
        )
        chosen_label = st.selectbox("Jogo por avaliar", pending["display"].tolist(), key="pending_result_select")
        chosen_record = pending[pending["display"] == chosen_label].iloc[0]
        result_col1, result_col2, result_col3 = st.columns([1, 1, 1.4], gap="small")
        with result_col1:
            home_score = st.number_input(
                f"Pontos {chosen_record['home_team']} (casa)", min_value=0, max_value=200, value=0, step=1,
                key=f"home_score_{chosen_record['record_id']}"
            )
        with result_col2:
            away_score = st.number_input(
                f"Pontos {chosen_record['away_team']} (fora)", min_value=0, max_value=200, value=0, step=1,
                key=f"away_score_{chosen_record['record_id']}"
            )
        with result_col3:
            st.write("")
            st.write("")
            if st.button("Registar resultado final", key=f"submit_result_{chosen_record['record_id']}", use_container_width=True):
                if home_score == 0 and away_score == 0:
                    st.error("Confirma os pontos antes de guardar o resultado.")
                else:
                    idx = log_df.index[log_df["record_id"].astype(str) == str(chosen_record["record_id"])][0]
                    total_score = int(home_score + away_score)
                    pick_ml = str(log_df.at[idx, "moneyline_pick"])
                    pick_total = str(log_df.at[idx, "total_pick"])
                    if home_score == away_score:
                        ml_result = "Empate — verificar resultado"
                    else:
                        home_won = home_score > away_score
                        away_won = away_score > home_score
                        ml_result = "Acertou" if ((pick_ml == "Casa" and home_won) or (pick_ml == "Fora" and away_won)) else "Falhou"
                    line = float(log_df.at[idx, "total_line"])
                    if pick_total == "Sem seleção" or total_score == line:
                        total_result = "Push/sem seleção"
                    else:
                        total_won = (pick_total == "Over" and total_score > line) or (pick_total == "Under" and total_score < line)
                        total_result = "Acertou" if total_won else "Falhou"
                    log_df.at[idx, "home_score"] = int(home_score)
                    log_df.at[idx, "away_score"] = int(away_score)
                    log_df.at[idx, "actual_total"] = total_score
                    log_df.at[idx, "moneyline_result"] = ml_result
                    log_df.at[idx, "total_result"] = total_result
                    save_log(log_df)
                    st.success(f"Resultado registado · Moneyline: {ml_result} · Total: {total_result}")
                    st.rerun()
    else:
        st.success("Todos os registos guardados têm resultado.")

    st.markdown("### Histórico das previsões guardadas")
    history = log_df.copy()
    history["Prob. casa"] = pd.to_numeric(history["p_home"], errors="coerce").map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    history["Prob. fora"] = pd.to_numeric(history["p_away"], errors="coerce").map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    history["Odd justa casa"] = pd.to_numeric(history["fair_home_odds"], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    history["Odd justa fora"] = pd.to_numeric(history["fair_away_odds"], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    history["Odd mercado casa"] = pd.to_numeric(history["home_market_odds"], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    history["Referência total"] = pd.to_numeric(history["total_reference"], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    history["Diferença vs linha"] = pd.to_numeric(history["total_gap"], errors="coerce").map(lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
    display_cols = [
        "game_date", "away_team", "home_team", "moneyline_pick", "Prob. casa", "Prob. fora",
        "Odd justa casa", "Odd justa fora", "Odd mercado casa", "Referência total",
        "total_line", "Diferença vs linha", "total_pick", "home_score", "away_score",
        "actual_total", "moneyline_result", "total_result"
    ]
    st.dataframe(
        history[display_cols].rename(columns={
            "game_date": "Data", "away_team": "Visitante", "home_team": "Casa",
            "moneyline_pick": "Moneyline previsto", "total_line": "Linha mercado",
            "total_pick": "Total previsto", "home_score": "Pontos casa",
            "away_score": "Pontos visitante", "actual_total": "Total real",
            "moneyline_result": "Resultado ML", "total_result": "Resultado O/U"
        }),
        use_container_width=True, hide_index=True
    )
    with st.expander("Ver fotografia completa de cada previsão"):
        for _, row in log_df.iloc[::-1].iterrows():
            title = f"{row['game_date']} · {row['away_team']} @ {row['home_team']} · {row['record_id']}"
            st.markdown(f"**{title}**")
            detail1, detail2, detail3 = st.columns(3)
            with detail1:
                st.write(f"Modelo: {row.get('model_label', '—')}")
                st.write(f"Prob. casa / fora: {float(row['p_home'])*100:.1f}% / {float(row['p_away'])*100:.1f}%")
                st.write(f"Odds justas casa / fora: {row.get('fair_home_odds', '—')} / {row.get('fair_away_odds', '—')}")
            with detail2:
                st.write(f"Odd mercado casa: {row.get('home_market_odds', '—')}")
                st.write(f"Edge indicativo: {row.get('moneyline_edge_pct', '—')}%")
                st.write(f"Escolha Moneyline: {row['moneyline_pick']}")
            with detail3:
                st.write(f"Total de referência: {row.get('total_reference', '—')} pts")
                st.write(f"Linha de mercado: {row.get('total_line', '—')} pts")
                st.write(f"Diferença referência/linha: {row.get('total_gap', '—')} pts")
                st.write(f"Escolha total: {row['total_pick']}")
            st.caption(f"Guardado em: {row.get('saved_at', '—')} · {row.get('model_label', 'Modelo antigo/sem etiqueta')}")
            st.divider()

    finished_ml = log_df[log_df["moneyline_result"].isin(["Acertou", "Falhou"])]
    finished_total = log_df[log_df["total_result"].isin(["Acertou", "Falhou"])]
    col1, col2, col3 = st.columns(3)
    with col1:
        rate = (finished_ml["moneyline_result"] == "Acertou").mean() * 100 if not finished_ml.empty else 0
        st.metric("Acerto Moneyline", f"{rate:.1f}%", f"{len(finished_ml)} jogos avaliados")
    with col2:
        rate = (finished_total["total_result"] == "Acertou").mean() * 100 if not finished_total.empty else 0
        st.metric("Acerto Over/Under", f"{rate:.1f}%", f"{len(finished_total)} jogos avaliados")
    with col3:
        st.metric("Previsões guardadas", int(len(log_df)))
    st.download_button(
        "Descarregar backup CSV",
        data=log_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="nba_prediction_log_backup.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.warning("O histórico fica num CSV no ambiente da aplicação. Na Streamlit Cloud, descarrega backups regularmente: ficheiros locais podem não sobreviver a reinícios ou novas implementações.")
