import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")

# Compact, two-column workspace: fixtures on the left, both markets on the right.
st.markdown("""
<style>
:root { --bg:#0a1018; --panel:#111b27; --line:#26384b; --muted:#8fa4ba; --white:#f2f6fb; --blue:#68c4ff; --green:#35d49a; }
.stApp {background:radial-gradient(ellipse at 10% 0%,rgba(27,91,133,.13),transparent 32%),var(--bg);color:var(--white)}
.block-container {max-width:1720px;padding:2.8rem 1.1rem .8rem!important}
header[data-testid="stHeader"] {background:#0a1018!important;height:2.45rem!important}
header[data-testid="stHeader"] [data-testid="stToolbar"] {background:transparent!important}
h1,h2,h3,p,label {margin-top:0}
h1 {font-size:1.3rem!important;line-height:1.4!important;letter-spacing:-.025em;font-weight:850!important;margin:0!important;padding:0!important;overflow:visible!important}
h2 {font-size:1.05rem!important;font-weight:800!important}
h3 {font-size:.95rem!important;font-weight:800!important}
p,label,[data-testid="stCaptionContainer"] p {color:var(--muted)}
hr {margin:.6rem 0!important;border-color:var(--line)!important}
.topbar {display:flex;align-items:center;gap:11px;padding:10px 14px;margin-bottom:10px;min-height:62px;box-sizing:border-box;overflow:visible;border:1px solid #263d51;border-radius:10px;background:linear-gradient(100deg,#132638,#111b27 70%,#12251f)}
.ball {font-size:1.55rem;line-height:1;flex:0 0 auto}
.brand-sub {font-size:.73rem;color:#9eb2c7;margin-top:2px;line-height:1.35}
.status {margin-left:auto;font-size:.66rem;color:#53dca9;font-weight:800;letter-spacing:.08em;white-space:nowrap}
.section-label {font-size:.67rem;text-transform:uppercase;letter-spacing:.12em;font-weight:850;color:#83a1bc;margin:0 0 5px}
.left-panel,.right-panel {background:rgba(14,22,32,.7);border:1px solid #243447;border-radius:12px;padding:12px}
.fixture-help {font-size:.72rem;color:#8fa4ba;margin-bottom:8px}
.game-button div[data-testid="stButton"] > button {text-align:left!important;justify-content:flex-start!important;white-space:normal!important;min-height:0!important;height:auto!important;padding:7px 9px!important;line-height:1.25!important;border-radius:7px!important;margin-bottom:2px!important;font-size:.76rem!important}
div[data-testid="stButton"] > button {border:1px solid #2a3d51;background:#121e2b;color:#eaf2fb;font-weight:650;justify-content:flex-start!important;text-align:left!important;transition:background .15s,border-color .15s}
div[data-testid="stButton"] > button:hover {border-color:#68c4ff;background:#182a3c;color:#fff}
div[data-testid="stButton"] > button[kind="primary"] {background:#105b49;border-color:#35d49a;color:#fff}
div[data-testid="stDateInput"] input {background:#eaf1f8!important;color:#182331!important;border-radius:7px!important;min-height:36px}
div[data-testid="stDateInput"] label {font-size:.75rem!important;font-weight:700!important}
div[data-testid="stMetric"] {background:#0c1520;border:1px solid #26384b;border-radius:8px;padding:7px 9px}
div[data-testid="stMetricLabel"] {font-size:.72rem!important;color:#9db0c4!important}
div[data-testid="stMetricValue"] {font-size:1.2rem!important;color:#f5f9ff!important;font-weight:850}
.market-card {border:1px solid #285743;border-radius:8px;padding:9px 10px;background:linear-gradient(140deg,#14251f,#101923);margin-bottom:5px;min-height:78px;box-sizing:border-box}
.market-card.total {border-color:#2a5277;background:linear-gradient(140deg,#142337,#101923)}
.market-kicker {font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;font-weight:850;color:#73dcb5}
.market-card.total .market-kicker {color:#79caff}
.market-title {font-size:1rem;font-weight:850;color:#fff;margin-top:3px}
.market-desc {font-size:.72rem;color:#91a5ba;margin-top:2px}
.match-header {border:1px solid #2a465a;border-radius:9px;padding:9px 12px;background:linear-gradient(110deg,#122131,#101a26);margin:6px 0 8px}
.match-title {font-size:1rem;font-weight:850;color:#fff}
.match-meta {font-size:.72rem;color:#9eb2c7;margin-top:3px}
.small-note {font-size:.69rem;color:#8195a9;line-height:1.4}
[data-testid="stAlert"] {padding:.55rem .75rem!important;border-radius:8px!important}
[data-testid="stNumberInput"] button {background:#eaf1f8!important;color:#182331!important;border:0!important}
.empty-state {padding:24px 14px;text-align:center;border:1px dashed #33485c;border-radius:10px;color:#9aafc4;background:#101923}
div[data-testid="stNumberInput"] input {font-size:.85rem}
[data-testid="stVerticalBlock"] {gap:.35rem}
[data-testid="stHorizontalBlock"] {gap:.6rem}
@media(max-width:900px){.block-container{padding:.65rem}.status{display:none}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="ball">🏀</div>
  <div><h1>NBA LAB <span style="color:#68c4ff">PRO</span></h1>
  <div class="brand-sub">Calendário e análise de mercados · espaço de trabalho compacto</div></div>
  <div class="status">● PRE-GAME WORKSPACE</div>
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
          <div class="match-meta">📅 {selected_date.strftime("%d/%m/%Y")} &nbsp; · &nbsp; 🕒 {selected_game["time"]} (Portugal) &nbsp; · &nbsp; Visitante / Casa</div>
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
                st.success(f"Edge indicativo: +{(market_odd / fair_home - 1) * 100:.1f}%")
            else:
                st.info("Odd abaixo da referência teórica.")
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
                st.markdown('<div class="market-card" style="padding:8px;border-color:#285743"><b style="color:#73dcb5">📈 OVER</b><div class="market-desc">Mais pontos</div></div>', unsafe_allow_html=True)
                if over_line < total_reference:
                    st.success(f"Referência +{total_reference-over_line:.1f} pts")
                else:
                    st.caption("Linha acima da referência.")
            with b:
                st.markdown('<div class="market-card total" style="padding:8px"><b style="color:#79caff">📉 UNDER</b><div class="market-desc">Menos pontos</div></div>', unsafe_allow_html=True)
                if over_line > total_reference:
                    st.success(f"Referência −{over_line-total_reference:.1f} pts")
                else:
                    st.caption("Linha abaixo da referência.")
            st.caption("Médias fixas por equipa; não é uma previsão calibrada.")

st.markdown('<hr><div class="small-note">NBA LAB PRO · Painel compacto de análise. As previsões atuais são demonstrativas e não resultam de um modelo estatístico treinado. Validar com dados históricos, odds e testes fora da amostra antes de decisões reais.</div>', unsafe_allow_html=True)
