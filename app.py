import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")

# ------------------------- DESIGN SYSTEM -------------------------
st.markdown("""
<style>
:root {
  --bg: #090e15;
  --panel: #111a26;
  --panel2: #152130;
  --line: #29384a;
  --muted: #91a2b7;
  --white: #f4f7fb;
  --blue: #5bb9ff;
  --green: #35d49a;
  --amber: #ffbf69;
}
.stApp {
  background:
    radial-gradient(ellipse at 8% -5%, rgba(28,104,157,.18), transparent 33%),
    radial-gradient(ellipse at 100% 0%, rgba(25,115,91,.10), transparent 27%),
    var(--bg);
  color: var(--white);
}
.block-container {max-width: 1540px; padding: 1.6rem 2rem 3rem 2rem;}
h1,h2,h3,h4 {color: var(--white) !important; letter-spacing: -.025em;}
h1 {font-size: 2.35rem !important; font-weight: 850 !important;}
h2 {font-size: 1.5rem !important; font-weight: 780 !important;}
h3 {font-size: 1.13rem !important; font-weight: 760 !important;}
p, label {color: var(--muted);}
hr {border-color: #233041 !important; margin: 1.25rem 0 !important;}
[data-testid="stCaptionContainer"] p {color: #91a2b7;}
.hero {
  border: 1px solid #2b4054; border-radius: 18px; padding: 22px 25px;
  background: linear-gradient(112deg, #13283a 0%, #111b28 56%, #11251f 100%);
  margin-bottom: 22px;
}
.hero-top {display:flex; align-items:center; gap:14px;}
.hero-ball {font-size:2.55rem; line-height:1;}
.hero-title {font-size:2.05rem; font-weight:900; letter-spacing:-.045em; color:#fff;}
.hero-sub {color:#a9bbce; margin-top:5px; font-size:.98rem;}
.hero-right {margin-left:auto; text-align:right; color:#a9bbce; font-size:.8rem;}
.eyebrow {font-size:.72rem; text-transform:uppercase; letter-spacing:.13em; font-weight:850; color:#83a0bb; margin-bottom:5px;}
.section-title {font-size:1.3rem; font-weight:820; color:#f5f8fc; margin-bottom:4px;}
.panel {
  background: linear-gradient(145deg, rgba(19,29,42,.98), rgba(14,22,32,.98));
  border:1px solid #29384a; border-radius:15px; padding:17px 18px;
}
.panel-head {display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px;}
.panel-note {font-size:.84rem; color:#91a2b7;}
.date-badge {display:inline-block; padding:5px 9px; border-radius:7px; background:#1a2c3e; border:1px solid #304c65; color:#a9d9ff; font-size:.8rem; font-weight:750;}
div[data-testid="stDateInput"] input {
  background:#edf3f9 !important; color:#17202b !important;
  border:1px solid #c6d3e0 !important; border-radius:9px !important; min-height:43px;
}
div[data-testid="stDateInput"] label {color:#b6c5d7 !important; font-weight:700;}
div[data-testid="stButton"] > button {
  border-radius:10px; border:1px solid #2a3a4d;
  background:#141f2d; color:#eaf2fb; font-weight:700;
  min-height:43px; transition:all .14s ease;
}
div[data-testid="stButton"] > button:hover {
  border-color:#5bb9ff; color:#fff; background:#1a2b3d;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background:linear-gradient(100deg,#087c5a,#11624c);
  border:1px solid #35d49a; color:#fff;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background:linear-gradient(100deg,#0b946b,#16745a);
}
div[data-testid="stMetric"] {
  background:#0d1621; border:1px solid #2a3a4d; border-radius:12px; padding:14px;
}
div[data-testid="stMetricLabel"] {color:#9db0c4 !important;}
div[data-testid="stMetricValue"] {color:#f7fbff !important; font-weight:850;}
div[data-testid="stTabs"] button {font-weight:750 !important;}
.fixture-row {
  border:1px solid #29384a; border-radius:11px; background:#111a26;
  padding:12px 14px; margin-bottom:8px;
}
.fixture-team {font-size:.96rem; font-weight:760; color:#f2f6fb;}
.fixture-meta {font-size:.78rem; color:#92a5ba; margin-top:4px;}
.fixture-time {
  display:inline-block; border:1px solid #2d4d69; background:#142b3e;
  color:#7bc8ff; padding:5px 8px; border-radius:7px;
  font-size:.78rem; font-weight:800; white-space:nowrap;
}
.market {
  height:100%; border-radius:15px; padding:18px;
  background:linear-gradient(145deg,#14231f,#101821);
  border:1px solid #285b49;
}
.market.total {background:linear-gradient(145deg,#142337,#101821); border-color:#2b5277;}
.market-kicker {font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; font-weight:850; color:#77dcb4;}
.market.total .market-kicker {color:#7cc9ff;}
.market-heading {font-size:1.2rem; font-weight:850; color:#fff; margin:7px 0 3px;}
.market-copy {font-size:.84rem; color:#9eafc2; min-height:40px;}
.market-value {font-size:2.45rem; line-height:1.1; font-weight:900; color:#f5fbff; margin:16px 0 4px;}
.market-subvalue {font-size:.85rem; color:#a7b8ca;}
.mini-rule {height:1px; background:#2b3b4b; margin:15px 0;}
.selection-banner {
  background:linear-gradient(100deg,#14283a,#14251f);
  border:1px solid #2b4658; border-radius:14px; padding:17px 19px; margin:10px 0 17px;
}
.selected-title {font-size:1.42rem; font-weight:880; color:#fff;}
.selected-sub {font-size:.83rem; color:#a1b3c6; margin-top:5px;}
.empty-state {
  border:1px dashed #33485c; background:#101923; border-radius:13px;
  padding:26px 20px; text-align:center; color:#9eafc2;
}
.empty-icon {font-size:2rem; margin-bottom:7px;}
.disclaimer {border-top:1px solid #233041; padding-top:12px; margin-top:22px; color:#8495a9; font-size:.8rem;}
@media(max-width: 900px) {
  .block-container {padding-left:1rem; padding-right:1rem;}
  .hero-title {font-size:1.65rem;}
  .hero-right {display:none;}
  .market-value {font-size:2rem;}
}
</style>
""", unsafe_allow_html=True)

# ------------------------- HEADER -------------------------
st.markdown("""
<div class="hero">
  <div class="hero-top">
    <div class="hero-ball">🏀</div>
    <div>
      <div class="hero-title">NBA LAB <span style="color:#65c5ff">PRO</span></div>
      <div class="hero-sub">NBA intelligence desk · calendário, probabilidades e análise de mercados</div>
    </div>
    <div class="hero-right">PRE-GAME ANALYSIS<br><span style="color:#35d49a">● WORKSPACE</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    API_KEY = st.secrets["API_SPORTS_NBA_KEY"]
except Exception:
    API_KEY = ""

MEDIAS_REAIS = {
    "Atlanta Hawks": 116.5, "Boston Celtics": 120.8, "Brooklyn Nets": 111.2,
    "Charlotte Hornets": 108.5, "Chicago Bulls": 112.3, "Cleveland Cavaliers": 114.8,
    "Dallas Mavericks": 117.9, "Denver Nuggets": 115.4, "Detroit Pistons": 110.1,
    "Golden State Warriors": 118.2, "Houston Rockets": 113.5, "Indiana Pacers": 121.2,
    "LA Clippers": 115.1, "Los Angeles Lakers": 116.8, "Memphis Grizzlies": 113.8,
    "Miami Heat": 110.5, "Milwaukee Bucks": 119.4, "Minnesota Timberwolves": 113.2,
    "New Orleans Pelicans": 115.0, "New York Knicks": 112.8, "Oklahoma City Thunder": 119.5,
    "Orlando Magic": 110.4, "Philadelphia 76ers": 114.2, "Phoenix Suns": 116.2,
    "Portland Trail Blazers": 106.8, "Sacramento Kings": 116.6, "San Antonio Spurs": 112.1,
    "Toronto Raptors": 112.5, "Utah Jazz": 115.7, "Washington Wizards": 113.1
}

# ------------------------- FETCH FIXTURES -------------------------
date_col, list_col = st.columns([0.78, 1.85], gap="large")
games = []
with date_col:
    st.markdown('<div class="eyebrow">01 / Matchday</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Escolher data</div>', unsafe_allow_html=True)
    selected_date = st.date_input("Calendário", value=datetime.date.today(), key="nba_matchday")
    date_str = selected_date.strftime("%Y-%m-%d")
    st.markdown(f'<span class="date-badge">📅 {selected_date.strftime("%A, %d %b %Y")}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if API_KEY:
        st.success("API ligada", icon="🟢")
    else:
        st.error("Configura API_SPORTS_NBA_KEY nas Secrets.", icon="🔑")
    st.caption("A lista é atualizada quando alteras a data.")

with list_col:
    st.markdown('<div class="eyebrow">02 / Official schedule</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Jogos do dia <span style="color:#7d91a8;font-size:.9rem;font-weight:600">· {selected_date.strftime("%d/%m/%Y")}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">Seleciona um confronto para abrir os dois mercados abaixo.</div>', unsafe_allow_html=True)
    if API_KEY:
        season = str(selected_date.year if selected_date.month >= 10 else selected_date.year - 1)
        params = {"date": date_str, "season": season, "league": "standard"}
        try:
            response = requests.get(
                "https://v2.nba.api-sports.io/games",
                headers={"x-apisports-key": API_KEY},
                params=params,
                timeout=20
            )
            if response.status_code != 200:
                st.error(f"Erro HTTP {response.status_code}: {response.text[:400]}")
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

games.sort(key=lambda x: x["timestamp"])
unique_games, seen = [], set()
for game in games:
    key = (game["home"], game["away"], game["time"])
    if key not in seen:
        seen.add(key)
        unique_games.append(game)
games = unique_games

# Reset selection when selected date changes, or when current game is absent.
current_game_ids = {g["id"] for g in games}
if st.session_state.get("nba_selection_date") != date_str:
    st.session_state["nba_selection_date"] = date_str
    st.session_state["nba_selected_game"] = games[0]["id"] if games else None
elif st.session_state.get("nba_selected_game") not in current_game_ids:
    st.session_state["nba_selected_game"] = games[0]["id"] if games else None

if games:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    for game in games:
        is_selected = st.session_state.get("nba_selected_game") == game["id"]
        row_cols = st.columns([4.6, 1.0, 1.15], gap="small")
        with row_cols[0]:
            marker = "●" if is_selected else "○"
            color = "#35d49a" if is_selected else "#647a91"
            st.markdown(
                f"""<div class="fixture-row">
                <div style="font-size:.72rem;font-weight:850;letter-spacing:.1em;color:{color};margin-bottom:5px">{marker} {"SELECTED GAME" if is_selected else "NBA MATCHUP"}</div>
                <div class="fixture-team">{game['away']} <span style="color:#71879d;font-weight:500">at</span> {game['home']}</div>
                <div class="fixture-meta">Visitante <span style="color:#647b91">/</span> Casa</div>
                </div>""",
                unsafe_allow_html=True
            )
        with row_cols[1]:
            st.markdown(f'<div style="padding-top:19px;text-align:center"><span class="fixture-time">🕒 {game["time"]}</span></div>', unsafe_allow_html=True)
        with row_cols[2]:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("✓ Selecionado" if is_selected else "Analisar jogo →",
                         key=f"choose_{date_str}_{game['id']}",
                         type="primary" if is_selected else "secondary"):
                st.session_state["nba_selected_game"] = game["id"]
                st.rerun()
else:
    if API_KEY:
        st.markdown('<div class="empty-state"><div class="empty-icon">🗓️</div><b style="color:#f3f7fb">Sem jogos para esta data</b><br>Escolhe outro dia para carregar os confrontos disponíveis.</div>', unsafe_allow_html=True)
    else:
        st.info("Quando a chave API estiver configurada, os jogos oficiais aparecerão aqui.")

selected_game = next((g for g in games if g["id"] == st.session_state.get("nba_selected_game")), None)

# ------------------------- TWO MARKET PANELS -------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">03 / Matchup analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Análise do confronto</div>', unsafe_allow_html=True)

if not selected_game:
    st.markdown('<div class="empty-state"><div class="empty-icon">👆</div><b style="color:#f3f7fb">Seleciona um jogo</b><br>As previsões Moneyline e Over / Under aparecem aqui, lado a lado.</div>', unsafe_allow_html=True)
else:
    home = selected_game["home"]
    away = selected_game["away"]
    st.markdown(
        f"""<div class="selection-banner">
        <div class="eyebrow">Selected matchup · {selected_date.strftime("%d %b %Y")}</div>
        <div class="selected-title">{away} <span style="color:#5c7690">at</span> {home}</div>
        <div class="selected-sub">🕒 {selected_game["time"]} · Hora de Portugal &nbsp;&nbsp; | &nbsp;&nbsp; Visitante vs Casa</div>
        </div>""",
        unsafe_allow_html=True
    )

    # Demonstration-only heuristics preserved from the existing version.
    factor = (len(home) + len(away) * 3) % 10
    p_home = min(0.90, max(0.10, 0.54 + factor * 0.015))
    p_away = 1 - p_home
    fair_home = 1 / p_home
    fair_away = 1 / p_away
    total_reference = MEDIAS_REAIS.get(home, 114.5) + MEDIAS_REAIS.get(away, 112.2)

    market1, market2 = st.columns(2, gap="large")
    with market1:
        st.markdown(f"""
        <div class="market">
          <div class="market-kicker">Mercado 01 · Moneyline</div>
          <div class="market-heading">🏆 Vencedor do jogo</div>
          <div class="market-copy">Probabilidade estimada para cada equipa e odd teórica correspondente.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2, gap="small")
        with m1:
            st.metric(f"{home} · Casa", f"{p_home*100:.1f}%", f"Odd teórica {fair_home:.2f}")
        with m2:
            st.metric(f"{away} · Fora", f"{p_away*100:.1f}%", f"Odd teórica {fair_away:.2f}")
        market_odd = st.number_input(
            f"Odd de mercado — vitória {home}",
            min_value=1.01, value=1.95, step=0.05,
            key=f"market_odd_{selected_game['id']}"
        )
        if market_odd > fair_home:
            edge = (market_odd / fair_home - 1) * 100
            st.success(f"Odd de mercado {edge:.1f}% acima da odd teórica demonstrativa.")
        else:
            st.info("A odd introduzida não supera a odd teórica demonstrativa.")
        st.caption("Estimativa heurística, ainda não validada com dados históricos.")

    with market2:
        st.markdown("""
        <div class="market total">
          <div class="market-kicker">Mercado 02 · Total points</div>
          <div class="market-heading">🏀 Over / Under</div>
          <div class="market-copy">Referência de pontos para a partida e comparação com a linha de mercado.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.metric("Total de pontos · referência", f"{total_reference:.1f}")
        over_line = st.number_input(
            "Linha de mercado — total",
            min_value=150.0, max_value=300.0, value=220.5, step=0.5,
            key=f"total_line_{selected_game['id']}"
        )
        over_col, under_col = st.columns(2, gap="small")
        with over_col:
            st.markdown('<div class="panel"><div style="color:#5bd6a6;font-weight:850">📈 OVER</div><div class="panel-note">Mais pontos</div></div>', unsafe_allow_html=True)
            if over_line < total_reference:
                st.success(f"Linha {total_reference-over_line:.1f} pts abaixo da referência.")
            else:
                st.info("A linha não está abaixo da referência.")
        with under_col:
            st.markdown('<div class="panel"><div style="color:#70c5ff;font-weight:850">📉 UNDER</div><div class="panel-note">Menos pontos</div></div>', unsafe_allow_html=True)
            if over_line > total_reference:
                st.success(f"Linha {over_line-total_reference:.1f} pts acima da referência.")
            else:
                st.info("A linha não está acima da referência.")
        st.caption("Soma de médias fixas por equipa; não é uma previsão calibrada.")

st.markdown("""
<div class="disclaimer">
NBA LAB PRO · Interface de análise, não casa de apostas. As probabilidades atuais são demonstrativas
e não resultam de um modelo treinado. Antes de usar previsões para decisões reais, é necessário
validar o modelo com dados históricos, odds de mercado e testes fora da amostra.
</div>
""", unsafe_allow_html=True)
