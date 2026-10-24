import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")

# -------------------- VISUAL DESIGN --------------------
st.markdown("""
<style>
:root {
    --bg: #0b0f15;
    --surface: #121923;
    --surface-2: #182230;
    --border: #263445;
    --muted: #9aa9bb;
    --text: #f4f7fb;
    --blue: #39a9ff;
    --green: #23d18b;
    --orange: #ffae57;
}
.stApp {
    background:
      radial-gradient(ellipse at 12% 0%, rgba(25,91,145,.17), transparent 36%),
      radial-gradient(ellipse at 92% 8%, rgba(35,209,139,.08), transparent 28%),
      var(--bg);
    color: var(--text);
}
.block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -.025em; }
h1 { font-weight: 850 !important; font-size: clamp(2rem, 3vw, 3rem) !important; }
h2 { font-size: 1.55rem !important; font-weight: 760 !important; }
h3 { font-size: 1.22rem !important; font-weight: 720 !important; }
p, label, .stCaption { color: var(--muted); }
div[data-testid="stCaptionContainer"] p { color: #91a0b2; }
hr { border-color: #202c3a !important; margin: 1.4rem 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(22,32,45,.98), rgba(15,22,32,.98));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
}
div[data-testid="stDateInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #eaf0f7 !important;
    color: #17202b !important;
    border-radius: 9px !important;
    border: 1px solid #c7d2df !important;
    min-height: 43px;
}
div[data-testid="stDateInput"] label,
div[data-testid="stNumberInput"] label { color: #aebbd0 !important; font-weight: 600; }
div[data-testid="stButton"] > button {
    width: 100%;
    min-height: 62px;
    white-space: normal;
    text-align: left;
    line-height: 1.35;
    padding: 12px 14px;
    border-radius: 11px;
    border: 1px solid #2a394b;
    background: linear-gradient(145deg, #182230, #111923);
    color: #eef5ff;
    font-weight: 650;
    box-shadow: 0 3px 12px rgba(0,0,0,.12);
    transition: all .16s ease-in-out;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--blue);
    background: #1b2a3b;
    color: #ffffff;
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(110deg, #087e61, #0e594d);
    border: 1px solid #27d79a;
    color: #ffffff;
    box-shadow: 0 0 0 1px rgba(35,209,139,.12), 0 5px 18px rgba(0,0,0,.2);
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(110deg, #09916f, #126a5b);
    border-color: #69efbf;
}
div[data-testid="stTabs"] button {
    color: #aebbd0 !important;
    font-weight: 700 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #45dba3 !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: var(--green) !important;
}
div[data-testid="stMetric"] {
    background: #111a25;
    border: 1px solid #29384a;
    border-radius: 13px;
    padding: 16px;
}
div[data-testid="stMetricLabel"] { color: #a8b6c9 !important; }
div[data-testid="stMetricValue"] { color: #f5f9ff !important; font-weight: 800; }
div[data-testid="stAlert"] { border-radius: 11px; }
.small-label {
    color: #9aabc0; text-transform: uppercase; font-size: .76rem;
    letter-spacing: .1em; font-weight: 800; margin-bottom: 6px;
}
.hero {
    background: linear-gradient(115deg, rgba(20,48,70,.9), rgba(16,29,43,.88) 58%, rgba(18,46,40,.75));
    border: 1px solid #2b4052; border-radius: 18px; padding: 22px 24px; margin: 8px 0 22px 0;
}
.hero-sub { color: #a9bbcf; font-size: 1rem; margin-top: 5px; }
.game-header {
    display:flex; justify-content:space-between; align-items:center; gap:12px;
    flex-wrap:wrap; padding: 2px 0 12px 0;
}
.time-pill {
    display:inline-block; border:1px solid rgba(57,169,255,.3);
    background:rgba(57,169,255,.1); color:#67c1ff; border-radius:8px;
    padding:5px 9px; font-size:.83rem; font-weight:750;
}
.market-card {
    border-radius: 15px; padding: 21px; height:100%;
    background: linear-gradient(145deg, #15251f, #111923);
    border: 1px solid #275d49;
}
.market-card-blue {
    border-radius: 15px; padding: 21px; height:100%;
    background: linear-gradient(145deg, #142437, #111923);
    border: 1px solid #27527a;
}
.market-title { color:#f6fbff; font-size:1.12rem; font-weight:800; margin-bottom:5px; }
.market-sub { color:#9bacc0; font-size:.9rem; }
.kicker { color:#9eafc3; text-transform:uppercase; font-size:.75rem; letter-spacing:.1em; font-weight:800; }
.disclaimer { color:#8392a5; font-size:.82rem; border-top:1px solid #202c3a; padding-top:12px; margin-top:22px; }
@media (max-width: 800px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  div[data-testid="stButton"] > button { min-height: 56px; }
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
  <div style="font-size:2.1rem;font-weight:900;letter-spacing:-.04em;color:#fff">🏀 NBA LAB PRO</div>
  <div class="hero-sub">Calendário de jogos, seleção de confrontos e análise de mercados num só lugar.</div>
</div>
""", unsafe_allow_html=True)

# Chave guardada em Streamlit Secrets.
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

# -------------------- DATE + FIXTURES --------------------
col_date, col_fixtures = st.columns([0.78, 1.7], gap="large")
jogos = []

with col_date:
    st.markdown('<div class="small-label">01 / Calendário</div>', unsafe_allow_html=True)
    st.subheader("Escolher dia")
    data = st.date_input("Data dos jogos", value=datetime.date.today(), key="nba_date")
    data_str = data.strftime("%Y-%m-%d")
    st.caption("Seleciona o dia para carregar os confrontos oficiais.")
    st.markdown("---")
    st.markdown('<div class="small-label">Estado da ligação</div>', unsafe_allow_html=True)
    if API_KEY:
        st.success("Chave API configurada")
    else:
        st.error("Falta configurar a chave API nas Secrets.")

with col_fixtures:
    st.markdown('<div class="small-label">02 / Calendário oficial</div>', unsafe_allow_html=True)
    st.subheader(f"Jogos de {data.strftime('%d/%m/%Y')}")
    if API_KEY:
        season = str(data.year if data.month >= 10 else data.year - 1)
        params = {"date": data_str, "season": season, "league": "standard"}
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
                        start = (item.get("date") or {}).get("start")
                        if not home.get("name") or not away.get("name") or not start:
                            continue
                        dt_pt = pd.to_datetime(start, utc=True).tz_convert("Europe/Lisbon")
                        jogos.append({
                            "id": str(item.get("id", f"{home['name']}-{away['name']}-{start}")),
                            "casa": home["name"],
                            "fora": away["name"],
                            "hora": dt_pt.strftime("%H:%M"),
                            "timestamp": dt_pt.timestamp()
                        })
        except requests.RequestException as exc:
            st.error(f"Erro de ligação à API-NBA: {exc}")
        except ValueError as exc:
            st.error(f"A API não devolveu JSON válido: {exc}")

jogos = sorted(jogos, key=lambda j: j["timestamp"])
dedup = []
seen = set()
for j in jogos:
    k = (j["casa"], j["fora"], j["hora"])
    if k not in seen:
        seen.add(k)
        dedup.append(j)
jogos = dedup

if jogos:
    available_ids = {j["id"] for j in jogos}
    if st.session_state.get("nba_selected_game") not in available_ids:
        st.session_state["nba_selected_game"] = jogos[0]["id"]

    st.markdown("#### Clica num confronto para abrir os mercados")
    # Dois cartões por linha; a seleção fica visualmente destacada.
    for start_idx in range(0, len(jogos), 2):
        row = st.columns(2, gap="medium")
        for offset, col in enumerate(row):
            idx = start_idx + offset
            if idx >= len(jogos):
                continue
            j = jogos[idx]
            selected = st.session_state.get("nba_selected_game") == j["id"]
            label = f"{'✓  ' if selected else '🏀  '}{j['casa']}  vs  {j['fora']}\\n🕒 {j['hora']} · Hora de Portugal"
            with col:
                if st.button(label, key=f"game_{data_str}_{j['id']}", type="primary" if selected else "secondary"):
                    st.session_state["nba_selected_game"] = j["id"]
                    st.rerun()
else:
    if API_KEY:
        st.info("Não foram devolvidos jogos para esta data. Experimenta outro dia ou confirma os dados devolvidos pela API.")
    else:
        st.info("Depois de configurares a chave API, os jogos do dia aparecem aqui.")

selected_game = next((j for j in jogos if j["id"] == st.session_state.get("nba_selected_game")), None)

# -------------------- MARKETS --------------------
st.markdown("---")
st.markdown('<div class="small-label">03 / Análise do confronto</div>', unsafe_allow_html=True)
st.subheader("Mercados do jogo selecionado")

if not selected_game:
    st.markdown("""
    <div style="border:1px dashed #334458;border-radius:14px;padding:28px;text-align:center;background:#111923">
      <div style="font-size:2rem">👆</div>
      <div style="font-size:1.05rem;font-weight:750;color:#eef5ff;margin:8px 0">Escolhe um jogo na lista acima</div>
      <div style="color:#91a0b2">Os mercados Moneyline e Over / Under aparecem aqui depois da seleção.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    casa = selected_game["casa"]
    fora = selected_game["fora"]
    st.markdown(
        f"""<div class="game-header">
          <div><div class="kicker">Confronto selecionado</div>
          <div style="font-size:1.5rem;font-weight:850;color:#fff;margin-top:4px">{casa} <span style="color:#5b718a">vs</span> {fora}</div></div>
          <div class="time-pill">🕒 {selected_game['hora']} · Portugal</div>
        </div>""",
        unsafe_allow_html=True
    )

    tab_moneyline, tab_total = st.tabs(["🏆  MONEYLINE · Vencedor", "🏀  OVER / UNDER · Total de pontos"])

    # Mantidos como placeholders demonstrativos: não são um modelo treinado.
    factor = (len(casa) + len(fora) * 3) % 10
    prob_home = min(0.90, max(0.10, 0.54 + factor * 0.015))
    prob_away = 1 - prob_home
    fair_home = 1 / prob_home
    fair_away = 1 / prob_away

    with tab_moneyline:
        st.markdown('<div class="kicker">Mercado 01</div>', unsafe_allow_html=True)
        st.markdown("### Qual é a estimativa para o vencedor?")
        left, right = st.columns(2, gap="medium")
        with left:
            st.metric(f"{casa} · Casa", f"{prob_home*100:.1f}%", f"Odd teórica {fair_home:.2f}")
        with right:
            st.metric(f"{fora} · Fora", f"{prob_away*100:.1f}%", f"Odd teórica {fair_away:.2f}")
        st.markdown("")
        odd_market = st.number_input(
            f"Odd de mercado para {casa} vencer",
            min_value=1.01, value=1.95, step=0.05,
            key=f"odd_ml_{selected_game['id']}"
        )
        if odd_market > fair_home:
            advantage = (odd_market / fair_home - 1) * 100
            st.success(f"A odd está {advantage:.1f}% acima da odd teórica demonstrativa. Isto não comprova valor real.")
        else:
            st.info("A odd de mercado não supera a odd teórica desta estimativa demonstrativa.")

    with tab_total:
        mean_home = MEDIAS_REAIS.get(casa, 114.5)
        mean_away = MEDIAS_REAIS.get(fora, 112.2)
        reference_total = mean_home + mean_away
        st.markdown('<div class="kicker">Mercado 02</div>', unsafe_allow_html=True)
        st.markdown("### Total de pontos da partida")
        st.metric("Referência demonstrativa", f"{reference_total:.1f} pontos")
        over_col, under_col = st.columns(2, gap="medium")
        with over_col:
            st.markdown('<div class="market-card"><div class="market-title">📈 OVER · Mais pontos</div><div class="market-sub">Compara a linha de mercado com a referência.</div></div>', unsafe_allow_html=True)
            over_line = st.number_input(
                "Linha de mercado — OVER", min_value=150.0, max_value=300.0,
                value=220.5, step=0.5, key=f"over_{selected_game['id']}"
            )
            if over_line < reference_total:
                st.success("A linha está abaixo da referência demonstrativa.")
            else:
                st.info("A linha não está abaixo da referência demonstrativa.")
        with under_col:
            st.markdown('<div class="market-card-blue"><div class="market-title">📉 UNDER · Menos pontos</div><div class="market-sub">Compara a linha de mercado com a referência.</div></div>', unsafe_allow_html=True)
            under_line = st.number_input(
                "Linha de mercado — UNDER", min_value=150.0, max_value=300.0,
                value=220.5, step=0.5, key=f"under_{selected_game['id']}"
            )
            if under_line > reference_total:
                st.success("A linha está acima da referência demonstrativa.")
            else:
                st.info("A linha não está acima da referência demonstrativa.")

st.markdown("""
<div class="disclaimer">
Nota técnica: as probabilidades de Moneyline são heurísticas e as médias de pontos são fixas.
Esta interface melhora a navegação e apresentação, mas as estimativas não devem ser tratadas como
previsões validadas nem como recomendação de aposta.
</div>
""", unsafe_allow_html=True)
