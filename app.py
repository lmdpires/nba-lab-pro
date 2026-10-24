import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")
st.markdown("""
<style>
.stApp {background:#0E1116;color:#F3F4F6;font-family:'Helvetica Neue',sans-serif}
h1,h2,h3 {color:#fff!important;font-weight:800}
.badge-time {background:rgba(0,145,234,.15);color:#00B0FF;padding:5px 9px;border-radius:6px;font-weight:bold;border:1px solid rgba(0,145,234,.25)}
div.stButton>button {width:100%;min-height:48px;border-radius:8px;font-weight:650;text-align:left}
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA LAB PRO")
st.write("Laboratório de análise NBA — calendário, seleção de jogos e mercados")
st.caption("As estimativas são demonstrativas: não resultam de um modelo XGBoost treinado nem de calibração validada com odds da Betclic.")
st.markdown("---")

MEDIAS_REAIS = {
    "Atlanta Hawks":116.5,"Boston Celtics":120.8,"Brooklyn Nets":111.2,"Charlotte Hornets":108.5,
    "Chicago Bulls":112.3,"Cleveland Cavaliers":114.8,"Dallas Mavericks":117.9,"Denver Nuggets":115.4,
    "Detroit Pistons":110.1,"Golden State Warriors":118.2,"Houston Rockets":113.5,"Indiana Pacers":121.2,
    "LA Clippers":115.1,"Los Angeles Lakers":116.8,"Memphis Grizzlies":113.8,"Miami Heat":110.5,
    "Milwaukee Bucks":119.4,"Minnesota Timberwolves":113.2,"New Orleans Pelicans":115.0,"New York Knicks":112.8,
    "Oklahoma City Thunder":119.5,"Orlando Magic":110.4,"Philadelphia 76ers":114.2,"Phoenix Suns":116.2,
    "Portland Trail Blazers":106.8,"Sacramento Kings":116.6,"San Antonio Spurs":112.1,"Toronto Raptors":112.5,
    "Utah Jazz":115.7,"Washington Wizards":113.1
}

try:
    API_KEY = st.secrets["API_SPORTS_NBA_KEY"]
except Exception:
    API_KEY = ""

# Layout principal: calendário à esquerda, lista de jogos à direita.
col_cal, col_games = st.columns([0.85, 1.65], gap="large")
with col_cal:
    st.subheader("📅 Escolher dia")
    data = st.date_input("Data dos jogos", value=datetime.date.today(), key="nba_data")
    data_str = data.strftime("%Y-%m-%d")
    st.caption("Escolhe uma data para atualizar a lista de confrontos.")

jogos = []
with col_games:
    st.subheader(f"🏟️ Jogos disponíveis — {data.strftime('%d/%m/%Y')}")
    if not API_KEY:
        st.error("Chave da API-NBA não configurada. Adiciona API_SPORTS_NBA_KEY às Secrets da aplicação Streamlit.")
    else:
        headers = {"x-apisports-key": API_KEY}
        season = str(data.year if data.month >= 10 else data.year - 1)
        params = {"date": data_str, "season": season, "league": "standard"}
        try:
            response = requests.get("https://v2.nba.api-sports.io/games", headers=headers, params=params, timeout=20)
            if response.status_code != 200:
                st.error(f"Erro HTTP {response.status_code}: {response.text[:500]}")
            else:
                payload = response.json()
                if payload.get("errors"):
                    st.error(f"Erro devolvido pela API-NBA: {payload['errors']}")
                else:
                    for item in payload.get("response", []):
                        teams = item.get("teams") or {}
                        home = teams.get("home") or {}
                        away = teams.get("visitors") or teams.get("away") or {}
                        start = (item.get("date") or {}).get("start")
                        home_name, away_name = home.get("name"), away.get("name")
                        if not home_name or not away_name or not start:
                            continue
                        dt_pt = pd.to_datetime(start, utc=True).tz_convert("Europe/Lisbon")
                        jogos.append({"id":str(item.get("id", f"{home_name}-{away_name}-{start}")),
                                      "casa":home_name,"fora":away_name,"hora":dt_pt.strftime("%H:%M"),
                                      "timestamp":dt_pt.timestamp()})
        except requests.RequestException as exc:
            st.error(f"Erro de comunicação com a API-NBA: {exc}")
        except ValueError as exc:
            st.error(f"A resposta da API-NBA não é JSON válido: {exc}")

jogos.sort(key=lambda x: x["timestamp"])
# Remover eventuais duplicados mantendo a ordem cronológica.
seen, jogos_unicos = set(), []
for jogo in jogos:
    chave = (jogo["casa"], jogo["fora"], jogo["hora"])
    if chave not in seen:
        seen.add(chave)
        jogos_unicos.append(jogo)
jogos = jogos_unicos

if jogos:
    ids = {j["id"] for j in jogos}
    if st.session_state.get("nba_jogo_id") not in ids:
        st.session_state["nba_jogo_id"] = jogos[0]["id"]
    with col_games:
        st.markdown("#### Clica num confronto para abrir os mercados")
        button_cols = st.columns(2, gap="medium")
        for i, jogo in enumerate(jogos):
            with button_cols[i % 2]:
                active = st.session_state.get("nba_jogo_id") == jogo["id"]
                label = ("✅ " if active else "🏀 ") + f"{jogo['casa']} vs {jogo['fora']} · {jogo['hora']}"
                if st.button(label, key=f"game_{data_str}_{jogo['id']}", type="primary" if active else "secondary"):
                    st.session_state["nba_jogo_id"] = jogo["id"]
                    st.rerun()
else:
    with col_games:
        st.info("Não foram devolvidos jogos para esta data. Experimenta outra data ou confirma a ligação à API.")

selecionado = next((j for j in jogos if j["id"] == st.session_state.get("nba_jogo_id")), None)
st.markdown("---")
st.subheader("📊 Mercados do confronto selecionado")
if not selecionado:
    st.info("Escolhe um dia com jogos disponíveis. Depois clica num confronto para veres os mercados.")
else:
    casa, fora = selecionado["casa"], selecionado["fora"]
    st.markdown(f"### 🏀 {casa} **vs** {fora} <span class='badge-time'>🕒 {selecionado['hora']} (Portugal)</span>", unsafe_allow_html=True)
    tab_ml, tab_ou = st.tabs(["🏆 MONEYLINE — Vencedor", "🏀 OVER / UNDER — Total de pontos"])

    # Cálculos de demonstração preservados, não são previsões treinadas.
    fator = (len(casa) + len(fora) * 3) % 10
    prob_casa = min(0.90, max(0.10, 0.54 + fator * 0.015))
    prob_fora = 1 - prob_casa
    odd_justa_casa, odd_justa_fora = 1 / prob_casa, 1 / prob_fora

    with tab_ml:
        st.markdown("#### Estimativa de vencedor")
        c1, c2 = st.columns(2)
        with c1:
            st.metric(f"{casa} — Casa", f"{prob_casa*100:.1f}%", f"Odd teórica {odd_justa_casa:.2f}")
        with c2:
            st.metric(f"{fora} — Fora", f"{prob_fora*100:.1f}%", f"Odd teórica {odd_justa_fora:.2f}")
        odd_mercado = st.number_input("Odd de mercado para a vitória da equipa da casa", min_value=1.01, value=1.95, step=0.05, key=f"odd_ml_{selecionado['id']}")
        if odd_mercado > odd_justa_casa:
            vantagem = (odd_mercado / odd_justa_casa - 1) * 100
            st.success(f"A odd de mercado supera a odd teórica demonstrativa em {vantagem:.1f}%. Isto não comprova valor real.")
        else:
            st.warning("A odd de mercado não supera a odd teórica desta estimativa demonstrativa.")

    with tab_ou:
        media_casa = MEDIAS_REAIS.get(casa, 114.5)
        media_fora = MEDIAS_REAIS.get(fora, 112.2)
        total_ref = media_casa + media_fora
        st.metric("Total de pontos de referência", f"{total_ref:.1f}")
        st.caption("Referência calculada a partir de médias fixas por equipa; não é uma previsão calibrada.")
        over_col, under_col = st.columns(2)
        with over_col:
            st.markdown("**OVER — Mais pontos**")
            linha_over = st.number_input("Linha de mercado do OVER", min_value=150.0, max_value=300.0, value=220.5, step=0.5, key=f"over_{selecionado['id']}")
            if linha_over < total_ref:
                st.success("A linha está abaixo da referência demonstrativa.")
            else:
                st.info("A linha não está abaixo da referência demonstrativa.")
        with under_col:
            st.markdown("**UNDER — Menos pontos**")
            linha_under = st.number_input("Linha de mercado do UNDER", min_value=150.0, max_value=300.0, value=220.5, step=0.5, key=f"under_{selecionado['id']}")
            if linha_under > total_ref:
                st.success("A linha está acima da referência demonstrativa.")
            else:
                st.info("A linha não está acima da referência demonstrativa.")

st.caption("Nota: as probabilidades Moneyline são heurísticas e as médias de pontos são fixas. É necessário treinar e validar um modelo com dados históricos antes de as usar para decisões reais.")
