import datetime
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1116; color: #F3F4F6; font-family: 'Helvetica Neue', sans-serif; }
    h1 { color: #FFFFFF; font-weight: 800; letter-spacing: -1px; margin-bottom: 5px; }
    h3 { color: #FFFFFF !important; font-weight: 700; margin-top: 0; letter-spacing: -0.5px; }
    
    /* Content Blocks */
    div[data-testid="stBlock"] { background-color: #161B22; padding: 25px; border-radius: 14px; border: 1px solid #21262D; }
    
    /* Lista de Jogos Cronológica Completa */
    .game-card { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px dashed #30363D; font-size: 0.95rem; color: #FFFFFF; }
    .badge-time { background-color: rgba(0, 145, 234, 0.15); color: #00B0FF; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem; border: 1px solid rgba(0, 145, 234, 0.2); }
    
    /* Painéis de Prognósticos em Alto Contraste */
    .panel-vencedor { background: linear-gradient(135deg, #1A2E1A 0%, #111827 100%); padding: 25px; border-radius: 12px; border: 1px solid #2e7d32; margin-top: 15px; }
    .panel-pontos { background: linear-gradient(135deg, #1A2A3A 0%, #111827 100%); padding: 25px; border-radius: 12px; border: 1px solid #0074D9; margin-top: 15px; }
    
    .badge-win { background-color: #00E676; color: #0E1116; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }
    .badge-ev { background-color: rgba(0, 230, 118, 0.15); color: #00E676; padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(0, 230, 118, 0.3); font-weight: bold; font-size: 1.05rem; margin-top: 15px; }
    .badge-no-ev { background-color: rgba(242, 139, 13, 0.1); color: #F28B0D; padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(242, 139, 13, 0.2); font-size: 0.95rem; margin-top: 15px; }
    
    .metric-value { font-size: 2.8rem; font-weight: 900; color: #00B0FF; margin: 5px 0; letter-spacing: -1px; }
    .directive-box { background-color: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; border: 1px solid #21262D; margin-top: 15px; color: #C9D1D9; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

st.title("🏀 NBA LAB PRO")
st.write("Laboratório de análise NBA — calendário e estimativas demonstrativas")
st.caption("Nota: as estimativas atuais abaixo são heurísticas de demonstração; não resultam de um modelo XGBoost treinado nem de calibração validada com odds da Betclic.")
st.markdown("---")

MEDIAS_REAIS = {
    "Atlanta Hawks": 116.5, "Boston Celtics": 120.8, "Brooklyn Nets": 111.2, "Charlotte Hornets": 108.5,
    "Chicago Bulls": 112.3, "Cleveland Cavaliers": 114.8, "Dallas Mavericks": 117.9, "Denver Nuggets": 115.4,
    "Detroit Pistons": 110.1, "Golden State Warriors": 118.2, "Houston Rockets": 113.5, "Indiana Pacers": 121.2,
    "LA Clippers": 115.1, "Los Angeles Lakers": 116.8, "Memphis Grizzlies": 113.8, "Miami Heat": 110.5,
    "Milwaukee Bucks": 119.4, "Minnesota Timberwolves": 113.2, "New Orleans Pelicans": 115.0, "New York Knicks": 112.8,
    "Oklahoma City Thunder": 119.5, "Orlando Magic": 110.4, "Philadelphia 76ers": 114.2, "Phoenix Suns": 116.2,
    "Portland Trail Blazers": 106.8, "Sacramento Kings": 116.6, "San Antonio Spurs": 112.1, "Toronto Raptors": 112.5,
    "Utah Jazz": 115.7, "Washington Wizards": 113.1
}

# Configure a chave em .streamlit/secrets.toml:
# API_SPORTS_NBA_KEY = "a_tua_chave"
try:
    API_KEY = st.secrets["API_SPORTS_NBA_KEY"]
except Exception:
    API_KEY = ""

col_esquerda, col_direita = st.columns([1.2, 1])
jogos_com_hora = []
equipa_casa = ""
equipa_fora = ""
odd_mercado = 1.95

with col_esquerda:
    st.subheader("⚙️ Configuração do Confronto")
    data_selecionada = st.date_input("Escolha a Data da Jornada:", datetime.date.today())
    data_str = data_selecionada.strftime("%Y-%m-%d")

    if not API_KEY:
        st.error("Chave da API-NBA não configurada. Adiciona API_SPORTS_NBA_KEY ao ficheiro .streamlit/secrets.toml.")
    else:
        headers = {"x-apisports-key": API_KEY}
        url = "https://v2.nba.api-sports.io/games"
        parametros_busca = {
            "date": data_str,
            "season": str(data_selecionada.year if data_selecionada.month >= 10 else data_selecionada.year - 1),
            "league": "standard",
        }

        try:
            response = requests.get(url, headers=headers, params=parametros_busca, timeout=20)
            if response.status_code != 200:
                st.error(f"Erro HTTP {response.status_code}: {response.text[:1000]}")
            else:
                res_json = response.json()
                erros = res_json.get("errors", {})
                if erros:
                    st.error(f"Erro devolvido pela API-NBA: {erros}")
                else:
                    records = res_json.get("response", [])
                    for jogo in records:
                        try:
                            equipas = jogo.get("teams") or {}
                            casa = equipas.get("home") or {}
                            fora = equipas.get("visitors") or equipas.get("away") or {}
                            nome_casa = casa.get("name")
                            nome_fora = fora.get("name")
                            data_inicio = (jogo.get("date") or {}).get("start")
                            if not nome_casa or not nome_fora or not data_inicio:
                                continue

                            dt_pt = pd.to_datetime(data_inicio, utc=True).tz_convert("Europe/Lisbon")
                            jogos_com_hora.append({
                                "casa": nome_casa,
                                "fora": nome_fora,
                                "hora_pt": dt_pt.strftime("%H:%M"),
                                "timestamp": dt_pt.timestamp(),
                            })
                        except (KeyError, TypeError, ValueError) as exc:
                            st.warning(f"Foi ignorado um registo com dados incompletos: {exc}")

                    if not jogos_com_hora:
                        st.info(
                            f"A API respondeu corretamente, mas não devolveu jogos para {data_str}. "
                            f"Parâmetros usados: {res_json.get('parameters', {})}"
                        )
        except requests.RequestException as exc:
            st.error(f"Erro de comunicação com a API-NBA: {exc}")
        except ValueError as exc:
            st.error(f"A resposta da API-NBA não é JSON válido: {exc}")

    if jogos_com_hora:
        df_clean = pd.DataFrame(jogos_com_hora)
        df_clean = df_clean.drop_duplicates(subset=["casa", "fora"])
        jogos_com_hora = df_clean.sort_values(by="timestamp").to_dict("records")

    lista_casas = [""] + [j["casa"] for j in jogos_com_hora]
    if jogos_com_hora:
        equipa_casa = st.selectbox(
            "🏠 Seleciona a Equipa da Casa:",
            lista_casas,
            key=f"casa_select_{data_str}",
        )
    else:
        st.warning(f"⚠️ A aguardar registos de jogos na API-Sports para o dia {data_str}.")

    if equipa_casa:
        jogo_encontrado = next((j for j in jogos_com_hora if j["casa"] == equipa_casa), None)
        if jogo_encontrado:
            equipa_fora = jogo_encontrado["fora"]
            st.text_input(
                "🚀 Equipa de Fora (Preenchimento Automático):",
                value=equipa_fora,
                disabled=True,
                key=f"fora_input_{data_str}",
            )

    odd_mercado = st.number_input(
        "💵 Insere a ODD de Mercado atual (Casa):",
        min_value=1.0,
        value=1.95,
        step=0.05,
        key=f"odd_input_{data_str}",
    )

with col_direita:
    st.subheader("📋 Lista de Jogos (Hora PT)")
    if jogos_com_hora:
        for jogo in jogos_com_hora:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"🏟️ **{jogo['casa']}** vs **{jogo['fora']}**")
            with c2:
                st.markdown(
                    f"<span class='badge-time'>🕒 {jogo['hora_pt']}</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Nenhum confronto devolvido pela API para a data selecionada.")

if equipa_casa and equipa_fora:
    st.markdown("### 📊 Estimativas demonstrativas")
    tab_vencedor, tab_pontos = st.tabs(["🏆 MERCADO: MONEYLINE", "🏀 MERCADO: OVER / UNDER"])

    # Placeholder heurístico: substituir por um modelo treinado e validado antes de usar em apostas reais.
    fator = (len(equipa_casa) + len(equipa_fora) * 3) % 10
    prob_c = min(0.90, max(0.10, 0.54 + (fator * 0.015)))
    prob_f = 1 - prob_c
    odd_justa_casa = 1 / prob_c
    odd_justa_fora = 1 / prob_f

    with tab_vencedor:
        st.markdown(f"""
            <div class="panel-vencedor">
                <h3 style="color:#00E676 !important; margin-bottom:15px;">ESTIMATIVA DE PROBABILIDADE DE VITÓRIA</h3>
                <p style="font-size:1.2rem; color:#FFFFFF; margin:5px 0;">🏟️ <b>{equipa_casa} (Casa):</b> <span style="font-weight:bold; color:#00E676;">{prob_c*100:.1f}%</span> | <span class="badge-win">ODD teórica: {odd_justa_casa:.2f}</span></p>
                <p style="font-size:1.1rem; color:#8B949E; margin:5px 0;">🚀 <b>{equipa_fora} (Fora):</b> {prob_f*100:.1f}% | ODD teórica: {odd_justa_fora:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

        if odd_mercado > odd_justa_casa:
            vantagem = ((odd_mercado / odd_justa_casa) - 1) * 100
            st.markdown(
                f'<div class="badge-ev">A odd de mercado supera a odd teórica em {vantagem:.1f}%. Esta indicação é apenas demonstrativa e não comprova valor real.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="badge-no-ev">A odd de mercado não supera a odd teórica desta estimativa demonstrativa.</div>',
                unsafe_allow_html=True,
            )

    with tab_pontos:
        media_c = MEDIAS_REAIS.get(equipa_casa, 114.5)
        media_f = MEDIAS_REAIS.get(equipa_fora, 112.2)
        linha_pontos = media_c + media_f
        st.markdown(f"""
            <div class="panel-pontos">
                <h3 style="color:#00B0FF !important; margin-bottom:5px;">TOTAL DE PONTOS — ESTIMATIVA SIMPLES</h3>
                <p style="color:#8B949E; margin:0;">Soma de médias fixas por equipa; não é uma previsão calibrada.</p>
                <div class="metric-value">{linha_pontos:.1f} <span style="font-size:1.2rem; font-weight:400; color:#F3F4F6;">Pontos</span></div>
                <div class="directive-box">
                    <b style="color:#FFFFFF;">Nota:</b><br>
                    Compara esta referência com a linha de mercado, mas não a uses isoladamente para decidir uma aposta.
                </div>
            </div>
        """, unsafe_allow_html=True)
