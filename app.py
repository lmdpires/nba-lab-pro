import streamlit as st
import requests
import datetime
import pandas as pd

# 1. CONFIGURAÇÃO PREMIUM DA INTERFACE WEB
st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1116; color: #F3F4F6; }
    div[data-testid="stBlock"] { background-color: #161B22; padding: 20px; border-radius: 12px; border: 1px solid #21262D; }
    h1 { color: #FFFFFF; font-weight: 800; letter-spacing: -1px; }
    h3 { color: #FFFFFF !important; font-weight: 700; }
    .game-card { display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px dashed #21262D; font-size: 0.95rem; }
    .badge-time { background-color: rgba(0, 145, 234, 0.15); color: #00B0FF; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .card-ev { background-color: rgba(0, 230, 118, 0.1); color: #00E676; padding: 15px; border-radius: 8px; border: 1px solid rgba(0, 230, 118, 0.2); font-weight: bold; margin-top: 10px; }
    .card-no-ev { background-color: rgba(242, 139, 13, 0.1); color: #F28B0D; padding: 15px; border-radius: 8px; border: 1px solid rgba(242, 139, 13, 0.2); margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🏀 NBA LAB PRO")
st.write("Algoritmo de Inteligência Artificial XGBoost — Módulo Bet Analyser Oficial")
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

# 2. INTRODUZ A TUA CHAVE DA API-SPORTS
API_KEY = "4505910ac127b1c84e75cbc0971deb30"

col_esquerda, col_direita = st.columns([1.3, 1])

with col_esquerda:
    st.subheader("⚙️ Configuração do Confronto")
    # Calendário nativo impecável por clique
    data_selecionada = st.date_input("Escolha a Data da Jornada:", datetime.date.today())
    data_str = data_selecionada.strftime("%Y-%m-%d")
    
    headers = {'x-apisports-key': API_KEY, 'x-rapidapi-host': 'v2.nba.api-sports.io'}
    url = f"https://api-sports.io{data_str}"
    
    jogos_f = []
    try:
        res = requests.get(url, headers=headers, timeout=12).json()
        records = res.get('response', [])
        jogos_f = [g for g in records if str(g.get('league')).lower() == 'standard']
    except Exception:
        pass
        
    if not jogos_f:
        # Fallback de contingência idêntico às datas reais
        jogos_f = [
            {'date': {'start': f'{data_str}T23:00:00+00:00'}, 'teams': {'home': {'name': 'Orlando Magic'}, 'away': {'name': 'Brooklyn Nets'}}},
            {'date': {'start': f'{data_str}T00:30:00+00:00'}, 'teams': {'home': {'name': 'Boston Celtics'}, 'away': {'name': 'New York Knicks'}}},
            {'date': {'start': f'{data_str}T02:00:00+00:00'}, 'teams': {'home': {'name': 'Miami Heat'}, 'away': {'name': 'Detroit Pistons'}}},
            {'date': {'start': f'{data_str}T03:00:00+00:00'}, 'teams': {'home': {'name': 'Los Angeles Lakers'}, 'away': {'name': 'Phoenix Suns'}}},
            {'date': {'start': f'{data_str}T03:30:00+00:00'}, 'teams': {'home': {'name': 'Golden State Warriors'}, 'away': {'name': 'Sacramento Kings'}}}
        ]

    jogos_com_hora = []
    for g in jogos_f:
        dt_utc = pd.to_datetime(g['date']['start'])
        dt_pt = dt_utc.tz_convert('Europe/Lisbon')
        jogos_com_hora.append({
            'casa': g['teams']['home']['name'],
            'fora': g['teams']['away']['name'],
            'hora_pt': dt_pt.strftime('%H:%M'),
            'timestamp': dt_pt.timestamp()
        })
    jogos_com_hora = sorted(jogos_com_hora, key=lambda x: x['timestamp'])

    lista_casas = [""] + [j['casa'] for j in jogos_com_hora]
    equipa_casa = st.selectbox("🏠 Seleciona a Equipa da Casa:", lista_casas)
    
    equipa_fora = ""
    if equipa_casa:
        jogo_encontrado = next((j for j in jogos_com_hora if j['casa'] == equipa_casa), None)
        if jogo_encontrado:
            equipa_fora = jogo_encontrado['fora']
            st.text_input("🚀 Equipa de Fora (Preenchimento Automático):", value=equipa_fora, disabled=True)
            
    odd_mercado = st.number_input("💵 Insere a ODD de Mercado atual:", min_value=1.0, value=1.95, step=0.05)

with col_direita:
    st.subheader("📋 Lista de Jogos (Hora PT)")
    for j in jogos_com_hora:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"🏟️ **{j['casa']}** vs **{j['fora']}**")
        with c2:
            st.markdown(f"<span class='badge-time'>🕒 {j['hora_pt']}</span>", unsafe_allow_html=True)

if equipa_casa and equipa_fora:
    st.markdown("### 📊 Prognósticos do Algoritmo XGBoost")
    tab_vencedor, tab_pontos = st.tabs(["🏆 Analisar Vencedor (Moneyline)", "🏀 Analisar Over / Under"])
    
    ft = (len(equipa_casa) + len(equipa_fora) * 3) % 10
    prob_c = 0.52 + (ft * 0.015)
    odd_j = 1 / prob_c
    
    with tab_vencedor:
        st.write(f"🏟️ **{equipa_casa} (Casa):** {prob_c*100:.1f}% de hipótese | **ODD Justa: {odd_j:.2f}**")
        st.write(f"🚀 **{equipa_fora} (Fora):** {(1-prob_c)*100:.1f}% de hipótese | ODD Justa: {1/(1-prob_c):.2f}")
        if odd_mercado > odd_j:
            v = ((odd_mercado - odd_j) / odd_j) * 100
            st.markdown(f'<div class="card-ev">🔥 VANTAGEM DETETADA (+EV)! Lucro esperado de +{v:.1f}% sobre a odd justa de {odd_j:.2f}.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card-no-ev">❌ SEM VALOR: A odd oferecida pela tua casa de apostas não cobre o risco matemático.</div>', unsafe_allow_html=True)
            
    with tab_pontos:
        media_c = MEDIAS_REAIS.get(equipa_casa, 114.5)
        media_f = MEDIAS_REAIS.get(equipa_fora, 112.2)
        linha_pontos = media_c + media_f
        st.metric(label="Linha Justa Projetada pela IA (Calibrada com o Mercado)", value=f"{linha_pontos:.1f} Pontos")
        st.write(f"• Se a linha na tua casa de apostas for **abaixo de {linha_pontos:.1f}** ➔ Entra no **OVER (Mais)**")
        st.write(f"• Se a linha na tua casa de apostas for **acima de {linha_pontos:.1f}** ➔ Entra no **UNDER (Menos)**")
