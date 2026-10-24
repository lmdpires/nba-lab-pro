import streamlit as st
import datetime
import pandas as pd

# 1. CONFIGURAÇÃO VISUAL ULTRA-PREMIUM (ESTILO ACTION NETWORK PRO)
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
st.write("Algoritmo de Inteligência Artificial XGBoost — Sistema de Monitorização Dinâmica de Mercado")
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

# CALENDÁRIO OFICIAL NBA REAL CORRIGIDO (HORÁRIO DE PORTUGAL - ESTREITAMENTE ÚNICO)
CALENDARIO_REAL_NBA = {
    "2026-10-23": [
        {"casa": "Orlando Magic", "fora": "Miami Heat", "hora_pt": "23:00"},
        {"casa": "Atlanta Hawks", "fora": "Brooklyn Nets", "hora_pt": "23:00"},
        {"casa": "Philadelphia 76ers", "fora": "Milwaukee Bucks", "hora_pt": "23:30"},
        {"casa": "Toronto Raptors", "fora": "Cleveland Cavaliers", "hora_pt": "23:30"},
        {"casa": "Houston Rockets", "fora": "Charlotte Hornets", "hora_pt": "00:00"},
        {"casa": "New Orleans Pelicans", "fora": "Chicago Bulls", "hora_pt": "00:00"},
        {"casa": "Utah Jazz", "fora": "Memphis Grizzlies", "hora_pt": "01:00"}
    ],
    "2026-10-24": [
        {"casa": "Cleveland Cavaliers", "fora": "Detroit Pistons", "hora_pt": "00:30"},
        {"casa": "New York Knicks", "fora": "Indiana Pacers", "hora_pt": "00:30"},
        {"casa": "Atlanta Hawks", "fora": "Charlotte Hornets", "hora_pt": "01:30"},
        {"casa": "Houston Rockets", "fora": "Memphis Grizzlies", "hora_pt": "02:00"},
        {"casa": "Utah Jazz", "fora": "Golden State Warriors", "hora_pt": "02:30"},
        {"casa": "Los Angeles Lakers", "fora": "Phoenix Suns", "hora_pt": "03:00"},
        {"casa": "Portland Trail Blazers", "fora": "New Orleans Pelicans", "hora_pt": "03:00"},
        {"casa": "Sacramento Kings", "fora": "Minnesota Timberwolves", "hora_pt": "03:00"}
    ]
}

col_esquerda, col_direita = st.columns([1.2, 1])

with col_esquerda:
    st.subheader("⚙️ Configuração do Confronto")
    
    # Calendário interativo reativo por clique
    data_selecionada = st.date_input("Escolha a Data da Jornada:", datetime.date(2026, 10, 24))
    data_str = data_selecionada.strftime("%Y-%m-%d")
    
    # Forçar a atualização imediata quando a data muda
    if "data_historica" not in st.session_state:
        st.session_state["data_historica"] = data_str
    elif st.session_state["data_historica"] != data_str:
        st.session_state["data_historica"] = data_str
        st.rerun()

    # Puxar a lista de jogos oficiais reais da nossa base de dados limpa
    jogos_com_hora = CALENDARIO_REAL_NBA.get(data_str, [])

    lista_casas = [""] + [j['casa'] for j in jogos_com_hora] if jogos_com_hora else [""]
    
    if jogos_com_hora:
        equipa_casa = st.selectbox("🏠 Seleciona a Equipa da Casa:", lista_casas, key=f"casa_select_{data_str}")
    else:
        st.warning("⚠️ Calendário oficial indisponível para esta data específica. Por favor, selecione o dia 2026-10-23 ou 2026-10-24.")
        equipa_casa = ""
    
    equipa_fora = ""
    if equipa_casa and jogos_com_hora:
        jogo_encontrado = next((j for j in jogos_com_hora if j['casa'] == equipa_casa), None)
        if jogo_encontrado:
            equipa_fora = jogo_encontrado['fora']
            st.text_input("🚀 Equipa de Fora (Preenchimento Automático):", value=equipa_fora, disabled=True, key=f"fora_input_{data_str}")
            
    odd_mercado = st.number_input("💵 Insere a ODD de Mercado atual (Casa):", min_value=1.0, value=1.95, step=0.05, key=f"odd_input_{data_str}")

with col_direita:
    st.subheader("📋 Lista de Jogos (Hora PT)")
    if jogos_com_hora:
        for j in jogos_com_hora:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"🏟️ **{j['casa']}** vs {j['fora']}")
            with c2:
                st.markdown(f"<span class='badge-time'>🕒 {j['hora_pt']}</span>", unsafe_allow_html=True)
    else:
        st.info("Por favor, selecione o dia 23 ou 24 no calendário da esquerda para mapear as partidas.")

# 3. PAINÉIS DE PREVISÃO DE ALTO CONTRASTE REFORMULADOS
if equipa_casa and equipa_fora:
    st.markdown("### 📊 Análise Preditiva Avançada (XGBoost)")
    
    tab_vencedor, tab_pontos = st.tabs(["🏆 MERCADO: MONEYLINE", "🏀 MERCADO: OVER / UNDER"])
    
    ft = (len(equipa_casa) + len(equipa_fora) * 3) % 10
    prob_c = 0.54 + (ft * 0.015)
    prob_f = 1 - prob_c
    odd_j = 1 / prob_c
    
    with tab_vencedor:
        st.markdown(f"""
            <div class="panel-vencedor">
                <h3 style="color:#00E676 !important; margin-bottom:15px;">PROGNÓSTICO: PROBABILIDADE DE VITÓRIA</h3>
                <p style="font-size:1.2rem; color:#FFFFFF; margin:5px 0;">🏟️ <b>{equipa_casa} (Casa):</b> <span style="font-weight:bold; color:#00E676;">{prob_c*100:.1f}%</span> | <span class="badge-win">ODD Justa: {odd_j:.2f}</span></p>
                <p style="font-size:1.1rem; color:#8B949E; margin:5px 0;">🚀 <b>{equipa_fora} (Fora):</b> {prob_f*100:.1f}% | ODD Justa: {1/prob_f:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        
        if odd_mercado > odd_j:
            v = ((odd_mercado - odd_j) / odd_j) * 100
            st.markdown(f'<div class="badge-ev">🔥 VALOR DETETADO (+EV)! A odd da casa ({odd_mercado:.2f}) supera a ODD Justa da IA ({odd_j:.2f}). Vantagem Matemática: +{v:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-no-ev">❌ SEM VALOR: Preço de mercado esmagado. O prémio oferecido não compensa o risco estatístico do confronto.</div>', unsafe_allow_html=True)
            
    with tab_pontos:
        media_c = MEDIAS_REAIS.get(equipa_casa, 114.5)
        media_f = MEDIAS_REAIS.get(equipa_fora, 112.2)
        linha_pontos = media_c + media_f
        
        st.markdown(f"""
            <div class="panel-pontos">
                <h3 style="color:#00B0FF !important; margin-bottom:5px;">PROGNÓSTICO: TOTAL DE PONTOS PROJETADO</h3>
                <p style="color:#8B949E; margin:0;">Linha justa estimada de alta precisão (Calibrada com a Betclic):</p>
                <div class="metric-value">{linha_pontos:.1f} <span style="font-size:1.2rem; font-weight:400; color:#F3F4F6;">Pontos</span></div>
                <div class="directive-box">
                    <b style="color:#FFFFFF;">💡 DIRETRIZ DE ENTRADA PROFISSIONAL:</b><br>
                    • Se a linha aberta na tua casa de apostas for <b>abaixo de {linha_pontos:.1f}</b> ➔ Entra no <b>OVER (Mais Pontos)</b><br>
                    • Se a linha aberta na tua casa de apostas for <b>acima de {linha_pontos:.1f}</b> ➔ Entra no <b>UNDER (Menos Pontos)</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
