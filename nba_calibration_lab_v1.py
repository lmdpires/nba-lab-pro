import os
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='NBA Calibration Lab', page_icon='🏀', layout='wide')
st.title('🏀 NBA LAB PRO — Calibration Lab v1')
st.caption('Laboratório separado da app principal. Recolha histórica, features sem leakage e backtest cronológico.')

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'nba_lab_data'
DATA_DIR.mkdir(exist_ok=True)
GAMES_PATH = DATA_DIR / 'nba_games_history.csv'
FEATURES_PATH = DATA_DIR / 'nba_features_history.csv'
BASE_URL = 'https://v2.nba.api-sports.io/games'

try:
    API_KEY = st.secrets['API_SPORTS_NBA_KEY']
except Exception:
    API_KEY = os.environ.get('API_SPORTS_NBA_KEY', '')

SEASON_OPTIONS = ['2021', '2022', '2023', '2024', '2025']

def score_value(obj):
    if isinstance(obj, dict):
        for key in ('points', 'total'):
            value = obj.get(key)
            if value is not None:
                try: return float(value)
                except (TypeError, ValueError): pass
    if obj is not None:
        try: return float(obj)
        except (TypeError, ValueError): pass
    return np.nan

def fetch_season(season, api_key):
    response = requests.get(BASE_URL, headers={'x-apisports-key': api_key}, params={'league': 'standard', 'season': season}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if payload.get('errors'):
        raise RuntimeError(str(payload['errors']))
    rows = []
    for item in payload.get('response', []):
        teams = item.get('teams') or {}
        home = teams.get('home') or {}
        away = teams.get('visitors') or teams.get('away') or {}
        scores = item.get('scores') or {}
        home_score = score_value(scores.get('home'))
        away_score = score_value(scores.get('visitors') or scores.get('away'))
        date_obj = item.get('date') or {}
        status_obj = item.get('status') or {}
        start = date_obj.get('start')
        if not start or not home.get('id') or not away.get('id'):
            continue
        rows.append({
            'game_id': str(item.get('id', '')), 'season': str(item.get('season', season)),
            'date': pd.to_datetime(start, utc=True, errors='coerce'),
            'status': str(status_obj.get('short') or status_obj.get('long') or ''),
            'home_id': str(home.get('id')), 'home_team': str(home.get('name') or home.get('nickname') or home.get('id')),
            'away_id': str(away.get('id')), 'away_team': str(away.get('name') or away.get('nickname') or away.get('id')),
            'home_score': home_score, 'away_score': away_score
        })
    return rows, response.headers.get('x-ratelimit-requests-remaining', 'n/d')

def build_features(games):
    df = games.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df.dropna(subset=['date', 'home_score', 'away_score'])
    df = df[(df['home_score'] >= 50) & (df['away_score'] >= 50) & (df['home_score'] <= 200) & (df['away_score'] <= 200)]
    df = df.sort_values(['date', 'game_id']).reset_index(drop=True)
    history = {}
    records = []
    for _, g in df.iterrows():
        hid, aid = str(g.home_id), str(g.away_id)
        hh, ah = history.get(hid, []), history.get(aid, [])
        def stats(hist, n):
            recent = hist[-n:]
            if not recent: return {'pf': 110.0, 'pa': 110.0, 'win': .5, 'n': 0}
            return {'pf': float(np.mean([x['pf'] for x in recent])), 'pa': float(np.mean([x['pa'] for x in recent])), 'win': float(np.mean([x['win'] for x in recent])), 'n': len(recent)}
        h5, a5, h10, a10 = stats(hh, 5), stats(ah, 5), stats(hh, 10), stats(ah, 10)
        last_home = hh[-1]['date'] if hh else None
        last_away = ah[-1]['date'] if ah else None
        date = g.date
        rest_h = min(10.0, (date-last_home).total_seconds()/86400) if last_home is not None else 3.0
        rest_a = min(10.0, (date-last_away).total_seconds()/86400) if last_away is not None else 3.0
        records.append({
            'game_id': g.game_id, 'date': date, 'season': g.season, 'home_team': g.home_team, 'away_team': g.away_team,
            'home_score': float(g.home_score), 'away_score': float(g.away_score),
            'home_win': int(g.home_score > g.away_score), 'total_points': float(g.home_score + g.away_score),
            'home_pf5': h5['pf'], 'home_pa5': h5['pa'], 'away_pf5': a5['pf'], 'away_pa5': a5['pa'],
            'home_win5': h5['win'], 'away_win5': a5['win'], 'home_pf10': h10['pf'], 'home_pa10': h10['pa'],
            'away_pf10': a10['pf'], 'away_pa10': a10['pa'], 'home_win10': h10['win'], 'away_win10': a10['win'],
            'home_games_prior': h10['n'], 'away_games_prior': a10['n'], 'rest_home_days': rest_h, 'rest_away_days': rest_a,
            'form_diff5': (h5['pf']-h5['pa'])-(a5['pf']-a5['pa']),
            'form_diff10': (h10['pf']-h10['pa'])-(a10['pf']-a10['pa']),
            'points_for_diff5': h5['pf']-a5['pf'], 'defence_diff5': a5['pa']-h5['pa'],
        })
        # Update history only after the pre-game features are frozen.
        history.setdefault(hid, []).append({'date': date, 'pf': float(g.home_score), 'pa': float(g.away_score), 'win': int(g.home_score > g.away_score)})
        history.setdefault(aid, []).append({'date': date, 'pf': float(g.away_score), 'pa': float(g.home_score), 'win': int(g.away_score > g.home_score)})
    return pd.DataFrame(records)

st.subheader('1. Recolher resultados históricos')
st.write('A API devolve os jogos por época. O primeiro passo usa apenas data, equipas, estado e resultado final; depois criamos estatísticas móveis que só usam jogos anteriores.')
col1, col2 = st.columns([2, 1])
with col1:
    seasons = st.multiselect('Épocas (ano de início)', SEASON_OPTIONS, default=['2021','2022','2023','2024','2025'])
with col2:
    st.metric('Épocas selecionadas', len(seasons))

if not API_KEY:
    st.warning('Não encontrei API_SPORTS_NBA_KEY. Configura-a em Streamlit Secrets (ou variável de ambiente) antes de recolher dados. Não coloques a chave no código.')

if st.button('Descarregar / atualizar épocas selecionadas', type='primary', disabled=(not API_KEY or not seasons)):
    existing = pd.read_csv(GAMES_PATH, dtype={'game_id': str}) if GAMES_PATH.exists() else pd.DataFrame()
    fetched = []
    progress = st.progress(0)
    status = st.empty()
    for i, season in enumerate(seasons, 1):
        try:
            status.info(f'A recolher época {season}…')
            rows, remaining = fetch_season(season, API_KEY)
            fetched.extend(rows)
            status.write(f'Época {season}: {len(rows)} jogos recebidos · pedidos restantes reportados pela API: {remaining}')
        except Exception as exc:
            st.error(f'Falha na época {season}: {exc}')
        progress.progress(i/len(seasons))
    if fetched:
        fresh = pd.DataFrame(fetched)
        if not existing.empty:
            merged = pd.concat([existing, fresh], ignore_index=True)
        else:
            merged = fresh
        merged = merged.drop_duplicates(subset=['game_id'], keep='last').sort_values('date')
        merged.to_csv(GAMES_PATH, index=False)
        st.success(f'Base guardada: {len(merged):,} jogos únicos. Ficheiro: {GAMES_PATH.name}')

if GAMES_PATH.exists():
    raw = pd.read_csv(GAMES_PATH, parse_dates=['date'], dtype={'game_id': str})
    finished = raw.dropna(subset=['home_score','away_score']).copy()
    st.subheader('Estado da base')
    m1,m2,m3 = st.columns(3)
    m1.metric('Jogos com resultado', f'{len(finished):,}')
    m2.metric('Épocas na base', finished['season'].nunique())
    m3.metric('Equipas casa distintas', finished['home_id'].nunique())
    st.download_button('Descarregar jogos históricos CSV', raw.to_csv(index=False).encode('utf-8-sig'), 'nba_games_history.csv', 'text/csv')

    if st.button('Construir features pré-jogo e executar backtest'):
        feat = build_features(finished)
        feat.to_csv(FEATURES_PATH, index=False)
        st.success(f'Features guardadas: {len(feat):,} jogos. A seguir começa o backtest cronológico.')
        st.session_state['features_ready'] = True

if FEATURES_PATH.exists():
    feat = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
    st.subheader('2. Backtest cronológico')
    st.caption('Treina nos primeiros 70% cronológicos e testa nos 30% finais. O teste não participa no treino nem na normalização.')
    min_prior = st.slider('Mínimo de jogos anteriores por equipa', 0, 10, 3)
    if st.button('Executar backtest fora da amostra'):
        feat = feat.sort_values('date').reset_index(drop=True)
        feat = feat[(feat.home_games_prior >= min_prior) & (feat.away_games_prior >= min_prior)].copy()
        if len(feat) < 200:
            st.error(f'Amostra insuficiente para um backtest inicial: {len(feat)} jogos após filtros.')
        else:
            split = int(len(feat)*0.70)
            train, test = feat.iloc[:split].copy(), feat.iloc[split:].copy()
            features = ['home_pf5','home_pa5','away_pf5','away_pa5','home_win5','away_win5','home_pf10','home_pa10','away_pf10','away_pa10','home_win10','away_win10','rest_home_days','rest_away_days','form_diff5','form_diff10','points_for_diff5','defence_diff5']
            Xtr, Xte = train[features].fillna(0), test[features].fillna(0)
            ytr, yte = train['home_win'].astype(int), test['home_win'].astype(int)
            cls = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            cls.fit(Xtr, ytr)
            p = cls.predict_proba(Xte)[:,1]
            pred = (p >= .5).astype(int)
            brier = brier_score_loss(yte, p)
            ll = log_loss(yte, np.clip(p, 1e-6, 1-1e-6), labels=[0,1])
            acc = accuracy_score(yte, pred)
            train_majority = max(float(ytr.mean()), 1.0-float(ytr.mean()))
            # Point total model: trained only on the training period.
            reg_features = ['home_pf5','home_pa5','away_pf5','away_pa5','home_pf10','home_pa10','away_pf10','away_pa10','rest_home_days','rest_away_days','form_diff5','form_diff10','points_for_diff5','defence_diff5']
            reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            reg.fit(train[reg_features].fillna(0), train['total_points'])
            total_pred = reg.predict(test[reg_features].fillna(0))
            mae = mean_absolute_error(test['total_points'], total_pred)
            rmse = mean_squared_error(test['total_points'], total_pred)**0.5
            # Simple baselines are essential: compare against 50% Moneyline and historical mean total.
            base_total = np.repeat(train['total_points'].mean(), len(test))
            base_mae = mean_absolute_error(test['total_points'], base_total)
            st.markdown('### Resultados fora da amostra')
            a,b,c,d,e = st.columns(5)
            a.metric('Jogos teste', f'{len(test):,}')
            b.metric('Moneyline accuracy', f'{acc:.1%}', f'baseline treino: {train_majority:.1%} maioria')
            c.metric('Brier Score', f'{brier:.4f}', 'menor é melhor')
            d.metric('Log Loss', f'{ll:.4f}', 'menor é melhor')
            e.metric('Total MAE', f'{mae:.2f} pts', f'baseline média: {base_mae:.2f}')
            st.write(f'RMSE do total: **{rmse:.2f} pontos**. O modelo é promissor apenas se superar consistentemente as referências simples em vários períodos.')
            st.caption(f'Período de treino: {train.date.min().date()} a {train.date.max().date()} · teste: {test.date.min().date()} a {test.date.max().date()}')
            results = test[['game_id','date','home_team','away_team','home_score','away_score','home_win','total_points']].copy()
            results['p_home_win'] = p
            results['predicted_total'] = total_pred
            results['moneyline_correct'] = (pred == yte.to_numpy())
            results['total_abs_error'] = np.abs(total_pred-test['total_points'].to_numpy())
            results.to_csv(DATA_DIR/'nba_backtest_predictions.csv', index=False)
            st.dataframe(results.head(30), use_container_width=True, hide_index=True)
            st.download_button('Descarregar previsões do backtest CSV', results.to_csv(index=False).encode('utf-8-sig'), 'nba_backtest_predictions.csv', 'text/csv')
            st.warning('Este é um primeiro benchmark, não uma calibração final. Ainda falta validar várias janelas temporais, verificar cobertura dos dados e incorporar odds históricas/informação de lesões se estiverem disponíveis.')

st.divider()
st.markdown('**Fontes de dados:** API-Sports NBA. A disponibilidade de estatísticas e épocas depende da cobertura da competição e do plano da API. O modelo não deve ser usado para apostas reais até os testes serem repetidos e auditados.')
