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

st.set_page_config(page_title='NBA OVER LAB V3', page_icon='🏀', layout='wide')
st.title('🏀 NBA LAB PRO — NBA OVER LAB V3')
st.caption('Laboratório dedicado à previsão de pontos totais e à validação cronológica de mercados Over/Under. Aplicação separada da app principal e das Calibration Labs V1/V2.')

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
    """Create pre-game features. Games with identical timestamps are treated as simultaneous:
    none of their results can affect another game's features at that same timestamp.
    """
    df = games.copy()
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    df = df.dropna(subset=['date', 'home_score', 'away_score'])
    df = df[(df['home_score'] >= 50) & (df['away_score'] >= 50) & (df['home_score'] <= 200) & (df['away_score'] <= 200)]
    df = df.sort_values(['date', 'game_id']).reset_index(drop=True)
    history = {}
    records = []

    def stats(hist, n):
        recent = hist[-n:]
        if not recent:
            return {'pf': 110.0, 'pa': 110.0, 'win': .5, 'n': 0}
        return {
            'pf': float(np.mean([x['pf'] for x in recent])),
            'pa': float(np.mean([x['pa'] for x in recent])),
            'win': float(np.mean([x['win'] for x in recent])),
            'n': len(recent),
        }

    # Process identical start times as a batch to prevent same-time result leakage.
    for date, batch in df.groupby('date', sort=True):
        pending_updates = []
        for _, g in batch.iterrows():
            hid, aid = str(g.home_id), str(g.away_id)
            hh, ah = history.get(hid, []), history.get(aid, [])
            h5, a5, h10, a10 = stats(hh, 5), stats(ah, 5), stats(hh, 10), stats(ah, 10)
            last_home = hh[-1]['date'] if hh else None
            last_away = ah[-1]['date'] if ah else None
            rest_h = min(10.0, (date-last_home).total_seconds()/86400) if last_home is not None else 3.0
            rest_a = min(10.0, (date-last_away).total_seconds()/86400) if last_away is not None else 3.0
            home_win = int(g.home_score > g.away_score)
            records.append({
                'game_id': str(g.game_id), 'date': date, 'season': str(g.season), 'home_team': g.home_team, 'away_team': g.away_team,
                'home_score': float(g.home_score), 'away_score': float(g.away_score),
                'home_win': home_win, 'total_points': float(g.home_score + g.away_score),
                'home_pf5': h5['pf'], 'home_pa5': h5['pa'], 'away_pf5': a5['pf'], 'away_pa5': a5['pa'],
                'home_win5': h5['win'], 'away_win5': a5['win'], 'home_pf10': h10['pf'], 'home_pa10': h10['pa'],
                'away_pf10': a10['pf'], 'away_pa10': a10['pa'], 'home_win10': h10['win'], 'away_win10': a10['win'],
                'home_games_prior': h10['n'], 'away_games_prior': a10['n'], 'rest_home_days': rest_h, 'rest_away_days': rest_a,
                'form_diff5': (h5['pf']-h5['pa'])-(a5['pf']-a5['pa']),
                'form_diff10': (h10['pf']-h10['pa'])-(a10['pf']-a10['pa']),
                'points_for_diff5': h5['pf']-a5['pf'], 'defence_diff5': a5['pa']-h5['pa'],
            })
            pending_updates.append((hid, aid, {
                'date': date, 'pf': float(g.home_score), 'pa': float(g.away_score), 'win': home_win
            }, {
                'date': date, 'pf': float(g.away_score), 'pa': float(g.home_score), 'win': int(g.away_score > g.home_score)
            }))
        for hid, aid, home_update, away_update in pending_updates:
            history.setdefault(hid, []).append(home_update)
            history.setdefault(aid, []).append(away_update)
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
    scored = raw.dropna(subset=['home_score','away_score']).copy()
    # API-Sports pode devolver códigos curtos ou descrições de estado. Normalizar
    # ambos e mostrar os valores observados para facilitar a auditoria.
    def is_final_status(value):
        status = str(value or '').strip().upper()
        if not status or status in {'NAN', 'NONE', 'NULL'}:
            return False
        compact = ''.join(ch for ch in status if ch.isalnum())
        # API-Sports NBA numeric status codes: 1=Not Started, 2=Live, 3=Finished, 4=Postponed, 5=Delayed, 6=Canceled.
        exact_codes = {'3', 'FT', 'AOT', 'AP', 'FINAL', 'FINISHED', 'COMPLETED', 'ENDED'}
        if compact in exact_codes:
            return True
        final_phrases = ('FINISHED', 'FINAL', 'COMPLETED', 'AFTER OVERTIME',
                         'AFTER PENALTIES', 'GAME ENDED', 'MATCH ENDED')
        if any(phrase in status for phrase in final_phrases):
            return True
        # Accept descriptions such as "FT - Finished" or "AOT (After Overtime)".
        return compact.startswith(('FTFINISHED', 'AOT', 'AP')) and not any(
            word in compact for word in ('PENDING', 'NOTFINAL', 'UNFINISHED')
        )

    status_norm = scored['status'].astype(str).str.strip().str.upper()
    finished = scored[status_norm.map(is_final_status)].copy()
    st.caption(f'Jogos com pontuação: {len(scored):,} · jogos com estado final reconhecido: {len(finished):,}.')
    with st.expander('Auditoria dos estados recebidos pela API'):
        status_counts = (raw['status'].fillna('(vazio)').astype(str).value_counts()
                         .rename_axis('status recebido').reset_index(name='número de jogos'))
        st.dataframe(status_counts, use_container_width=True, hide_index=True)
    if finished.empty and not scored.empty:
        st.warning('Ainda não foi reconhecido nenhum estado final. Consulta a tabela "Auditoria dos estados recebidos pela API" e envia-me os valores apresentados; não executes o backtest.')
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
    st.subheader('2. NBA OVER LAB — comparação de modelos')
    st.caption('Objetivo: prever melhor o total de pontos, com teste cronológico fora da amostra. Não utiliza odds históricas, pelo que não calcula rentabilidade real.')
    min_prior = st.slider('Mínimo de jogos anteriores por equipa', 0, 10, 3)
    folds = st.selectbox('Número de janelas cronológicas', [3, 4, 5], index=1)
    if st.button('Comparar modelos de totais (Over)', type='primary'):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
        from sklearn.dummy import DummyRegressor
        feat = feat.sort_values(['date', 'game_id']).reset_index(drop=True)
        feat = feat[(feat.home_games_prior >= min_prior) & (feat.away_games_prior >= min_prior)].copy()
        feat = feat.dropna(subset=['total_points'])
        reg_features = [
            'home_pf5','home_pa5','away_pf5','away_pa5','home_pf10','home_pa10','away_pf10','away_pa10',
            'rest_home_days','rest_away_days','form_diff5','form_diff10','points_for_diff5','defence_diff5'
        ]
        if len(feat) < 500:
            st.error(f'Amostra insuficiente após filtros: {len(feat)} jogos.')
        else:
            n = len(feat)
            initial_train = int(n * 0.60)
            remaining = n - initial_train
            test_size = max(1, remaining // folds)
            model_specs = {
                'Ridge (referência)': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                'Extra Trees': lambda: ExtraTreesRegressor(n_estimators=250, min_samples_leaf=4, max_features=0.9, random_state=42, n_jobs=-1),
                'Random Forest': lambda: RandomForestRegressor(n_estimators=250, min_samples_leaf=5, max_features=0.9, random_state=42, n_jobs=-1),
                'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=120, learning_rate=0.035, max_depth=2, loss='huber', random_state=42),
                'Média do treino (baseline)': lambda: DummyRegressor(strategy='mean'),
            }
            all_results = []
            fold_metrics = []
            for fold in range(folds):
                test_start = initial_train + fold * test_size
                test_end = n if fold == folds - 1 else min(n, test_start + test_size)
                if test_start >= n or test_end <= test_start:
                    continue
                train = feat.iloc[:test_start].copy()
                test = feat.iloc[test_start:test_end].copy()
                X_train = train[reg_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                X_test = test[reg_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                y_train = train['total_points'].astype(float)
                y_test = test['total_points'].astype(float).to_numpy()
                for model_name, factory in model_specs.items():
                    model = factory()
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    result = test[['game_id','date','season','home_team','away_team','home_score','away_score','total_points']].copy()
                    result['fold'] = fold + 1
                    result['model'] = model_name
                    result['predicted_total'] = pred
                    result['error'] = result['total_points'].to_numpy() - pred
                    result['abs_error'] = np.abs(result['error'])
                    result['actual_over_215_5'] = (result['total_points'] > 215.5).astype(int)
                    result['pred_over_215_5'] = (pred > 215.5).astype(int)
                    result['actual_over_220_5'] = (result['total_points'] > 220.5).astype(int)
                    result['pred_over_220_5'] = (pred > 220.5).astype(int)
                    result['actual_over_225_5'] = (result['total_points'] > 225.5).astype(int)
                    result['pred_over_225_5'] = (pred > 225.5).astype(int)
                    result['actual_over_230_5'] = (result['total_points'] > 230.5).astype(int)
                    result['pred_over_230_5'] = (pred > 230.5).astype(int)
                    all_results.append(result)
                    fold_metrics.append({
                        'Janela': fold + 1, 'Modelo': model_name, 'Jogos': len(test),
                        'MAE': mean_absolute_error(y_test, pred),
                        'RMSE': mean_squared_error(y_test, pred) ** 0.5,
                        'Correlação previsto-real': float(np.corrcoef(y_test, pred)[0,1]) if np.std(pred) > 1e-9 and np.std(y_test) > 1e-9 else np.nan,
                        'Viés médio (previsto - real)': float(np.mean(pred-y_test)),
                    })
            if all_results:
                results = pd.concat(all_results, ignore_index=True)
                metrics = pd.DataFrame(fold_metrics)
                aggregate = results.groupby('model', sort=False).apply(lambda g: pd.Series({
                    'Jogos': len(g),
                    'MAE': mean_absolute_error(g['total_points'], g['predicted_total']),
                    'RMSE': mean_squared_error(g['total_points'], g['predicted_total']) ** 0.5,
                    'Correlação previsto-real': float(np.corrcoef(g['total_points'], g['predicted_total'])[0,1]) if g['predicted_total'].std() > 1e-9 else np.nan,
                    'Viés médio (previsto - real)': float((g['predicted_total']-g['total_points']).mean()),
                })).reset_index().rename(columns={'model':'Modelo'}).sort_values('MAE')
                st.markdown('### Comparação agregada — jogos nunca vistos no treino')
                st.dataframe(aggregate.round(3), use_container_width=True, hide_index=True)
                best = aggregate.iloc[0]
                st.success(f"Melhor MAE nesta comparação: {best['Modelo']} — {best['MAE']:.2f} pontos. Esta escolha é exploratória; precisa de confirmação num período futuro independente.")
                st.markdown('### Desempenho por janela')
                st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
                st.markdown('### Diagnóstico específico para mercados Over')
                st.caption('A previsão de total acima da linha não equivale a uma aposta lucrativa: aqui medimos apenas se o modelo identifica jogos acima/abaixo de linhas fixas. Não há odds históricas neste conjunto.')
                diagnostics = []
                for model_name, g in results.groupby('model', sort=False):
                    row = {'Modelo': model_name}
                    for line in [215.5,220.5,225.5,230.5]:
                        actual_col = f'actual_over_{str(line).replace(".", "_")}'
                        pred_col = f'pred_over_{str(line).replace(".", "_")}'
                        row[f'Acerto direção Over/Under {line}'] = float((g[actual_col] == g[pred_col]).mean())
                        row[f'Over real {line}'] = float(g[actual_col].mean())
                        row[f'Over previsto {line}'] = float(g[pred_col].mean())
                    diagnostics.append(row)
                diag_df = pd.DataFrame(diagnostics)
                st.dataframe(diag_df.round(3), use_container_width=True, hide_index=True)
                st.markdown('### Previsões para auditoria')
                st.dataframe(results[results['model'] != 'Média do treino (baseline)'].head(40), use_container_width=True, hide_index=True)
                st.download_button('Descarregar comparação de modelos CSV', aggregate.to_csv(index=False).encode('utf-8-sig'), 'nba_over_model_comparison.csv', 'text/csv')
                st.download_button('Descarregar métricas por janela CSV', metrics.to_csv(index=False).encode('utf-8-sig'), 'nba_over_fold_metrics.csv', 'text/csv')
                st.download_button('Descarregar previsões Over CSV', results.to_csv(index=False).encode('utf-8-sig'), 'nba_over_predictions.csv', 'text/csv')
                st.warning('Não uses estes resultados diretamente para apostar. A seleção do melhor modelo nestas janelas pode sobrestimar o desempenho; precisamos de um período futuro independente e de odds históricas para testar valor esperado e ROI.')

st.divider()
st.markdown('**Fontes de dados:** API-Sports NBA. Esta aplicação é um laboratório separado da app principal e da Calibration Lab V1/V2.')
