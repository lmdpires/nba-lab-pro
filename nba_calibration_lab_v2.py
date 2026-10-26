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
st.title('🏀 NBA LAB PRO — Calibration Lab v2')
st.caption('Laboratório separado da app principal. Auditoria de resultados, features pré-jogo e validação cronológica em várias janelas.')

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
        exact_codes = {'FT', 'AOT', 'AP', 'FINAL', 'FINISHED', 'COMPLETED', 'ENDED'}
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
    st.subheader('2. Auditoria e backtest cronológico')
    st.caption('Usa várias janelas cronológicas com treino expansivo. Jogos com o mesmo instante de início não usam os resultados uns dos outros para criar features.')
    min_prior = st.slider('Mínimo de jogos anteriores por equipa', 0, 10, 3)
    folds = st.selectbox('Número de janelas de teste', [3, 4, 5], index=1, help='Cada janela testa um período posterior ao treino; as janelas são comparadas separadamente.')
    if st.button('Executar auditoria em várias janelas', type='primary'):
        feat = feat.sort_values(['date', 'game_id']).reset_index(drop=True)
        feat = feat[(feat.home_games_prior >= min_prior) & (feat.away_games_prior >= min_prior)].copy()
        feat = feat.dropna(subset=['home_win', 'total_points'])
        if len(feat) < 500:
            st.error(f'Amostra insuficiente após filtros: {len(feat)} jogos. Tenta reduzir o mínimo de jogos anteriores ou confirma a base histórica.')
        else:
            features = ['home_pf5','home_pa5','away_pf5','away_pa5','home_win5','away_win5','home_pf10','home_pa10','away_pf10','away_pa10','home_win10','away_win10','rest_home_days','rest_away_days','form_diff5','form_diff10','points_for_diff5','defence_diff5']
            reg_features = ['home_pf5','home_pa5','away_pf5','away_pa5','home_pf10','home_pa10','away_pf10','away_pa10','rest_home_days','rest_away_days','form_diff5','form_diff10','points_for_diff5','defence_diff5']
            n = len(feat)
            # Reserve the first 60% for initial training, then evaluate successive non-overlapping periods.
            initial_train = int(n * 0.60)
            remaining = n - initial_train
            test_size = max(1, remaining // folds)
            all_results = []
            fold_rows = []
            for fold in range(folds):
                test_start = initial_train + fold * test_size
                test_end = n if fold == folds - 1 else min(n, test_start + test_size)
                if test_start >= n or test_end <= test_start:
                    continue
                train = feat.iloc[:test_start].copy()
                test = feat.iloc[test_start:test_end].copy()
                if train['home_win'].nunique() < 2 or len(test) < 30:
                    continue
                cls = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
                cls.fit(train[features].fillna(0), train['home_win'].astype(int))
                p = cls.predict_proba(test[features].fillna(0))[:, 1]
                pred = (p >= .5).astype(int)
                y = test['home_win'].astype(int).to_numpy()
                brier = brier_score_loss(y, p)
                ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0,1])
                acc = accuracy_score(y, pred)
                majority = max(float(train['home_win'].mean()), 1.0-float(train['home_win'].mean()))
                reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                reg.fit(train[reg_features].fillna(0), train['total_points'])
                total_pred = reg.predict(test[reg_features].fillna(0))
                mae = mean_absolute_error(test['total_points'], total_pred)
                rmse = mean_squared_error(test['total_points'], total_pred) ** 0.5
                baseline_pred = np.repeat(train['total_points'].mean(), len(test))
                baseline_mae = mean_absolute_error(test['total_points'], baseline_pred)
                result = test[['game_id','date','season','home_team','away_team','home_score','away_score','home_win','total_points']].copy()
                result['fold'] = fold + 1
                result['p_home_win'] = p
                result['predicted_home_win'] = pred
                result['predicted_total'] = total_pred
                result['moneyline_correct'] = (pred == y)
                result['total_abs_error'] = np.abs(total_pred-test['total_points'].to_numpy())
                all_results.append(result)
                fold_rows.append({
                    'Janela': fold + 1, 'Treino até': str(train['date'].max())[:10],
                    'Teste desde': str(test['date'].min())[:10], 'Teste até': str(test['date'].max())[:10],
                    'Jogos': len(test), 'Accuracy': acc, 'Maioria treino': majority,
                    'Brier': brier, 'Log Loss': ll, 'Total MAE': mae, 'Baseline MAE': baseline_mae, 'Total RMSE': rmse,
                })
            if not all_results:
                st.error('Não foi possível criar janelas de teste suficientes. Verifica a cobertura da base e os filtros.')
            else:
                results = pd.concat(all_results, ignore_index=True).sort_values(['date','game_id'])
                folds_df = pd.DataFrame(fold_rows)
                st.markdown('### Resultados por janela')
                st.dataframe(folds_df, use_container_width=True, hide_index=True)
                st.markdown('### Resultado agregado fora da amostra')
                y_all = results['home_win'].astype(int)
                p_all = results['p_home_win'].clip(1e-6, 1-1e-6)
                agg_acc = accuracy_score(y_all, results['predicted_home_win'])
                agg_brier = brier_score_loss(y_all, p_all)
                agg_ll = log_loss(y_all, p_all, labels=[0,1])
                agg_mae = mean_absolute_error(results['total_points'], results['predicted_total'])
                agg_rmse = mean_squared_error(results['total_points'], results['predicted_total']) ** 0.5
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric('Jogos teste', f'{len(results):,}')
                c2.metric('Moneyline accuracy', f'{agg_acc:.1%}')
                c3.metric('Brier Score', f'{agg_brier:.4f}', 'menor é melhor')
                c4.metric('Log Loss', f'{agg_ll:.4f}', 'menor é melhor')
                c5.metric('Total MAE', f'{agg_mae:.2f} pts')
                st.write(f'RMSE agregado do total: **{agg_rmse:.2f} pontos**.')
                st.markdown('### Auditoria da calibração')
                bins = np.linspace(0, 1, 11)
                cal = results.copy()
                cal['intervalo_probabilidade'] = pd.cut(cal['p_home_win'], bins=bins, include_lowest=True, duplicates='drop')
                calibration = cal.groupby('intervalo_probabilidade', observed=False).agg(
                    jogos=('home_win','size'), probabilidade_media=('p_home_win','mean'),
                    taxa_real_vitoria_casa=('home_win','mean')
                ).reset_index()
                st.dataframe(calibration, use_container_width=True, hide_index=True)
                st.caption('Uma taxa real diferente da probabilidade média pode indicar má calibração; grupos pequenos exigem cautela.')
                st.markdown('### Previsões de exemplo')
                st.dataframe(results.head(30), use_container_width=True, hide_index=True)
                st.download_button('Descarregar previsões multi-janela CSV', results.to_csv(index=False).encode('utf-8-sig'), 'nba_backtest_multifold.csv', 'text/csv')
                st.download_button('Descarregar métricas por janela CSV', folds_df.to_csv(index=False).encode('utf-8-sig'), 'nba_backtest_fold_metrics.csv', 'text/csv')
                st.warning('A auditoria melhora a validação, mas ainda não prova rentabilidade. Antes de qualquer uso em apostas, valida também a qualidade/cobertura dos dados, odds históricas e eventuais mudanças no modelo.')

st.divider()
st.markdown('**Fontes de dados:** API-Sports NBA. A disponibilidade de estatísticas e épocas depende da cobertura da competição e do plano da API. O modelo não deve ser usado para apostas reais até os testes serem repetidos e auditados.')
