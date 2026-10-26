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

st.set_page_config(page_title='NBA OVER LAB V3.1', page_icon='🏀', layout='wide')
st.title('🏀 NBA LAB PRO — NBA OVER LAB V3.1 | Probabilidades Over')
st.caption('Versão focada em probabilidades Over/Under, novas variáveis pré-jogo e validação cronológica. Mantém a app principal e as Labs V1/V2/V3 intactas.')

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'nba_lab_data'
DATA_DIR.mkdir(exist_ok=True)
GAMES_PATH = DATA_DIR / 'nba_games_history.csv'
FEATURES_PATH = DATA_DIR / 'nba_features_history_v31.csv'
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
            return {'pf': 110.0, 'pa': 110.0, 'win': .5, 'total': 220.0, 'pf_std': 12.0, 'pa_std': 12.0, 'total_std': 20.0, 'n': 0}
        return {
            'pf': float(np.mean([x['pf'] for x in recent])),
            'pa': float(np.mean([x['pa'] for x in recent])),
            'win': float(np.mean([x['win'] for x in recent])),
            'total': float(np.mean([x.get('pf', 110.0) + x.get('pa', 110.0) for x in recent])),
            'pf_std': float(np.std([x['pf'] for x in recent], ddof=1)) if len(recent) > 1 else 12.0,
            'pa_std': float(np.std([x['pa'] for x in recent], ddof=1)) if len(recent) > 1 else 12.0,
            'total_std': float(np.std([x.get('pf', 110.0) + x.get('pa', 110.0) for x in recent], ddof=1)) if len(recent) > 1 else 20.0,
            'n': len(recent),
        }

    # Process identical start times as a batch to prevent same-time result leakage.
    for date, batch in df.groupby('date', sort=True):
        pending_updates = []
        for _, g in batch.iterrows():
            hid, aid = str(g.home_id), str(g.away_id)
            hh, ah = history.get(hid, []), history.get(aid, [])
            h3, a3 = stats(hh, 3), stats(ah, 3)
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
                'home_pf3': h3['pf'], 'home_pa3': h3['pa'], 'away_pf3': a3['pf'], 'away_pa3': a3['pa'],
                'home_total3': h3['total'], 'away_total3': a3['total'],
                'home_pf5': h5['pf'], 'home_pa5': h5['pa'], 'away_pf5': a5['pf'], 'away_pa5': a5['pa'],
                'home_total5': h5['total'], 'away_total5': a5['total'],
                'home_pf_std5': h5['pf_std'], 'away_pf_std5': a5['pf_std'],
                'home_pa_std5': h5['pa_std'], 'away_pa_std5': a5['pa_std'],
                'home_total_std5': h5['total_std'], 'away_total_std5': a5['total_std'],
                'home_win5': h5['win'], 'away_win5': a5['win'], 'home_pf10': h10['pf'], 'home_pa10': h10['pa'],
                'away_pf10': a10['pf'], 'away_pa10': a10['pa'],
                'home_total10': h10['total'], 'away_total10': a10['total'],
                'home_win10': h10['win'], 'away_win10': a10['win'],
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
    st.subheader('2. NBA OVER LAB V3.1 — probabilidades e calibração')
    st.caption('Compara previsões de pontos e classificadores probabilísticos por linha. O teste é cronológico e não usa odds históricas; não calcula ROI real.')
    min_prior = st.slider('Mínimo de jogos anteriores por equipa', 0, 10, 3)
    folds = st.selectbox('Número de janelas cronológicas', [3, 4, 5], index=2)
    if st.button('Executar comparação probabilística Over', type='primary'):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.dummy import DummyRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, mean_absolute_error, mean_squared_error
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression, Ridge

        feat = feat.sort_values(['date', 'game_id']).reset_index(drop=True)
        feat = feat[(feat.home_games_prior >= min_prior) & (feat.away_games_prior >= min_prior)].copy()
        feat = feat.dropna(subset=['total_points'])
        reg_features = [
            'home_pf3','home_pa3','away_pf3','away_pa3','home_total3','away_total3',
            'home_pf5','home_pa5','away_pf5','away_pa5','home_total5','away_total5',
            'home_pf10','home_pa10','away_pf10','away_pa10','home_total10','away_total10',
            'home_pf_std5','away_pf_std5','home_pa_std5','away_pa_std5','home_total_std5','away_total_std5',
            'home_win5','away_win5','home_win10','away_win10','rest_home_days','rest_away_days',
            'form_diff5','form_diff10','points_for_diff5','defence_diff5'
        ]
        lines = [215.5, 220.5, 225.5, 230.5]
        if len(feat) < 500:
            st.error(f'Amostra insuficiente após filtros: {len(feat)} jogos.')
        else:
            n = len(feat)
            initial_train = int(n * 0.60)
            remaining = n - initial_train
            test_size = max(1, remaining // folds)
            reg_specs = {
                'Ridge': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0)),
                'Gradient Boosting': lambda: make_pipeline(SimpleImputer(strategy='median'), GradientBoostingRegressor(n_estimators=150, learning_rate=0.03, max_depth=2, loss='huber', random_state=42)),
                'Extra Trees': lambda: make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(n_estimators=220, min_samples_leaf=5, max_features=0.85, random_state=42, n_jobs=-1)),
                'Random Forest': lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=220, min_samples_leaf=6, max_features=0.85, random_state=42, n_jobs=-1)),
                'Média treino': lambda: make_pipeline(SimpleImputer(strategy='median'), DummyRegressor(strategy='mean')),
            }
            clf_specs = {
                'Logistic Regression': lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.35, max_iter=1500, class_weight=None)),
                'Hist Gradient Boosting': lambda: make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=3.0, random_state=42)),
                'Random Forest Classifier': lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=120, min_samples_leaf=8, max_features=0.85, class_weight='balanced_subsample', random_state=42, n_jobs=-1)),
            }
            reg_rows, prob_rows, fold_rows = [], [], []
            progress = st.progress(0)
            status = st.empty()
            fold_ranges = []
            for fold in range(folds):
                test_start = initial_train + fold * test_size
                test_end = n if fold == folds - 1 else min(n, test_start + test_size)
                if test_start < n and test_end > test_start:
                    fold_ranges.append((fold + 1, test_start, test_end))
            total_steps = max(1, len(fold_ranges) * (len(reg_specs) + len(lines) * len(clf_specs)))
            step = 0
            for fold, test_start, test_end in fold_ranges:
                train, test = feat.iloc[:test_start].copy(), feat.iloc[test_start:test_end].copy()
                X_train = train[reg_features].replace([np.inf, -np.inf], np.nan)
                X_test = test[reg_features].replace([np.inf, -np.inf], np.nan)
                y_train = train['total_points'].astype(float)
                y_test = test['total_points'].astype(float).to_numpy()
                common = test[['game_id','date','season','home_team','away_team','home_score','away_score','total_points']].copy()
                common['fold'] = fold
                for name, factory in reg_specs.items():
                    status.info(f'Janela {fold}/{len(fold_ranges)} — modelo de total: {name}')
                    model = factory()
                    model.fit(X_train, y_train)
                    pred = np.asarray(model.predict(X_test), dtype=float)
                    pred = np.clip(pred, 150, 300)
                    # Estimar a distribuição dos erros uma única vez por modelo/janela,
                    # usando um bloco cronológico de validação dentro do treino.
                    split = max(30, int(len(train) * 0.80))
                    if split < len(train) - 20:
                        cal_model = factory()
                        cal_model.fit(train.iloc[:split][reg_features].replace([np.inf,-np.inf],np.nan), train.iloc[:split]['total_points'].astype(float))
                        cal_pred = cal_model.predict(train.iloc[split:][reg_features].replace([np.inf,-np.inf],np.nan))
                        residuals = train.iloc[split:]['total_points'].to_numpy(dtype=float) - np.asarray(cal_pred, dtype=float)
                    else:
                        residuals = y_train.to_numpy() - np.asarray(model.predict(X_train), dtype=float)
                    for i, line in enumerate(lines):
                        actual = (y_test > line).astype(int)
                        p = np.array([np.mean(pred[j] + residuals > line) for j in range(len(pred))], dtype=float)
                        p = np.clip(p, 0.01, 0.99)
                        for j in range(len(test)):
                            reg_rows.append({**common.iloc[j].to_dict(), 'model': name, 'line': line, 'predicted_total': float(pred[j]), 'actual_over': int(actual[j]), 'over_probability': float(p[j]), 'method': 'Regressão + resíduos cronológicos'})
                    fold_rows.append({'Janela':fold,'Tipo':'Regressão','Modelo':name,'Jogos':len(test),'MAE':mean_absolute_error(y_test,pred),'RMSE':mean_squared_error(y_test,pred)**0.5,'Correlação':float(np.corrcoef(y_test,pred)[0,1]) if np.std(pred)>1e-9 else np.nan,'Viés previsto-real':float(np.mean(pred-y_test))})
                    step += 1; progress.progress(min(0.99, step/total_steps))
                for line in lines:
                    ytr = (y_train.to_numpy() > line).astype(int)
                    yte = (y_test > line).astype(int)
                    for name, factory in clf_specs.items():
                        status.info(f'Janela {fold}/{len(fold_ranges)} — linha Over {line}: {name}')
                        if len(np.unique(ytr)) < 2:
                            continue
                        clf = factory()
                        clf.fit(X_train, ytr)
                        p = np.asarray(clf.predict_proba(X_test)[:,1], dtype=float)
                        p = np.clip(p, 0.01, 0.99)
                        for j in range(len(test)):
                            prob_rows.append({**common.iloc[j].to_dict(), 'fold':fold, 'model':name, 'line':line, 'actual_over':int(yte[j]), 'over_probability':float(p[j]), 'predicted_side':int(p[j]>=0.5), 'method':'Classificador direto'})
                        fold_rows.append({'Janela':fold,'Tipo':f'Over {line}','Modelo':name,'Jogos':len(test),'MAE':np.nan,'RMSE':np.nan,'Correlação':np.nan,'Viés previsto-real':np.nan})
                        step += 1; progress.progress(min(0.99, step/total_steps))
            progress.progress(1.0)
            status.success('Comparação concluída.')
            if reg_rows:
                reg_df = pd.DataFrame(reg_rows)
                reg_summary = reg_df.groupby('model').agg(Jogos=('total_points','size'), MAE=('total_points', lambda x: np.nan), RMSE=('total_points', lambda x: np.nan)).reset_index()
                reg_summary = reg_df.groupby('model').apply(lambda g: pd.Series({'Jogos':len(g.drop_duplicates('game_id')),'MAE':mean_absolute_error(g.drop_duplicates('game_id')['total_points'],g.drop_duplicates('game_id')['predicted_total']),'RMSE':mean_squared_error(g.drop_duplicates('game_id')['total_points'],g.drop_duplicates('game_id')['predicted_total'])**0.5,'Correlação':np.corrcoef(g.drop_duplicates('game_id')['total_points'],g.drop_duplicates('game_id')['predicted_total'])[0,1]})).reset_index().sort_values('MAE')
                st.markdown('### A. Qual modelo prevê melhor o total?')
                st.dataframe(reg_summary.round(3), use_container_width=True, hide_index=True)
                st.download_button('Descarregar previsões de totais CSV', reg_df.to_csv(index=False).encode('utf-8-sig'), 'nba_over_regression_predictions_v31.csv', 'text/csv')
            if prob_rows:
                prob_df = pd.DataFrame(prob_rows)
                def safe_logloss(y, p):
                    return log_loss(y, p, labels=[0,1]) if len(np.unique(y)) == 2 else np.nan
                summary_rows = []
                for (line, name), g in prob_df.groupby(['line','model']):
                    y, p = g['actual_over'].to_numpy(dtype=int), g['over_probability'].to_numpy(dtype=float)
                    summary_rows.append({'Linha':line,'Modelo':name,'Jogos':len(g),'Taxa Over real':float(y.mean()),'Probabilidade Over média':float(p.mean()),'Brier':brier_score_loss(y,p),'Log Loss':safe_logloss(y,p),'Acerto lado (p≥50%)':accuracy_score(y,(p>=0.5).astype(int)),'Over previsto (p≥50%)':float((p>=0.5).mean())})
                prob_summary = pd.DataFrame(summary_rows).sort_values(['Linha','Brier'])
                st.markdown('### B. Probabilidades Over por linha')
                st.caption('Brier e Log Loss mais baixos são melhores. A taxa Over real é a referência simples. Uma boa previsão probabilística deve ser precisa e calibrada; isto ainda não demonstra rentabilidade sem odds históricas.')
                st.dataframe(prob_summary.round(4), use_container_width=True, hide_index=True)
                st.download_button('Descarregar probabilidades Over CSV', prob_df.to_csv(index=False).encode('utf-8-sig'), 'nba_over_probabilities_v31.csv', 'text/csv')
                st.markdown('### C. Auditoria da calibração')
                cal_rows=[]
                for (line,name),g in prob_df.groupby(['line','model']):
                    bins = pd.cut(g['over_probability'], bins=[0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.000001], include_lowest=True)
                    for interval, part in g.groupby(bins, observed=False):
                        if len(part):
                            cal_rows.append({'Linha':line,'Modelo':name,'Intervalo probabilidade':str(interval),'Jogos':len(part),'Prob. média':part['over_probability'].mean(),'Over real':part['actual_over'].mean()})
                cal_df=pd.DataFrame(cal_rows)
                st.dataframe(cal_df.round(4),use_container_width=True,hide_index=True)
                st.download_button('Descarregar calibração CSV',cal_df.to_csv(index=False).encode('utf-8-sig'),'nba_over_calibration_v31.csv','text/csv')
            if fold_rows:
                st.markdown('### D. Métricas por janela')
                fold_df=pd.DataFrame(fold_rows)
                st.dataframe(fold_df.round(4),use_container_width=True,hide_index=True)
                st.download_button('Descarregar métricas por janela CSV',fold_df.to_csv(index=False).encode('utf-8-sig'),'nba_over_fold_metrics_v31.csv','text/csv')
            st.warning('Não uses estas probabilidades diretamente para apostar. A validação ainda precisa de um período futuro independente e odds históricas para avaliar valor esperado, ROI e drawdown.')

st.divider()
st.markdown('**Fontes de dados:** API-Sports NBA. A V3.1 é um laboratório separado; não altera a app principal nem as versões anteriores.')
