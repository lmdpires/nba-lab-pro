import datetime
import copy
import uuid
import io
import json
import re
import zipfile
from html import escape
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
from nba_over_m2_live import predict_total, probability_over
from nba_moneyline_m1_live import M1Live, parse_api_sports_payload, api_season_query

st.set_page_config(page_title="NBA LAB PRO", page_icon="🏀", layout="wide")


# Notificações temporárias após operações concluídas com sucesso.
_pending_bet_toast = st.session_state.pop("nba_bet_success_toast", None)
if _pending_bet_toast:
    st.toast(_pending_bet_toast, icon="✅", duration="short")

# Registo CSV local; descarregar backups regularmente se a app estiver na Streamlit Cloud.
LOG_PATH = Path(__file__).with_name("nba_prediction_log.csv")
MARKET_DRAFT_PATH = Path(__file__).with_name("nba_market_inputs.csv")
BET_LOG_PATH = Path(__file__).with_name("nba_actual_bets.csv")
BET_COLUMNS = ["bet_id", "saved_at", "game_id", "game_date", "home_team", "away_team", "selection", "line", "odds", "stake", "actual_total", "result", "profit_loss"]
LOG_COLUMNS = [
    "record_id", "saved_at", "game_date", "game_id", "start_time",
    "freeze_at", "away_team", "home_team", "model_label", "capture_mode",
    "moneyline_pick", "p_home", "p_away", "fair_home_odds", "fair_away_odds",
    "home_market_odds", "away_market_odds", "moneyline_edge_pct", "away_moneyline_edge_pct",
    "total_reference", "total_line", "total_gap", "total_pick",
    "m2_probability_over", "m2_probability_under", "m2_model_version",
    "home_score", "away_score", "actual_total", "game_status", "result_updated_at",
    "moneyline_result", "total_result"
]

def load_log():
    if LOG_PATH.exists():
        try:
            df = pd.read_csv(LOG_PATH)
            for col in LOG_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[LOG_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=LOG_COLUMNS)

def save_log(df):
    df.to_csv(LOG_PATH, index=False)

def load_bets():
    if BET_LOG_PATH.exists():
        try:
            df = pd.read_csv(BET_LOG_PATH)
            for col in BET_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[BET_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=BET_COLUMNS)

def save_bets(df):
    df.to_csv(BET_LOG_PATH, index=False)


# Boletins reais: um registo por boletim e uma linha por seleção.
SLIP_PATH = Path(__file__).with_name("nba_bet_slips.csv")
LEG_PATH = Path(__file__).with_name("nba_bet_selections.csv")
DELETED_LEGACY_PATH = Path(__file__).with_name("nba_deleted_legacy_bets.csv")


# Persistência opcional no GitHub (necessária na Streamlit Community Cloud).
# Configurar em Streamlit Cloud > App > Settings > Secrets:
# GITHUB_TOKEN = "ghp_..."
# GITHUB_REPOSITORY = "lmdpires/nba-lab-pro"
# GITHUB_BRANCH = "main"
GITHUB_API = "https://api.github.com"
GITHUB_FILES = ("nba_bet_slips.csv", "nba_bet_selections.csv")

def github_configured():
    try:
        return bool(st.secrets.get("GITHUB_TOKEN")) and bool(
            st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
        )
    except Exception:
        return False

def github_headers():
    return {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def github_read_headers():
    # Autenticar também as leituras para usar o limite superior da API GitHub.
    # O token está guardado apenas nos Secrets da Streamlit Cloud.
    return github_headers()


def github_diagnose_access():
    """Diagnóstico seguro: devolve estados HTTP e utilizador, nunca credenciais."""
    if not github_configured():
        return "Secrets: GITHUB_TOKEN ou GITHUB_REPOSITORY em falta."

    repo = st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    headers = github_headers()
    checks = [
        ("Identidade do token", f"{GITHUB_API}/user"),
        ("Acesso ao repositório", f"{GITHUB_API}/repos/{repo}"),
        ("Acesso à branch", f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}"),
    ]
    results = [f"Repositório configurado: {repo}", f"Branch configurada: {branch}"]
    for label, url in checks:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            detail = ""
            if response.status_code == 200:
                try:
                    payload = response.json()
                    if label == "Identidade do token":
                        detail = f" (conta: {payload.get('login', 'não identificada')})"
                    elif label == "Acesso ao repositório":
                        detail = f" (repositório: {payload.get('full_name', 'não identificado')})"
                except Exception:
                    pass
            elif response.status_code in (401, 403, 404):
                try:
                    detail = f" ({response.json().get('message', 'sem detalhe')})"
                except Exception:
                    pass
            results.append(f"{label}: HTTP {response.status_code}{detail}")
        except Exception as exc:
            results.append(f"{label}: erro de ligação ({type(exc).__name__})")
    return "\\n".join(results)

def github_get_branch_head():
    repo = st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    url = f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}"
    response = requests.get(url, headers=github_read_headers(), timeout=20)
    response.raise_for_status()
    return response.json()["object"]["sha"]

def github_read_csv(filename, commit_sha=None):
    if not github_configured():
        return None
    repo = st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
    ref = commit_sha or st.secrets.get("GITHUB_BRANCH", "main")
    url = f"{GITHUB_API}/repos/{repo}/contents/{filename}"
    response = requests.get(url, headers=github_read_headers(),
                            params={"ref": ref}, timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    import base64
    return base64.b64decode(response.json()["content"])

def github_commit_bet_csvs(slips_bytes, legs_bytes):
    """Publica os dois CSV num único commit Git, atualizando a branch sem force."""
    if not github_configured():
        raise RuntimeError("GitHub não está configurado nos Secrets da aplicação.")

    import base64
    repo = st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    headers = github_headers()

    # Captura o estado-base da branch; a atualização final só avança se não houver corrida.
    # GET usa /git/ref/... (singular); PATCH usa /git/refs/... (plural).
    ref_url = f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}"
    update_ref_url = f"{GITHUB_API}/repos/{repo}/git/refs/heads/{branch}"
    ref_response = requests.get(ref_url, headers=github_read_headers(), timeout=20)
    ref_response.raise_for_status()
    base_commit_sha = ref_response.json()["object"]["sha"]

    commit_url = f"{GITHUB_API}/repos/{repo}/git/commits/{base_commit_sha}"
    base_commit_response = requests.get(commit_url, headers=github_read_headers(), timeout=20)
    base_commit_response.raise_for_status()
    base_tree_sha = base_commit_response.json()["tree"]["sha"]

    def create_blob(content):
        response = requests.post(
            f"{GITHUB_API}/repos/{repo}/git/blobs",
            headers=headers,
            json={
                "content": base64.b64encode(content).decode("ascii"),
                "encoding": "base64",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["sha"]

    slips_blob = create_blob(slips_bytes)
    legs_blob = create_blob(legs_bytes)

    tree_response = requests.post(
        f"{GITHUB_API}/repos/{repo}/git/trees",
        headers=headers,
        json={
            "base_tree": base_tree_sha,
            "tree": [
                {"path": "nba_bet_slips.csv", "mode": "100644", "type": "blob", "sha": slips_blob},
                {"path": "nba_bet_selections.csv", "mode": "100644", "type": "blob", "sha": legs_blob},
            ],
        },
        timeout=30,
    )
    tree_response.raise_for_status()
    new_tree_sha = tree_response.json()["sha"]

    new_commit_response = requests.post(
        f"{GITHUB_API}/repos/{repo}/git/commits",
        headers=headers,
        json={
            "message": "Update NBA LAB PRO bet history",
            "tree": new_tree_sha,
            "parents": [base_commit_sha],
        },
        timeout=30,
    )
    new_commit_response.raise_for_status()
    new_commit_sha = new_commit_response.json()["sha"]

    update_response = requests.patch(
        update_ref_url,
        headers=headers,
        json={"sha": new_commit_sha, "force": False},
        timeout=30,
    )
    if update_response.status_code in (409, 422):
        raise RuntimeError(
            "A branch main mudou durante a gravação. Nenhum dos dois CSV foi publicado; "
            "atualize a aplicação e tente novamente."
        )
    update_response.raise_for_status()
    return new_commit_sha

def sync_bet_csvs_from_github():
    """Lê os dois CSV do mesmo commit e só substitui os locais após validar ambos."""
    if not github_configured():
        return

    try:
        commit_sha = github_get_branch_head()
        slips_content = github_read_csv("nba_bet_slips.csv", commit_sha)
        legs_content = github_read_csv("nba_bet_selections.csv", commit_sha)

        if slips_content is None or legs_content is None:
            raise RuntimeError(
                "Não encontrei os dois ficheiros de apostas no GitHub. "
                "Os CSV locais não foram alterados."
            )

        # Validar estrutura antes de tocar nos ficheiros locais.
        import io
        slips_check = pd.read_csv(io.BytesIO(slips_content))
        legs_check = pd.read_csv(io.BytesIO(legs_content))
        if not set(SLIP_COLUMNS).issubset(slips_check.columns):
            raise RuntimeError("O CSV de boletins no GitHub tem colunas inesperadas.")
        if not set(LEG_COLUMNS).issubset(legs_check.columns):
            raise RuntimeError("O CSV de seleções no GitHub tem colunas inesperadas.")

        SLIP_PATH.write_bytes(slips_content)
        LEG_PATH.write_bytes(legs_content)
    except Exception as exc:
        diagnostic = github_diagnose_access()
        st.error(
            "Não foi possível confirmar a leitura do histórico persistente no GitHub. "
            "Por segurança, a aplicação não deve continuar a gravar apostas até resolvermos o problema. "
            f"Detalhe: {exc}\\n\\nDiagnóstico de acesso (sem credenciais):\\n{diagnostic}"
        )
        st.stop()

SLIP_COLUMNS = ["slip_id", "saved_at", "bet_type", "stake", "combined_odds", "status", "return_amount", "profit_loss"]
LEG_COLUMNS = ["selection_id", "slip_id", "game_id", "game_date", "home_team", "away_team",
               "selection", "line", "odds", "away_score", "home_score", "actual_total", "result"]

def load_slips():
    if SLIP_PATH.exists():
        try:
            df = pd.read_csv(SLIP_PATH)
            for col in SLIP_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[SLIP_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=SLIP_COLUMNS)

def load_legs():
    if LEG_PATH.exists():
        try:
            df = pd.read_csv(LEG_PATH)
            for col in LEG_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[LEG_COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=LEG_COLUMNS)

def save_slip_data(slips, legs):
    """Guarda os dois CSV no mesmo commit GitHub; só depois atualiza a cópia local."""
    slips_csv = slips[SLIP_COLUMNS].to_csv(index=False).encode("utf-8")
    legs_csv = legs[LEG_COLUMNS].to_csv(index=False).encode("utf-8")

    if not github_configured():
        st.error(
            "As apostas não foram consideradas guardadas em segurança: "
            "a persistência GitHub não está configurada. Não feche a aplicação "
            "até configurar a ligação."
        )
        st.stop()

    try:
        github_commit_bet_csvs(slips_csv, legs_csv)
        SLIP_PATH.write_bytes(slips_csv)
        LEG_PATH.write_bytes(legs_csv)
    except Exception as exc:
        diagnostic = github_diagnose_access()
        st.error(
            "Falha ao guardar as apostas no GitHub. Os ficheiros persistentes não foram "
            "atualizados; não considere esta operação concluída. "
            f"Detalhe: {exc}\\n\\nDiagnóstico de acesso (sem credenciais):\\n{diagnostic}"
        )
        st.stop()

def load_deleted_legacy_bet_ids():
    """IDs de apostas antigas eliminadas; evita que a migração as recrie."""
    if not DELETED_LEGACY_PATH.exists():
        return set()
    try:
        deleted = pd.read_csv(DELETED_LEGACY_PATH)
        if "legacy_bet_id" not in deleted.columns:
            return set()
        return set(deleted["legacy_bet_id"].dropna().astype(str))
    except Exception:
        return set()


def add_deleted_legacy_bet_id(legacy_bet_id):
    """Regista a eliminação sem alterar o CSV original de apostas antigas."""
    deleted_ids = load_deleted_legacy_bet_ids()
    deleted_ids.add(str(legacy_bet_id))
    pd.DataFrame({"legacy_bet_id": sorted(deleted_ids)}).to_csv(
        DELETED_LEGACY_PATH, index=False
    )


def confirm_delete_slip(slip_id):
    """Elimina o boletim e as seleções associadas."""
    slip_id = str(slip_id)
    slips = load_slips()
    legs = load_legs()

    if slip_id not in set(slips["slip_id"].astype(str)):
        st.session_state["delete_slip_feedback"] = (
            "Não foi possível encontrar o boletim. Atualiza o histórico e tenta novamente."
        )
        st.session_state["delete_slip_feedback_error"] = True
        st.session_state["pending_delete_slip_id"] = None
        return

    # Os boletins legacy são importados de nba_actual_bets.csv.
    # Guardamos um tombstone para não voltarem a aparecer após o rerun.
    if slip_id.startswith("legacy-"):
        add_deleted_legacy_bet_id(slip_id[len("legacy-"):])

    slips = slips[slips["slip_id"].astype(str) != slip_id].copy()
    legs = legs[legs["slip_id"].astype(str) != slip_id].copy()
    save_slip_data(slips, legs)

    if st.session_state.get("manage_slip_id") == slip_id:
        st.session_state["manage_slip_id"] = None
    st.session_state["pending_delete_slip_id"] = None
    st.session_state["delete_slip_feedback"] = "Boletim eliminado com sucesso."
    st.session_state["delete_slip_feedback_error"] = False


def migrate_legacy_bets(slips, legs):
    """Importa apostas antigas como boletins simples, sem apagar o CSV original."""
    old = load_bets()
    if old.empty:
        return slips, legs, False
    known = set(slips["slip_id"].astype(str)) if not slips.empty else set()
    deleted_legacy_ids = load_deleted_legacy_bet_ids()
    added = False
    for _, row in old.iterrows():
        legacy_id = str(row.get("bet_id") or uuid.uuid4().hex[:8])
        slip_id = f"legacy-{legacy_id}"
        if slip_id in known or legacy_id in deleted_legacy_ids:
            continue
        try:
            stake = float(row["stake"])
            odds = float(row["odds"])
            line = float(row["line"])
        except (TypeError, ValueError):
            continue
        slip = {"slip_id": slip_id, "saved_at": row.get("saved_at"),
                "bet_type": "Simples", "stake": stake, "combined_odds": odds,
                "status": row.get("result", "Pendente"),
                "return_amount": None, "profit_loss": row.get("profit_loss")}
        leg = {"selection_id": f"{slip_id}-1", "slip_id": slip_id,
               "game_id": row.get("game_id"), "game_date": row.get("game_date"),
               "home_team": row.get("home_team"), "away_team": row.get("away_team"),
               "selection": row.get("selection"), "line": line, "odds": odds,
               "away_score": None, "home_score": None,
               "actual_total": row.get("actual_total"), "result": row.get("result", "Pendente")}
        slips = pd.concat([slips, pd.DataFrame([slip])], ignore_index=True)
        legs = pd.concat([legs, pd.DataFrame([leg])], ignore_index=True)
        known.add(slip_id)
        added = True
    return slips, legs, added

def reconcile_slips(slips, legs, games_by_id):
    """Liquida cada seleção com resultados finais da API; não consulta nem altera M2."""
    if legs.empty:
        return slips, legs, False
    changed = False
    for idx, leg in legs.iterrows():
        game = games_by_id.get(normalize_game_id(leg.get("game_id")))
        if not game or str(game.get("status", "")).upper() != "FT":
            continue
        hs = pd.to_numeric(game.get("home_score"), errors="coerce")
        aws = pd.to_numeric(game.get("away_score"), errors="coerce")
        if pd.isna(hs) or pd.isna(aws):
            continue
        actual = float(hs + aws)
        side = str(leg.get("selection", ""))
        side_norm = side.casefold()
        if "moneyline" in side_norm:
            # Moneyline is settled against the selected team's final winner.
            if "casa" in side_norm or "home" in side_norm:
                result = "Acertou" if hs > aws else ("Push" if hs == aws else "Falhou")
            elif "visitante" in side_norm or "fora" in side_norm or "away" in side_norm:
                result = "Acertou" if aws > hs else ("Push" if hs == aws else "Falhou")
            else:
                # Unknown legacy label: do not silently classify it.
                continue
            actual_value = float(hs + aws)
        else:
            try:
                line = float(leg["line"])
            except (TypeError, ValueError):
                continue
            result = "Push" if actual == line else (
                "Acertou" if ((side == "Over" and actual > line) or (side == "Under" and actual < line)) else "Falhou"
            )
            actual_value = actual

        old_actual = pd.to_numeric(legs.at[idx, "actual_total"], errors="coerce")
        actual_changed = (
            pd.isna(old_actual)
            or (not pd.isna(old_actual) and abs(float(old_actual) - actual_value) > 0.001)
        )
        old_away_score = pd.to_numeric(legs.at[idx, "away_score"], errors="coerce")
        old_home_score = pd.to_numeric(legs.at[idx, "home_score"], errors="coerce")
        scores_changed = (
            pd.isna(old_away_score) or pd.isna(old_home_score)
            or int(old_away_score) != int(aws) or int(old_home_score) != int(hs)
        )
        if str(legs.at[idx, "result"]) != result or actual_changed or scores_changed:
            # The matchup is displayed Away @ Home, so preserve both scores explicitly.
            legs.at[idx, "away_score"] = int(aws)
            legs.at[idx, "home_score"] = int(hs)
            legs.at[idx, "actual_total"] = actual_value
            legs.at[idx, "result"] = result
            changed = True

    for idx, slip in slips.iterrows():
        sid = str(slip["slip_id"])
        group = legs[legs["slip_id"].astype(str) == sid]
        if group.empty:
            continue
        results = group["result"].fillna("Pendente").astype(str).tolist()
        try:
            stake = float(slip["stake"])
            combined = float(slip["combined_odds"])
        except (TypeError, ValueError):
            continue
        if "Falhou" in results:
            status, returned = "Perdido", 0.0
        elif all(r in ("Acertou", "Push") for r in results):
            status = "Ganho" if "Acertou" in results else "Push"
            if all(r == "Acertou" for r in results):
                returned = stake * combined
            elif all(r == "Push" for r in results):
                returned = stake
            else:
                # Push anula essa perna; odds remanescentes dão uma estimativa.
                remaining = pd.to_numeric(group.loc[group["result"] == "Acertou", "odds"], errors="coerce").dropna()
                returned = stake * float(remaining.prod()) if not remaining.empty else stake
        else:
            status, returned = "Pendente", None
        profit = round(returned - stake, 2) if returned is not None else None
        old_status = str(slip.get("status", ""))
        old_return = pd.to_numeric(slip.get("return_amount"), errors="coerce")
        old_profit = pd.to_numeric(slip.get("profit_loss"), errors="coerce")
        if old_status != status or (pd.isna(old_return) and returned is not None) or (returned is not None and not pd.isna(old_return) and abs(float(old_return)-returned) > 0.001) or (pd.isna(old_profit) and profit is not None) or (profit is not None and not pd.isna(old_profit) and abs(float(old_profit)-profit) > 0.001):
            slips.at[idx, "status"] = status
            slips.at[idx, "return_amount"] = returned
            slips.at[idx, "profit_loss"] = profit
            changed = True
    return slips, legs, changed

def reconcile_bets(bets, games_by_id):
    """Settle bets from final API game scores, independently of M2 predictions."""
    if bets.empty or not games_by_id:
        return bets, 0
    changed = 0
    for idx, bet in bets.iterrows():
        game = games_by_id.get(normalize_game_id(bet.get("game_id")))
        if not game or str(game.get("status", "")).upper() != "FT":
            continue
        home_score = pd.to_numeric(game.get("home_score"), errors="coerce")
        away_score = pd.to_numeric(game.get("away_score"), errors="coerce")
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        actual = home_score + away_score
        try:
            line, odds, stake = float(bet["line"]), float(bet["odds"]), float(bet["stake"])
        except (TypeError, ValueError):
            continue
        selection = str(bet.get("selection", ""))
        result = "Push" if actual == line else ("Acertou" if ((selection == "Over" and actual > line) or (selection == "Under" and actual < line)) else "Falhou")
        profit = 0.0 if result == "Push" else (round(stake * (odds - 1), 2) if result == "Acertou" else round(-stake, 2))
        if str(bets.at[idx, "result"]) != result or pd.to_numeric(bets.at[idx, "actual_total"], errors="coerce") != actual:
            bets.at[idx, "actual_total"] = int(actual)
            bets.at[idx, "result"] = result
            bets.at[idx, "profit_loss"] = profit
            changed += 1
    return bets, changed

def normalize_game_id(value):
    if value is None:
        return ""
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()

def load_market_drafts():
    """Load the latest user-entered odds and total line for each game."""
    if not MARKET_DRAFT_PATH.exists():
        return {}
    try:
        df = pd.read_csv(MARKET_DRAFT_PATH)
        drafts = {}
        for _, row in df.iterrows():
            drafts[normalize_game_id(row.get("game_id"))] = {
                "home_market_odds": pd.to_numeric(row.get("home_market_odds"), errors="coerce"),
                "away_market_odds": pd.to_numeric(row.get("away_market_odds"), errors="coerce"),
                "total_line": pd.to_numeric(row.get("total_line"), errors="coerce"),
                "updated_at": row.get("updated_at", "")
            }
        return drafts
    except Exception:
        return {}

def save_market_draft(game_id, home_odd, away_odd, total_line):
    """Persist the current inputs immediately on every Streamlit rerun."""
    key = normalize_game_id(game_id)
    drafts = load_market_drafts()
    drafts[key] = {
        "home_market_odds": float(home_odd),
        "away_market_odds": float(away_odd),
        "total_line": float(total_line),
        "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    }
    rows = []
    for gid, values in drafts.items():
        rows.append({"game_id": gid, **values})
    pd.DataFrame(rows, columns=[
        "game_id", "home_market_odds", "away_market_odds", "total_line", "updated_at"
    ]).to_csv(MARKET_DRAFT_PATH, index=False)

def model_snapshot(home, away, kickoff=None):
    """Frozen model snapshot: M1 v1.1 for Moneyline + unchanged M2 HGB v1 total."""
    try:
        m1_result = predict_moneyline_m1(home, away, kickoff)
        p_home = float(m1_result["p_home_m1"])
        p_away = float(m1_result["p_away_m1"])
    except Exception:
        # Fail closed: never substitute the old demonstrative Moneyline formula.
        p_home, p_away = None, None
    total_reference, m2_error = predict_total(home, away, kickoff=kickoff)
    return p_home, p_away, total_reference

def evaluate_record(row, home_score, away_score):
    """Evaluate saved Moneyline and total picks against the official final score."""
    total_score = int(home_score) + int(away_score)
    pick_ml = str(row.get("moneyline_pick", ""))
    if int(home_score) == int(away_score):
        ml_result = "Push"
    else:
        home_won = int(home_score) > int(away_score)
        ml_result = "Acertou" if ((pick_ml == "Casa" and home_won) or
                                  (pick_ml == "Fora" and not home_won)) else "Falhou"

    try:
        line = float(row.get("total_line"))
    except (TypeError, ValueError):
        line = None
    pick_total = str(row.get("total_pick", ""))
    if line is None or pick_total in ("", "Sem seleção", "nan"):
        total_result = "Sem linha/seleção"
    elif total_score == line:
        total_result = "Push"
    else:
        won = (pick_total == "Over" and total_score > line) or (pick_total == "Under" and total_score < line)
        total_result = "Acertou" if won else "Falhou"
    return total_score, ml_result, total_result

def recover_predictions_from_saved_drafts(games, log_df, drafts, now=None):
    """Recover a prediction from pre-game saved inputs if T-5 capture was missed."""
    now = now or datetime.datetime.now().astimezone()
    existing_ids = set(log_df["game_id"].map(normalize_game_id)) if not log_df.empty else set()
    recovered = []
    for game in games:
        gid = normalize_game_id(game.get("id"))
        if not gid or gid in existing_ids:
            continue
        draft = drafts.get(gid)
        if not draft:
            continue
        try:
            kickoff = datetime.datetime.fromtimestamp(float(game["timestamp"]), tz=now.tzinfo)
            draft_time = pd.to_datetime(draft.get("updated_at"), utc=True, errors="coerce")
            if pd.isna(draft_time):
                continue
            draft_time = draft_time.tz_convert(now.tzinfo)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        # Only use inputs saved before kickoff; never create a pre-game record from post-game edits.
        if draft_time.to_pydatetime() >= kickoff or now < kickoff - datetime.timedelta(minutes=5):
            continue
        home, away = game["home"], game["away"]
        p_home, p_away, total_reference = model_snapshot(home, away, kickoff=kickoff)
        if p_home is None or p_away is None:
            continue
        try:
            home_odd = float(draft.get("home_market_odds"))
            away_odd = float(draft.get("away_market_odds"))
        except (TypeError, ValueError):
            continue
        # Fixed reference line for consistent model evaluation; never substitute
        # the user's exploratory market line from the draft.
        total_line = 220.5 if total_reference is not None else None
        p_over, p_under = probability_over(total_reference, total_line) if total_reference is not None else (None, None)
        record = {
            "record_id": str(uuid.uuid4())[:8],
            "saved_at": draft_time.isoformat(timespec="seconds"),
            "game_date": kickoff.strftime("%Y-%m-%d"),
            "game_id": gid,
            "start_time": kickoff.strftime("%H:%M"),
            "freeze_at": (kickoff - datetime.timedelta(minutes=5)).isoformat(timespec="seconds"),
            "away_team": away, "home_team": home,
            "model_label": "M1 v1.1 Moneyline; M2 HGB v1 congelado" if total_reference is not None else "M1 v1.1 Moneyline; M2 indisponível",
            "capture_mode": "Recuperado de inputs pré-jogo",
            "moneyline_pick": "Casa" if p_home >= p_away else "Fora",
            "p_home": round(float(p_home), 4), "p_away": round(float(p_away), 4),
            "fair_home_odds": round(1 / p_home, 3), "fair_away_odds": round(1 / p_away, 3),
            "home_market_odds": home_odd, "away_market_odds": away_odd,
            "moneyline_edge_pct": round((home_odd / (1 / p_home) - 1) * 100, 2),
            "away_moneyline_edge_pct": round((away_odd / (1 / p_away) - 1) * 100, 2),
            "total_reference": round(float(total_reference), 1) if total_reference is not None else None,
            "total_line": total_line,
            "total_gap": round(float(total_reference) - total_line, 1) if total_reference is not None and total_line is not None else None,
            "total_pick": ("Over" if total_reference > total_line else ("Under" if total_reference < total_line else "Sem seleção")) if total_reference is not None and total_line is not None else "Sem seleção",
            "m2_probability_over": p_over, "m2_probability_under": p_under, "m2_model_version": "M2-HGB-v1-frozen" if total_reference is not None else None,
            "home_score": None, "away_score": None, "actual_total": None,
            "game_status": "Pendente", "result_updated_at": None,
            "moneyline_result": "Pendente", "total_result": "Pendente"
        }
        recovered.append(record)
        existing_ids.add(gid)
    if recovered:
        log_df = pd.concat([log_df, pd.DataFrame(recovered)], ignore_index=True)
    return log_df, len(recovered)

def upsert_api_results(games, log_df):
    """Apply API final scores to previously frozen predictions."""
    if log_df.empty:
        return log_df, 0

    # CSV inference can turn columns containing mostly blanks into float columns.
    # Pandas 3 / NumPy then reject string assignments (e.g. result_updated_at).
    # Normalize mutable columns to object before writing mixed text/numeric results.
    mutable_columns = [
        "home_score", "away_score", "actual_total", "game_status",
        "result_updated_at", "moneyline_result", "total_result"
    ]
    for column in mutable_columns:
        if column not in log_df.columns:
            log_df[column] = None
        log_df[column] = log_df[column].astype("object")

    changed = 0
    by_id = {normalize_game_id(g.get("id")): g for g in games}
    for idx, row in log_df.iterrows():
        game = by_id.get(normalize_game_id(row.get("game_id")))
        if not game or game.get("status") != "FT":
            continue
        home_score, away_score = game.get("home_score"), game.get("away_score")
        if home_score is None or away_score is None:
            continue
        if str(row.get("game_status", "")) == "Final" and pd.notna(row.get("actual_total")):
            continue
        total_score, ml_result, total_result = evaluate_record(row, home_score, away_score)
        log_df.at[idx, "home_score"] = int(home_score)
        log_df.at[idx, "away_score"] = int(away_score)
        log_df.at[idx, "actual_total"] = int(total_score)
        log_df.at[idx, "game_status"] = "Final"
        log_df.at[idx, "result_updated_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        log_df.at[idx, "moneyline_result"] = ml_result
        log_df.at[idx, "total_result"] = total_result
        changed += 1
    return log_df, changed

def auto_freeze_predictions(games, log_df, now=None):
    """Freeze an M1 v1.1 + M2 HGB v1 snapshot at T-5 when the app is active."""
    now = now or datetime.datetime.now().astimezone()
    if log_df.empty:
        existing_ids = set()
    else:
        existing_ids = set(log_df["game_id"].astype(str))
    records = []
    for game in games:
        try:
            kickoff = datetime.datetime.fromtimestamp(float(game["timestamp"]), tz=now.tzinfo)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        freeze_at = kickoff - datetime.timedelta(minutes=5)
        if now < freeze_at or str(game["id"]) in existing_ids:
            continue
        if game.get("status") in ("FT", "LIVE", "Q1", "Q2", "Q3", "Q4", "HT", "OT"):
            # Do not create a new "pre-game" prediction once play has started.
            continue
        home, away = game["home"], game["away"]
        p_home, p_away, total_reference = model_snapshot(home, away, kickoff=kickoff)
        if p_home is None or p_away is None:
            # Do not freeze an incomplete or demonstrative Moneyline snapshot.
            continue
        # No bookmaker odds feed is assumed here. Odds are deliberately left blank.
        # 220.5 is a technical reference only, never represented as a bookmaker quote.
        total_line = 220.5 if total_reference is not None else None
        p_over, p_under = probability_over(total_reference, total_line) if total_reference is not None else (None, None)
        records.append({
            "record_id": str(uuid.uuid4())[:8],
            "saved_at": now.isoformat(timespec="seconds"),
            "game_date": kickoff.strftime("%Y-%m-%d"),
            "game_id": str(game["id"]),
            "start_time": kickoff.strftime("%H:%M"),
            "freeze_at": freeze_at.isoformat(timespec="seconds"),
            "away_team": away, "home_team": home,
            "model_label": "M1 v1.1 Moneyline; M2 HGB v1 congelado" if total_reference is not None else "M1 v1.1 Moneyline; M2 indisponível",
            "capture_mode": "Automático após T-5 (app ativa)",
            "moneyline_pick": "Casa" if p_home >= p_away else "Fora",
            "p_home": round(float(p_home), 4), "p_away": round(float(p_away), 4),
            "fair_home_odds": round(1 / p_home, 3), "fair_away_odds": round(1 / p_away, 3),
            "home_market_odds": None, "away_market_odds": None,
            "moneyline_edge_pct": None, "away_moneyline_edge_pct": None,
            "total_reference": round(float(total_reference), 1) if total_reference is not None else None,
            "total_line": total_line,
            "total_gap": round(float(total_reference) - total_line, 1) if total_reference is not None and total_line is not None else None,
            "total_pick": ("Over" if total_reference > total_line else ("Under" if total_reference < total_line else "Sem seleção")) if total_reference is not None and total_line is not None else "Sem seleção",
            "m2_probability_over": p_over, "m2_probability_under": p_under, "m2_model_version": "M2-HGB-v1-frozen" if total_reference is not None else None,
            "home_score": None, "away_score": None, "actual_total": None,
            "game_status": "Pendente", "result_updated_at": None,
            "moneyline_result": "Pendente", "total_result": "Pendente"
        })
        existing_ids.add(str(game["id"]))
    if records:
        log_df = pd.concat([log_df, pd.DataFrame(records)], ignore_index=True)
    return log_df, len(records)

# Compact, two-column workspace: fixtures on the left, both markets on the right.
st.markdown("""
<style>
:root { --bg:#0a1018; --panel:#111b27; --line:#26384b; --muted:#8fa4ba; --white:#f2f6fb; --blue:#68c4ff; --green:#35d49a; }
.stApp {background:radial-gradient(ellipse at 10% 0%,rgba(27,91,133,.13),transparent 32%),var(--bg);color:var(--white)}
.block-container {max-width:1720px;padding:2.25rem 1.1rem .9rem!important}
header[data-testid="stHeader"] {background:#0a1018!important;height:2rem!important;min-height:2rem!important;border-bottom:1px solid #172432!important}
header[data-testid="stHeader"] [data-testid="stToolbar"] {background:transparent!important;height:2rem!important;min-height:2rem!important}
h1,h2,h3,p,label {margin-top:0}
h1 {font-size:1.42rem!important;line-height:1.4!important;letter-spacing:-.025em;font-weight:900!important;margin:0!important;padding:0!important;overflow:visible!important}
h2 {font-size:1.05rem!important;font-weight:800!important}
h3 {font-size:.95rem!important;font-weight:800!important}
p,label,[data-testid="stCaptionContainer"] p {color:var(--muted)}
hr {margin:.6rem 0!important;border-color:var(--line)!important}
.topbar {display:flex;align-items:center;gap:13px;padding:10px 16px;margin-bottom:11px;min-height:66px;box-sizing:border-box;overflow:visible;border:1px solid #2c4559;border-radius:10px;background:linear-gradient(100deg,#14283a,#111b27 70%,#12251f);box-shadow:0 5px 18px rgba(0,0,0,.12)}
.ball {font-size:1.55rem;line-height:1;flex:0 0 auto}
.brand-sub {font-size:.74rem;color:#b0c2d4;margin-top:3px;line-height:1.4}
.status {margin-left:auto;font-size:.66rem;color:#53dca9;font-weight:800;letter-spacing:.09em;white-space:nowrap;padding:6px 9px;border:1px solid #245344;border-radius:999px;background:rgba(16,76,59,.22)}
.section-label {font-size:.67rem;text-transform:uppercase;letter-spacing:.12em;font-weight:850;color:#83a1bc;margin:0 0 6px}
.left-panel,.right-panel {background:rgba(14,22,32,.7);border:1px solid #243447;border-radius:12px;padding:12px}
.fixture-help {font-size:.72rem;color:#8fa4ba;margin-bottom:8px}
.game-button div[data-testid="stButton"] > button {text-align:center!important;justify-content:center!important;white-space:normal!important;min-height:0!important;height:auto!important;padding:9px 10px!important;line-height:1.3!important;border-radius:8px!important;margin-bottom:3px!important;font-size:.78rem!important;font-weight:650!important;letter-spacing:.005em!important}\n.game-button div[data-testid="stButton"] > button[kind="primary"] {background:linear-gradient(105deg,#0e6b55,#125844)!important;border:1px solid #35d9a2!important;border-left:4px solid #64f3c0!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;font-weight:850!important;font-size:.82rem!important;letter-spacing:.01em!important;box-shadow:0 0 0 1px rgba(53,217,162,.12),0 4px 12px rgba(0,0,0,.18)!important}\n.game-button div[data-testid="stButton"] > button[kind="primary"] p {color:#ffffff!important;font-weight:850!important}\n.game-button div[data-testid="stButton"] > button[kind="primary"]:hover {background:linear-gradient(105deg,#11785f,#17694f)!important;border-color:#73f4c5!important}
div[data-testid="stButton"] > button {border:1px solid #2a3d51;background:#121e2b;color:#eaf2fb;font-weight:650;justify-content:flex-start!important;text-align:left!important;transition:background .15s,border-color .15s}
div[data-testid="stButton"] > button:hover {border-color:#68c4ff;background:#182a3c;color:#fff}
div[data-testid="stButton"] > button[kind="primary"] {background:#105b49;border-color:#35d49a;color:#fff!important;-webkit-text-fill-color:#fff!important;font-weight:800;box-shadow:inset 3px 0 0 #72f0c4,0 2px 8px rgba(0,0,0,.12)}
.st-key-save_prediction_area {background:linear-gradient(100deg,rgba(17,53,52,.7),rgba(16,27,40,.98) 72%);border:1px solid #2b665c;border-left:4px solid #35d9a2;border-radius:10px;padding:12px 14px 13px;margin-top:10px;margin-bottom:12px}
.st-key-save_prediction_area [data-testid="stMarkdownContainer"] p {margin-bottom:4px}
.st-key-save_prediction_area div[data-testid="stButton"] > button {background:linear-gradient(100deg,#147b63,#176b60)!important;border:1px solid #45d8ad!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;font-size:.9rem!important;font-weight:800!important;letter-spacing:.005em!important;min-height:46px!important;height:46px!important;padding:10px 14px!important;border-radius:8px!important;justify-content:center!important;text-align:center!important;box-shadow:0 3px 10px rgba(0,0,0,.16)!important}
.st-key-save_prediction_area div[data-testid="stButton"] > button:hover {background:linear-gradient(100deg,#199575,#1b8171)!important;border-color:#83f2d0!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;transform:translateY(-1px)}

div[data-testid="stDateInput"] input {background:#eaf1f8!important;color:#182331!important;border-radius:7px!important;min-height:36px}
div[data-testid="stDateInput"] label {font-size:.75rem!important;font-weight:700!important}
div[data-testid="stMetric"] {background:#0c1520;border:1px solid #26384b;border-radius:8px;padding:7px 9px;min-height:88px;box-sizing:border-box}
div[data-testid="stMetricLabel"] {font-size:.72rem!important;color:#9db0c4!important}
div[data-testid="stMetricValue"] {font-size:1.2rem!important;color:#f5f9ff!important;font-weight:850}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {margin-top:3px}
.market-card {border:1px solid #285743;border-radius:8px;padding:9px 10px;background:linear-gradient(140deg,#14251f,#101923);margin-bottom:5px;min-height:78px;box-sizing:border-box}
.market-card.total {border-color:#2a5277;background:linear-gradient(140deg,#142337,#101923)}
.market-kicker {font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;font-weight:850;color:#73dcb5}
.market-card.total .market-kicker {color:#79caff}
.market-title {font-size:1rem;font-weight:850;color:#fff;margin-top:3px}
.market-desc {font-size:.72rem;color:#91a5ba;margin-top:2px}
.match-header {border:1px solid #3b607a;border-left:4px solid #68c4ff;border-radius:10px;padding:13px 15px;background:linear-gradient(105deg,#172c40 0%,#122131 62%,#101a26 100%);margin:6px 0 10px;box-shadow:0 5px 18px rgba(0,0,0,.13)}
.match-title {font-size:1.35rem;line-height:1.35;font-weight:900;letter-spacing:-.025em;color:#fff}
.match-meta {font-size:.76rem;color:#b5c8d9;margin-top:6px;line-height:1.4}
.small-note {font-size:.69rem;color:#8195a9;line-height:1.4}
[data-testid="stAlert"] {padding:.55rem .75rem!important;border-radius:8px!important}
[data-testid="stNumberInput"] button {background:#eaf1f8!important;color:#182331!important;border:0!important}
.empty-state {padding:24px 14px;text-align:center;border:1px dashed #33485c;border-radius:10px;color:#9aafc4;background:#101923}
div[data-testid="stNumberInput"] input {font-size:.85rem}
[data-testid="stVerticalBlock"] {gap:.35rem}
[data-testid="stHorizontalBlock"] {gap:.6rem}
.stColumn:has(.market-card) {background:linear-gradient(180deg,rgba(17,29,42,.82),rgba(12,21,31,.78));border:1px solid #26384b;border-radius:11px;padding:9px 10px 8px;box-sizing:border-box}
.stColumn:has(.market-card) .market-card {margin-top:0;margin-bottom:7px;min-height:76px}
.stColumn:has(.market-card) [data-testid="stMetric"] {height:100%;box-sizing:border-box}
.stColumn:has(.market-card) label {font-size:.74rem!important}
.edge-box {padding:9px 11px;border-radius:8px;font-size:.82rem;margin:4px 0 7px;border:1px solid #25473e;background:#0c241f;color:#9de8ca}
.edge-box.neutral {border-color:#344454;background:#111c28;color:#b8c7d7}
.outcome-box {padding:8px 9px;border-radius:8px;border:1px solid #285743;background:#10231f;min-height:58px;box-sizing:border-box}
.outcome-box.under {border-color:#2a5277;background:#111f2d}
.outcome-title {font-size:.78rem;font-weight:850;letter-spacing:.02em}
.outcome-sub {font-size:.68rem;color:#91a5ba;margin-top:3px}
.result-line {font-size:.76rem;padding:7px 9px;margin-top:4px;border-radius:7px;background:#0c241f;color:#9de8ca}
.result-line.neutral {background:#111c28;color:#9aafc4}
.final-score-panel {
  border:1px solid #2c5878; border-left:5px solid #55ddb0; border-radius:12px;
  padding:14px 18px; margin:8px 0 12px; background:linear-gradient(110deg,#10291f,#101d2c 75%);
}
.final-score-kicker {font-size:.72rem;letter-spacing:.12em;font-weight:900;color:#6ee7b7;margin-bottom:10px}
.final-score-teams {display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:14px}
.final-team {display:flex;align-items:center;justify-content:space-between;gap:10px;color:#dce8f4;font-weight:750}
.final-team span {font-size:1rem}
.final-team strong {font-size:2rem;line-height:1;color:#fff;font-weight:900}
.final-vs {font-size:1.5rem;color:#7790a8;font-weight:800}
.final-total {border-top:1px solid #294456;margin-top:12px;padding-top:9px;display:flex;justify-content:space-between;color:#9eb4c9;font-size:.72rem;letter-spacing:.06em;font-weight:800}
.final-total strong {color:#fff;font-size:.95rem}
.market-outcome {border:1px solid #2b4054;border-radius:10px;padding:12px 14px;margin:3px 0 12px;min-height:124px;background:#101c29}
.outcome-heading {font-size:.72rem;letter-spacing:.08em;font-weight:900;color:#9ab0c6}
.outcome-pick {font-size:.95rem;font-weight:750;color:#eef5fb;margin:7px 0}
.outcome-result {font-size:1rem;font-weight:950;letter-spacing:.03em;padding:7px 10px;border-radius:7px;display:inline-block;margin:2px 0 7px}
.outcome-detail {font-size:.76rem;color:#a8bacb}
.market-outcome.won {border-color:#239b72;background:linear-gradient(130deg,#103b30,#10241f)}
.market-outcome.won .outcome-result {color:#09291f;background:#66e0ae}
.market-outcome.lost {border-color:#b84e59;background:linear-gradient(130deg,#401e29,#241922)}
.market-outcome.lost .outcome-result {color:#fff;background:#c84f5b}
.market-outcome.push {border-color:#b58a35;background:linear-gradient(130deg,#3b3018,#211f1a)}
.market-outcome.push .outcome-result {color:#2b2109;background:#e7c36b}
.market-outcome.pending .outcome-result {color:#d8e5f1;background:#304356}
div[data-testid="stAlert"] {margin:.25rem 0!important}

@media(max-width:900px){.block-container{padding:.65rem}.status{display:none}}
.st-key-nav_Ver\\ calendario, .st-key-nav_Ver\\ apostas, .st-key-nav_Ver\\ ROI, .st-key-nav_Estatísticas, .st-key-nav_Adicionar\\ aposta, .st-key-nav_Histórico {margin-bottom:3px}
/* Left navigation: consistent centered labels and icons */
.st-key-nav_Ver\\ calendario div[data-testid="stButton"] > button,
.st-key-nav_Ver\\ apostas div[data-testid="stButton"] > button,
.st-key-nav_Ver\\ ROI div[data-testid="stButton"] > button,
.st-key-nav_Estatísticas div[data-testid="stButton"] > button,
.st-key-nav_Adicionar\\ aposta div[data-testid="stButton"] > button,
.st-key-nav_Histórico div[data-testid="stButton"] > button {
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    text-align:center!important;
    width:100%!important;
    padding-left:8px!important;
    padding-right:8px!important;
}
.st-key-nav_Ver\\ calendario div[data-testid="stButton"] > button p,
.st-key-nav_Ver\\ apostas div[data-testid="stButton"] > button p,
.st-key-nav_Ver\\ ROI div[data-testid="stButton"] > button p,
.st-key-nav_Estatísticas div[data-testid="stButton"] > button p,
.st-key-nav_Adicionar\\ aposta div[data-testid="stButton"] > button p,
.st-key-nav_Histórico div[data-testid="stButton"] > button p {
    width:100%!important;
    text-align:center!important;
    margin:0!important;
}
\n/* Receipt toggle: quiet text action, visually subordinate to the ticket */
div[class*="st-key-select_slip_"] div[data-testid="stButton"] > button {
    width:auto!important;
    min-height:22px!important;
    height:22px!important;
    padding:0 2px!important;
    margin-top:5px!important;
    border:0!important;
    border-radius:0!important;
    background:transparent!important;
    color:#80cfff!important;
    -webkit-text-fill-color:#80cfff!important;
    box-shadow:none!important;
    font-size:.64rem!important;
    font-weight:850!important;
    letter-spacing:.055em!important;
    justify-content:flex-start!important;
}
div[class*="st-key-select_slip_"] div[data-testid="stButton"] > button:hover {
    background:transparent!important;
    color:#d2efff!important;
    -webkit-text-fill-color:#d2efff!important;
    text-decoration:underline!important;
}


/* Per-slip delete action: discreet until hover */
div[class*="st-key-delete_slip_"] div[data-testid="stButton"] > button {
    min-width:30px!important;
    width:30px!important;
    height:30px!important;
    min-height:30px!important;
    padding:0!important;
    border:1px solid #59313a!important;
    border-radius:8px!important;
    background:rgba(74,31,40,.28)!important;
    color:#ff929b!important;
    -webkit-text-fill-color:#ff929b!important;
    justify-content:center!important;
    text-align:center!important;
    font-size:.85rem!important;
}
div[class*="st-key-delete_slip_"] div[data-testid="stButton"] > button:hover {
    border-color:#e06b78!important;
    background:#552630!important;
    color:#fff!important;
    -webkit-text-fill-color:#fff!important;
}
/* History receipt cards: each ticket is a distinct, elevated surface */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="st-key-select_slip_"]) {
    position:relative!important;
    background:linear-gradient(112deg,#172b3d 0%,#132333 48%,#101c29 100%)!important;
    border:1px solid #36536b!important;
    border-radius:13px!important;
    box-shadow:
        0 9px 24px rgba(0,0,0,.30),
        0 2px 5px rgba(0,0,0,.22),
        inset 0 1px 0 rgba(210,235,255,.075),
        inset 0 0 0 1px rgba(90,145,180,.055)!important;
    padding:8px 11px!important;
    margin:0 0 13px 0!important;
    overflow:hidden!important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[class*="st-key-select_slip_"])::before {
    content:"";
    position:absolute;
    left:0;
    top:10px;
    bottom:10px;
    width:3px;
    border-radius:0 4px 4px 0;
    background:linear-gradient(180deg,#54b9e8,#347da8);
    opacity:.9;
}
div[class*="st-key-select_slip_"] div[data-testid="stButton"] > button {
    border:1px solid #294d67!important;
    border-radius:7px!important;
    background:rgba(37,91,125,.16)!important;
    color:#91d3ff!important;
    -webkit-text-fill-color:#91d3ff!important;
    font-size:.74rem!important;
    font-weight:750!important;
    min-height:29px!important;
}
div[class*="st-key-select_slip_"] div[data-testid="stButton"] > button:hover {
    background:rgba(53,127,170,.25)!important;
    border-color:#4b8bb2!important;
}

/* Modern receipt table: spreadsheet clarity with the NBA LAB dark theme */
.receipt-table-wrap {
    width:100%;
    overflow-x:auto;
    border:1px solid #263e52;
    border-radius:10px;
    background:#0f1a25;
    margin:4px 0 12px 0;
}
.receipt-table {
    width:100%;
    min-width:760px;
    border-collapse:separate;
    border-spacing:0;
    color:#dce7f1;
    font-size:.82rem;
}
.receipt-table thead th {
    padding:11px 13px;
    background:#172838;
    color:#8fa9be;
    font-size:.66rem;
    font-weight:850;
    letter-spacing:.075em;
    text-align:left;
    white-space:nowrap;
    border-bottom:1px solid #2b4358;
}
.receipt-table thead th.right { text-align:right; }
.receipt-table tbody td {
    padding:12px 13px;
    background:#111e2b;
    border-bottom:1px solid #203448;
    vertical-align:middle;
    white-space:nowrap;
}
.receipt-table tbody tr:nth-child(even) td { background:#142332; }
.receipt-table tbody tr:last-child td { border-bottom:0; }
.receipt-table td.date { color:#9fb5c8; font-variant-numeric:tabular-nums; }
.receipt-table td.match { color:#f0f5fa; font-weight:750; }
.receipt-table td.match .at { color:#66849d; padding:0 8px; font-weight:600; }
.receipt-table td.num { text-align:right; font-variant-numeric:tabular-nums; color:#d7e4ef; }
.receipt-table td.odds { color:#9bd7ff; font-weight:800; }
.receipt-table .market {
    display:inline-block;
    padding:4px 9px;
    border-radius:6px;
    font-size:.74rem;
    font-weight:800;
    letter-spacing:.01em;
    white-space:nowrap;
}
.receipt-table .market-over {
    background:#15364a;
    border:1px solid #28749a;
    color:#8edaff;
}
.receipt-table .market-under {
    background:#3b2d18;
    border:1px solid #8a642b;
    color:#ffd27a;
}
.receipt-table .market-moneyline {
    background:#302348;
    border:1px solid #70539d;
    color:#d7c0ff;
}
.receipt-table .market-other {
    background:#263341;
    border:1px solid #465b70;
    color:#c6d3df;
}
.receipt-table .result {
    display:inline-block;
    padding:4px 9px;
    border-radius:6px;
    font-size:.71rem;
    font-weight:850;
    white-space:nowrap;
}

/* Exportações: botões compactos, em formato pill, com acabamento premium */
.st-key-export_slips_compact button,
.st-key-export_legs_compact button,
.st-key-export_full_backup_compact button {
    width:100%!important;
    min-height:36px!important;
    height:36px!important;
    padding:6px 10px!important;
    border-radius:999px!important;
    justify-content:center!important;
    text-align:center!important;
    white-space:nowrap!important;
    font-size:.72rem!important;
    font-weight:800!important;
    letter-spacing:.01em!important;
    color:#dcecff!important;
    -webkit-text-fill-color:#dcecff!important;
    background:linear-gradient(135deg,#17283a,#101b29)!important;
    border:1px solid #34506a!important;
    box-shadow:0 3px 10px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.035)!important;
    transition:all .16s ease!important;
}
.st-key-export_slips_compact button:hover,
.st-key-export_legs_compact button:hover,
.st-key-export_full_backup_compact button:hover {
    transform:translateY(-1px)!important;
    border-color:#68c4ff!important;
    background:linear-gradient(135deg,#203c55,#152a3d)!important;
    color:#ffffff!important;
    -webkit-text-fill-color:#ffffff!important;
    box-shadow:0 5px 14px rgba(0,0,0,.24),0 0 0 1px rgba(104,196,255,.12)!important;
}
.st-key-export_full_backup_compact button {
    background:linear-gradient(135deg,#16483f,#12352f)!important;
    border-color:#287b68!important;
    color:#8df0cf!important;
    -webkit-text-fill-color:#8df0cf!important;
}
.st-key-export_full_backup_compact button:hover {
    background:linear-gradient(135deg,#1b6656,#16483f)!important;
    border-color:#48d6ad!important;
    color:#ffffff!important;
    -webkit-text-fill-color:#ffffff!important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="ball">🏀</div>
  <div><h1>NBA LAB <span style="color:#68c4ff">PRO</span></h1>
  <div class="brand-sub">Calendário e análise de mercados · espaço de trabalho compacto</div></div>
  <div class="status">● PRÉ-JOGO · NBA</div>
</div>
""", unsafe_allow_html=True)

try:
    API_KEY = st.secrets["API_SPORTS_NBA_KEY"]
except Exception:
    API_KEY = ""

@st.cache_resource(show_spinner=False)
def load_m1_base_engine(history_path, history_mtime):
    """Train M1 once from the validated 2021-2025 history and retain its initial state."""
    return M1Live(history_path)

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_m1_full_season(season_start_year, api_key):
    """Fetch the complete current NBA season once; results are filtered locally by ET day."""
    try:
        response = requests.get(
            "https://v2.nba.api-sports.io/games",
            headers={"x-apisports-key": api_key},
            params=api_season_query(int(season_start_year)),
            timeout=(5, 20),
        )
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:180]}"
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("errors"):
            return None, f"Resposta API-Sports inválida: {(payload or {}).get('errors', 'payload inválido')}"
        if not isinstance(payload.get("response"), list):
            return None, "Resposta API-Sports sem lista de jogos."
        return payload, None
    except (requests.RequestException, ValueError, TypeError) as exc:
        return None, str(exc)

def predict_moneyline_m1(home, away, kickoff):
    """Predict with a fresh same-day M1 state, applying only earlier ET-day regular-season finals."""
    if not API_KEY:
        raise RuntimeError("Chave API_SPORTS_NBA_KEY não configurada; M1 live indisponível.")
    if kickoff is None:
        raise RuntimeError("Data/hora do jogo indisponível para calcular M1.")
    kickoff_ts = pd.Timestamp(kickoff)
    if kickoff_ts.tzinfo is None:
        raise RuntimeError("A data/hora do jogo não contém fuso horário.")
    target_day = kickoff_ts.tz_convert("America/New_York").date()
    season_start_year = target_day.year if target_day.month >= 10 else target_day.year - 1
    history_path = str(Path(__file__).with_name("nba_games_history.csv"))
    if not Path(history_path).exists():
        raise RuntimeError("Ficheiro nba_games_history.csv não encontrado.")
    baseline = load_m1_base_engine(history_path, Path(history_path).stat().st_mtime)
    payload, api_error = fetch_m1_full_season(season_start_year, API_KEY)
    if api_error or payload is None:
        raise RuntimeError(f"Não foi possível atualizar o estado M1 da época {season_start_year}: {api_error}")
    season_games = parse_api_sports_payload(payload, baseline)
    # The 2026-27 regular season began on 20 October 2026. Exclude preseason and
    # never apply games from the target ET date or later (same-day leakage guard).
    regular_start = datetime.date(season_start_year, 10, 20)
    regular_end = datetime.date(season_start_year + 1, 4, 11) if season_start_year == 2026 else None
    # The 2026 Emirates NBA Cup Championship (11 Dec) is excluded from standings
    # and therefore must not update the M1 team state; other Cup games count.
    cup_final_days = {datetime.date(2026, 12, 11)} if season_start_year == 2026 else set()
    eligible = []
    for game in season_games:
        game_day = baseline._day(game.date)
        if game_day < regular_start or game_day >= target_day:
            continue
        if regular_end is not None and game_day > regular_end:
            continue
        if game_day in cup_final_days:
            continue
        if str(game.status).lower() in {"final", "finished", "completed", "3"}:
            eligible.append(game)
    engine = copy.deepcopy(baseline)
    if eligible:
        engine.apply_completed_games(eligible)
    return engine.predict(kickoff_ts, home, away)

@st.cache_data(ttl=90, show_spinner=False)
def fetch_nba_api_date(api_date_str, season, api_key):
    """Fetch one NBA date once per short cache window; fail fast if API-Sports stalls."""
    try:
        response = requests.get(
            "https://v2.nba.api-sports.io/games",
            headers={"x-apisports-key": api_key},
            params={"date": api_date_str, "season": str(season), "league": "standard"},
            timeout=(3.5, 7)
        )
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:180]}"
        payload = response.json()
        if payload.get("errors"):
            return None, f"API-Sports: {payload['errors']}"
        return payload, None
    except requests.RequestException as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, f"Resposta inválida: {exc}"

MEDIAS_REAIS = {
    "Atlanta Hawks":116.5,"Boston Celtics":120.8,"Brooklyn Nets":111.2,"Charlotte Hornets":108.5,
    "Chicago Bulls":112.3,"Cleveland Cavaliers":114.8,"Dallas Mavericks":117.9,"Denver Nuggets":115.4,
    "Detroit Pistons":110.1,"Golden State Warriors":118.2,"Houston Rockets":113.5,"Indiana Pacers":121.2,
    "LA Clippers":115.1,"Los Angeles Lakers":116.8,"Memphis Grizzlies":113.8,"Miami Heat":110.5,
    "Milwaukee Bucks":119.4,"Minnesota Timberwolves":113.2,"New Orleans Pelicans":115.0,
    "New York Knicks":112.8,"Oklahoma City Thunder":119.5,"Orlando Magic":110.4,
    "Philadelphia 76ers":114.2,"Phoenix Suns":116.2,"Portland Trail Blazers":106.8,
    "Sacramento Kings":116.6,"San Antonio Spurs":112.1,"Toronto Raptors":112.5,
    "Utah Jazz":115.7,"Washington Wizards":113.1
}

# Calendar date persists across navigation so reconciliation can continue in the background.
def persist_matchday():
    st.session_state["nba_matchday"] = st.session_state["nba_matchday_widget"]

selected_date = st.session_state.get("nba_matchday", datetime.date.today())
date_str = selected_date.strftime("%Y-%m-%d")

def normalize_game_status(status_obj):
    """Normalize API-Sports status variants into a small stable set."""
    status_obj = status_obj or {}
    short = str(status_obj.get("short") or "").strip().upper()
    long_status = str(status_obj.get("long") or "").strip().lower()
    if short in {"FT", "AOT", "FTOT", "FINAL", "FINISHED", "ENDED"} or any(
        word in long_status for word in ("finished", "final", "game over", "ended", "terminé", "terminado")
    ):
        return "FT"
    if short in {"Q1", "Q2", "Q3", "Q4", "OT", "HT", "LIVE", "IN"} or any(
        word in long_status for word in ("in play", "live", "quarter", "halftime", "overtime", "en cours")
    ):
        return short if short in {"Q1", "Q2", "Q3", "Q4", "OT", "HT"} else "LIVE"
    if short in {"NS", "TBD", "SCHED", "PST"} or any(
        word in long_status for word in ("not started", "scheduled", "postponed", "time to be defined")
    ):
        return "NS"
    return short or ("FT" if "final" in long_status else "NS")

def normalize_score(value):
    """Handle both numeric scores and nested API-Sports score objects."""
    if isinstance(value, dict):
        for key in ("points", "total", "score"):
            if value.get(key) is not None:
                return normalize_score(value[key])
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# Load fixtures for the chosen date, checking the previous API date too.
games = []
api_debug = []

if API_KEY:
    dates_to_query = [
        selected_date - datetime.timedelta(days=1),
        selected_date
    ]
    seen_game_ids = set()

    for api_date in dates_to_query:
        api_date_str = api_date.strftime("%Y-%m-%d")
        season = str(
            api_date.year if api_date.month >= 10 else api_date.year - 1
        )

        try:
            payload, api_error = fetch_nba_api_date(api_date_str, season, API_KEY)
            if api_error:
                api_debug.append({
                    "data_consultada": api_date_str,
                    "erro": api_error
                })
                st.warning(
                    f"API NBA temporariamente indisponível para {api_date_str}: "
                    f"{api_error}. A aplicação continua disponível; tenta novamente daqui a pouco."
                )
                continue

            raw_games = payload.get("response", [])
            debug_games = []

            for item in raw_games:
                teams = item.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("visitors") or teams.get("away") or {}
                start_time = (item.get("date") or {}).get("start")

                if not home.get("name") or not away.get("name") or not start_time:
                    continue

                local_dt = (
                    pd.to_datetime(start_time, utc=True)
                    .tz_convert("Europe/Lisbon")
                )

                debug_games.append({
                    "ID": item.get("id"),
                    "Jogo": f"{home['name']} vs {away['name']}",
                    "Data API": start_time,
                    "Data Portugal": local_dt.strftime("%d/%m/%Y %H:%M"),
                    "Estado": (item.get("status") or {}).get("long", "")
                })

                # Keep only games belonging to the selected date in Portugal.
                if local_dt.date() != selected_date:
                    continue

                game_id = str(
                    item.get("id") or
                    f"{home['name']}-{away['name']}-{start_time}"
                )

                if game_id in seen_game_ids:
                    continue

                status_obj = item.get("status") or {}
                scores_obj = item.get("scores") or {}

                home_score = normalize_score(scores_obj.get("home"))
                away_score = normalize_score(
                    scores_obj.get("visitors") or scores_obj.get("away")
                )

                games.append({
                    "id": game_id,
                    "home": home["name"],
                    "away": away["name"],
                    "time": local_dt.strftime("%H:%M"),
                    "timestamp": local_dt.timestamp(),
                    "status": normalize_game_status(status_obj),
                    "status_long": str(status_obj.get("long") or ""),
                    "home_score": home_score,
                    "away_score": away_score
                })

                seen_game_ids.add(game_id)

            api_debug.append({
                "data_consultada": api_date_str,
                "total_recebido": len(raw_games),
                "jogos": debug_games
            })

        except requests.RequestException as exc:
            api_debug.append({"data_consultada": api_date_str, "erro": str(exc)})
            st.error(f"Erro de ligação à API ({api_date_str}): {exc}")
        except ValueError as exc:
            api_debug.append({"data_consultada": api_date_str, "erro": str(exc)})
            st.error(f"Resposta inválida da API ({api_date_str}): {exc}")




games.sort(key=lambda g: g["timestamp"])


def parse_api_game(item):
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("visitors") or teams.get("away") or {}
    start_time = (item.get("date") or {}).get("start")
    if not home.get("name") or not away.get("name") or not start_time:
        return None
    local_dt = pd.to_datetime(start_time, utc=True).tz_convert("Europe/Lisbon")
    status_obj = item.get("status") or {}
    scores_obj = item.get("scores") or {}
    home_score = normalize_score(scores_obj.get("home"))
    away_score = normalize_score(scores_obj.get("visitors") or scores_obj.get("away"))
    return {
        "id": str(item.get("id", f"{home['name']}-{away['name']}-{start_time}")),
        "home": home["name"], "away": away["name"],
        "time": local_dt.strftime("%H:%M"), "timestamp": local_dt.timestamp(),
        "status": normalize_game_status(status_obj),
        "status_long": str(status_obj.get("long") or ""),
        "home_score": home_score, "away_score": away_score
    }

# Load persistent records before fetching dates needed to enrich old bet slips.
log_df = load_log()
bets_df = load_bets()
market_drafts = load_market_drafts()
sync_bet_csvs_from_github()
slips_df = load_slips()
legs_df = load_legs()
result_games_by_id = {normalize_game_id(g["id"]): g for g in games}
if API_KEY:
    unresolved_dates = set()
    if not log_df.empty:
        unresolved_dates.update(
            str(d) for d in log_df.loc[
                ~log_df["game_status"].astype(str).str.lower().eq("final"), "game_date"
            ].dropna().tolist() if str(d) and str(d) != date_str
        )
    if not bets_df.empty:
        unresolved_dates.update(
            str(d) for d in bets_df.loc[
                ~bets_df["result"].isin(["Acertou", "Falhou", "Push"]), "game_date"
            ].dropna().tolist() if str(d) and str(d) != date_str
        )
    if not legs_df.empty:
        missing_scores = legs_df["away_score"].isna() | legs_df["home_score"].isna()
        unresolved_dates.update(
            str(d) for d in legs_df.loc[missing_scores, "game_date"].dropna().tolist()
            if str(d) and str(d) != date_str
        )
    unresolved_dates = sorted(unresolved_dates)
    # Query only dates represented in the log, rather than re-fetching the entire season.
    for historical_date in unresolved_dates:
        try:
            hist_dt = datetime.date.fromisoformat(historical_date)
            hist_season = str(hist_dt.year if hist_dt.month >= 10 else hist_dt.year - 1)
            hist_payload, hist_error = fetch_nba_api_date(
                historical_date, hist_season, API_KEY
            )
            if hist_error or not hist_payload:
                continue
            for item in hist_payload.get("response", []):
                parsed = parse_api_game(item)
                if parsed:
                    result_games_by_id[str(parsed["id"])] = parsed
        except (requests.RequestException, ValueError, TypeError):
            continue

# Reconcile real bet slips independently from the prediction log.
slips_df, legs_df, migrated_slips = migrate_legacy_bets(slips_df, legs_df)
slips_df, legs_df, reconciled_slips = reconcile_slips(slips_df, legs_df, result_games_by_id)
if migrated_slips or reconciled_slips:
    save_slip_data(slips_df, legs_df)

# Automatic M1/M2 prediction freeze and result reconciliation. Runs whenever the app reruns.
log_df, new_snapshots = auto_freeze_predictions(games, log_df)
# Keep the statistical forecast immutable: total_line=220.5 is a fixed
# reference line for model evaluation. User-selected market lines stay in drafts
# and, when an actual wager is placed, are recorded in the separate bet ledger.
if new_snapshots and not log_df.empty:
    for idx, row in log_df.tail(new_snapshots).iterrows():
        draft = market_drafts.get(normalize_game_id(row.get("game_id")), {})
        for source_key in ("home_market_odds", "away_market_odds"):
            value = draft.get(source_key)
            if pd.notna(value):
                target_key = source_key
                log_df.at[idx, target_key] = float(value)
log_df, recovered_snapshots = recover_predictions_from_saved_drafts(
    list(result_games_by_id.values()), log_df, market_drafts
)
log_df, updated_results = upsert_api_results(list(result_games_by_id.values()), log_df)
if new_snapshots or recovered_snapshots or updated_results:
    save_log(log_df)

current_ids = {g["id"] for g in games}
if st.session_state.get("nba_selection_date") != date_str:
    st.session_state["nba_selection_date"] = date_str
    st.session_state["nba_selected_game"] = games[0]["id"] if games else None
elif st.session_state.get("nba_selected_game") not in current_ids:
    st.session_state["nba_selected_game"] = games[0]["id"] if games else None

active_pages = ["Ver calendário", "Ver apostas", "Ver ROI", "Estatísticas", "Adicionar aposta", "Histórico"]
if "nba_active_page" not in st.session_state:
    st.session_state["nba_active_page"] = "Ver calendário"
active_page = st.session_state["nba_active_page"]
if "bet_slip_draft_legs" not in st.session_state:
    st.session_state["bet_slip_draft_legs"] = []
if "manage_slip_id" not in st.session_state:
    st.session_state["manage_slip_id"] = None
nav_area, main_area = st.columns([1, 5.2], gap="large")
with nav_area:
    st.markdown("#### MENU")
    st.caption("Navegação")
    nav_icons = {"Ver calendário": "📅", "Ver apostas": "🎟️", "Ver ROI": "📈", "Estatísticas": "📊", "Adicionar aposta": "＋", "Histórico": "🗂️"}
    for page_name in active_pages:
        if st.button(f"{nav_icons[page_name]}  {page_name}", key=f"nav_{page_name}", type="primary" if active_page == page_name else "secondary", use_container_width=True):
            st.session_state["nba_active_page"] = page_name
            st.rerun()



# ---------------------------------------------------------------------------
# RELATÓRIO DIÁRIO NBA LAB PRO — captura pré-jogo e atualização pós-jogo.
# Esta secção é isolada: não altera os CSVs operacionais nem os motores M1/M2.
# ---------------------------------------------------------------------------
DAILY_REPORT_DIR = "nba_daily_reports"
DUNKAPI_ODDS_URL = "https://mcp.dunkapi.com/v1/get_odds"
DAILY_REFERENCE_TOTAL_LINE = 220.5


def _daily_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _daily_local_report_path(filename):
    folder = Path(__file__).with_name(DAILY_REPORT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def _daily_github_read_bytes(path):
    if not github_configured():
        return None
    repo = st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    response = requests.get(
        f"{GITHUB_API}/repos/{repo}/contents/{path}",
        headers=github_read_headers(),
        params={"ref": branch},
        timeout=20,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    import base64
    return base64.b64decode(response.json()["content"])


def _daily_read_json(filename):
    path = f"{DAILY_REPORT_DIR}/{filename}"
    if github_configured():
        content = _daily_github_read_bytes(path)
        if content is None:
            return None
        return json.loads(content.decode("utf-8"))
    local_path = _daily_local_report_path(filename)
    if not local_path.exists():
        return None
    return json.loads(local_path.read_text(encoding="utf-8"))


def _daily_create_json(filename, payload):
    """Create-only persistence: never overwrite an existing pre-game snapshot."""
    path = f"{DAILY_REPORT_DIR}/{filename}"
    content = _daily_json_bytes(payload)
    if github_configured():
        existing = _daily_github_read_bytes(path)
        if existing is not None:
            return json.loads(existing.decode("utf-8")), False
        import base64
        repo = st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        response = requests.put(
            f"{GITHUB_API}/repos/{repo}/contents/{path}",
            headers=github_headers(),
            json={
                "message": f"Add NBA daily report {filename}",
                "content": base64.b64encode(content).decode("ascii"),
                "branch": branch,
            },
            timeout=30,
        )
        if response.status_code in (409, 422):
            existing = _daily_github_read_bytes(path)
            if existing is not None:
                return json.loads(existing.decode("utf-8")), False
        response.raise_for_status()
        return payload, True

    local_path = _daily_local_report_path(filename)
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8")), False
    local_path.write_bytes(content)
    return payload, True


def _daily_list_pending_reports():
    """Return pre-game captures without a corresponding results archive."""
    if github_configured():
        repo = st.secrets.get("GITHUB_REPOSITORY", "lmdpires/nba-lab-pro")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        response = requests.get(
            f"{GITHUB_API}/repos/{repo}/contents/{DAILY_REPORT_DIR}",
            headers=github_read_headers(),
            params={"ref": branch},
            timeout=20,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json() if item.get("type") == "file"]
    else:
        folder = Path(__file__).with_name(DAILY_REPORT_DIR)
        names = [p.name for p in folder.glob("*.json")] if folder.exists() else []

    pregame_dates = sorted(
        name[:-len("_pregame.json")]
        for name in names
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}_pregame\.json", name)
    )
    return [
        day for day in pregame_dates
        if f"{day}_results.json" not in names
    ]


def _daily_save_results(filename, payload):
    """Results are a separate immutable file; the original pre-game file is untouched."""
    return _daily_create_json(filename, payload)


def _daily_get_dunk_key():
    try:
        return str(st.secrets.get("DUNKAPI_KEY", "") or "").strip()
    except Exception:
        return ""


def _daily_fetch_dunk_odds(et_dates):
    """One POST per distinct Eastern date; no automatic retries."""
    api_key = _daily_get_dunk_key()
    if not api_key:
        raise RuntimeError("DUNKAPI_KEY não encontrada nos Secrets do Streamlit Cloud.")
    responses = []
    for et_day in sorted(set(et_dates)):
        response = requests.post(
            DUNKAPI_ODDS_URL,
            headers={"x-api-key": api_key, "Accept": "application/json"},
            json={"date": et_day},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"DunkAPI devolveu HTTP {response.status_code} para {et_day}: "
                f"{response.text[:250]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"DunkAPI devolveu JSON inválido para {et_day}.") from exc
        if isinstance(payload, dict) and payload.get("errors"):
            raise RuntimeError(f"DunkAPI devolveu erros para {et_day}: {payload['errors']}")
        responses.append({
            "query_date_et": et_day,
            "requested_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "estimated_credits": 10,
            "payload": payload,
        })
    return responses


def _daily_flatten_odds(raw_responses, games_by_id):
    """Flatten bookmaker records where possible, while retaining the untouched raw payload."""
    records = []
    wanted = (
        "home_ml_open", "home_ml_close", "away_ml_open", "away_ml_close",
        "captured_at", "provider", "game_id",
    )

    def walk(node, inherited=None):
        inherited = dict(inherited or {})
        if isinstance(node, list):
            for child in node:
                walk(child, inherited)
            return
        if not isinstance(node, dict):
            return

        merged = dict(inherited)
        for key in ("game_id", "id", "home_team", "away_team", "home", "away",
                    "starts_at", "provider", "bookmaker", *wanted):
            if key in node and node[key] is not None:
                merged[key] = node[key]

        has_odds = any(k in node for k in (
            "home_ml_open", "home_ml_close", "away_ml_open", "away_ml_close"
        ))
        if has_odds:
            gid = str(merged.get("game_id", merged.get("id", "")))
            matched = games_by_id.get(gid, {})
            starts_at = merged.get("starts_at") or matched.get("starts_at")
            captured_at = merged.get("captured_at")
            before_start = None
            minutes_before = None
            try:
                start_ts = pd.to_datetime(starts_at, utc=True)
                capture_ts = pd.to_datetime(captured_at, utc=True)
                minutes_before = (start_ts - capture_ts).total_seconds() / 60
                before_start = bool(capture_ts < start_ts)
            except Exception:
                pass

            provider = merged.get("provider", merged.get("bookmaker", ""))
            if isinstance(provider, dict):
                provider = provider.get("name") or provider.get("title") or provider.get("id")
            records.append({
                "game_id": gid,
                "home_team": merged.get("home_team", matched.get("home_team", "")),
                "away_team": merged.get("away_team", matched.get("away_team", "")),
                "starts_at": starts_at,
                "provider": provider,
                "home_ml_open": merged.get("home_ml_open"),
                "home_ml_close": merged.get("home_ml_close"),
                "away_ml_open": merged.get("away_ml_open"),
                "away_ml_close": merged.get("away_ml_close"),
                "captured_at": captured_at,
                "captured_before_start": before_start,
                "minutes_before_start": minutes_before,
            })

        for key, value in node.items():
            if isinstance(value, (dict, list)):
                child_context = dict(merged)
                if key.lower() in {"bookmaker", "bookmakers", "provider", "providers"} and isinstance(value, dict):
                    child_context.setdefault("provider", value.get("name") or value.get("title") or value.get("id"))
                walk(value, child_context)

    for response in raw_responses:
        walk(response.get("payload"))
    return records


def _daily_build_predictions(day_games, captured_at):
    rows = []
    for game in day_games:
        kickoff = datetime.datetime.fromtimestamp(
            float(game["timestamp"]), tz=datetime.timezone.utc
        )
        m1_result = predict_moneyline_m1(game["home"], game["away"], pd.Timestamp(kickoff))
        p_home = float(m1_result["p_home_m1"])
        p_away = float(m1_result["p_away_m1"])
        if not (0 <= p_home <= 1 and 0 <= p_away <= 1) or abs(p_home + p_away - 1) > 1e-8:
            raise RuntimeError(f"Probabilidades M1 inválidas no jogo {game['id']}.")

        total_reference, m2_error = predict_total(
            game["home"], game["away"], kickoff=pd.Timestamp(kickoff)
        )
        if total_reference is None:
            raise RuntimeError(f"M2 indisponível para {game['away']} @ {game['home']}: {m2_error}")
        total_line = DAILY_REFERENCE_TOTAL_LINE
        p_over, p_under = probability_over(total_reference, total_line)
        market_draft = load_market_drafts().get(normalize_game_id(game["id"]), {})
        rows.append({
            "captured_at_utc": captured_at,
            "game_id": str(game["id"]),
            "game_date_portugal": pd.Timestamp(kickoff).tz_convert("Europe/Lisbon").date().isoformat(),
            "game_date_et": pd.Timestamp(kickoff).tz_convert("America/New_York").date().isoformat(),
            "starts_at_utc": pd.Timestamp(kickoff).isoformat(),
            "home_team": game["home"],
            "away_team": game["away"],
            "game_status_at_capture": game.get("status"),
            "m1_version": "M1 v1.1",
            "m1_p_home": p_home,
            "m1_p_away": p_away,
            "m1_fair_home_odds": (1 / p_home) if p_home > 0 else None,
            "m1_fair_away_odds": (1 / p_away) if p_away > 0 else None,
            "m1_pick": "Casa" if p_home >= p_away else "Fora",
            "m2_version": "M2 HGB v1 frozen",
            "m2_total_points": float(total_reference),
            "m2_reference_line": total_line,
            "m2_p_over": float(p_over),
            "m2_p_under": float(p_under),
            "m2_fair_over_odds": (1 / float(p_over)) if p_over > 0 else None,
            "m2_fair_under_odds": (1 / float(p_under)) if p_under > 0 else None,
            "market_home_odds_manual": market_draft.get("home_market_odds"),
            "market_away_odds_manual": market_draft.get("away_market_odds"),
            "market_total_line_manual": market_draft.get("total_line"),
        })
    return rows


def _daily_collect_results(pregame_payload):
    """Fetch final scores from API-Sports for the games in the immutable pre-game snapshot."""
    if not API_KEY:
        raise RuntimeError("API_SPORTS_NBA_KEY não configurada.")
    games_saved = pregame_payload.get("games", [])
    et_dates = sorted({str(g["game_date_et"]) for g in games_saved})
    found = {}
    api_calls = []
    for et_day in et_dates:
        et_date = datetime.date.fromisoformat(et_day)
        season = str(et_date.year if et_date.month >= 10 else et_date.year - 1)
        payload, api_error = fetch_nba_api_date(et_day, season, API_KEY)
        if api_error or payload is None:
            raise RuntimeError(f"API-Sports falhou para {et_day}: {api_error}")
        api_calls.append({"date_et": et_day, "games_received": len(payload.get("response", []))})
        for item in payload.get("response", []):
            parsed = parse_api_game(item)
            if parsed:
                found[str(parsed["id"])] = parsed

    results = []
    unresolved = []
    for saved in games_saved:
        gid = str(saved["game_id"])
        current = found.get(gid)
        if current is None:
            unresolved.append(gid)
            continue
        status = str(current.get("status", "NS")).upper()
        is_final = status == "FT" and current.get("home_score") is not None and current.get("away_score") is not None
        is_cancelled = status in {"CANC", "CANCELLED", "POSTPONED", "PST"}
        if not is_final and not is_cancelled:
            unresolved.append(gid)
        result = {
            "game_id": gid,
            "home_team": saved["home_team"],
            "away_team": saved["away_team"],
            "status": status,
            "home_score": current.get("home_score"),
            "away_score": current.get("away_score"),
            "actual_total": (
                int(current["home_score"]) + int(current["away_score"])
                if is_final else None
            ),
            "result_updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if is_final:
            home_won = int(current["home_score"]) > int(current["away_score"])
            result["actual_home_win"] = int(home_won)
            result["actual_over_220_5"] = int(result["actual_total"] > DAILY_REFERENCE_TOTAL_LINE)
            result["m1_moneyline_correct"] = (
                (saved["m1_pick"] == "Casa" and home_won)
                or (saved["m1_pick"] == "Fora" and not home_won)
            )
            result["m2_over_under_correct"] = (
                (saved["m2_p_over"] >= 0.5 and result["actual_total"] > DAILY_REFERENCE_TOTAL_LINE)
                or (saved["m2_p_over"] < 0.5 and result["actual_total"] < DAILY_REFERENCE_TOTAL_LINE)
            ) if result["actual_total"] != DAILY_REFERENCE_TOTAL_LINE else None
            result["m1_brier"] = (float(saved["m1_p_home"]) - float(home_won)) ** 2
            result["m2_over_brier"] = (float(saved["m2_p_over"]) - float(result["actual_over_220_5"])) ** 2
        results.append(result)

    return {
        "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "api_sports_queries": api_calls,
        "results": results,
        "unresolved_game_ids": unresolved,
        "all_games_resolved": len(unresolved) == 0 and len(results) == len(games_saved),
    }


def _daily_zip_bytes(pregame_payload, odds_rows, results_payload=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report_metadata.json", _daily_json_bytes({
            "report_date": pregame_payload.get("report_date"),
            "phase": "completed" if results_payload else "pregame",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sources": ["M1 v1.1", "M2 HGB v1", "DunkAPI get_odds", "API-Sports NBA"],
            "dunkapi_estimated_credits": pregame_payload.get("dunkapi_estimated_credits", 0),
            "results_included": bool(results_payload),
        }))
        archive.writestr("daily_predictions.csv", pd.DataFrame(pregame_payload.get("predictions", [])).to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("dunkapi_odds_rows.csv", pd.DataFrame(odds_rows or []).to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("dunkapi_raw_responses.json", _daily_json_bytes(pregame_payload.get("dunkapi_responses", [])))
        archive.writestr("games_at_capture.json", _daily_json_bytes(pregame_payload.get("games", [])))
        archive.writestr("model_versions.json", _daily_json_bytes({
            "moneyline": "M1 v1.1",
            "total_points": "M2 HGB v1 frozen",
            "m2_reference_line": DAILY_REFERENCE_TOTAL_LINE,
        }))
        if results_payload:
            archive.writestr("games_results.csv", pd.DataFrame(results_payload.get("results", [])).to_csv(index=False).encode("utf-8-sig"))
            archive.writestr("results_update.json", _daily_json_bytes(results_payload))
        quality = {
            "games_captured": len(pregame_payload.get("games", [])),
            "games_excluded_after_tipoff_or_unconfirmed": len(pregame_payload.get("excluded_games", [])),
            "predictions_captured": len(pregame_payload.get("predictions", [])),
            "odds_rows": len(odds_rows or []),
            "odds_rows_with_verified_pregame_timestamp": sum(1 for r in (odds_rows or []) if r.get("captured_before_start") is True),
            "odds_rows_with_missing_or_invalid_timestamp": sum(1 for r in (odds_rows or []) if r.get("captured_before_start") is None),
            "odds_rows_captured_after_start": sum(1 for r in (odds_rows or []) if r.get("captured_before_start") is False),
            "unresolved_game_ids": results_payload.get("unresolved_game_ids", []) if results_payload else None,
        }
        archive.writestr("data_quality_report.json", _daily_json_bytes(quality))

        # Include relevant current data files as read-only copies. Never include Secrets.
        data_files = [
            "nba_prediction_log.csv", "nba_market_inputs.csv",
            "nba_bet_slips.csv", "nba_bet_selections.csv",
            "nba_actual_bets.csv", "nba_games_history.csv",
            "nba_over_m2_history.csv", "nba_over_m2_metadata.json",
        ]
        for filename in data_files:
            file_path = Path(__file__).with_name(filename)
            if file_path.exists() and file_path.is_file():
                archive.write(file_path, arcname=f"app_data/{filename}")
    return buffer.getvalue()


def _daily_report_action(selected_day, current_games):
    """Single button state machine: capture first; on a later click, resolve a completed pending day."""
    today_key = selected_day.isoformat()
    pending_days = _daily_list_pending_reports()

    # Prefer an older pending report that is now fully final; this supports the next-day click.
    candidate_days = [d for d in pending_days if d <= today_key]
    target_day = candidate_days[0] if candidate_days else None

    if target_day:
        pregame = _daily_read_json(f"{target_day}_pregame.json")
        results_payload = _daily_collect_results(pregame)
        if not results_payload["all_games_resolved"]:
            remaining = len(results_payload["unresolved_game_ids"])
            raise RuntimeError(
                f"O relatório de {target_day} ainda tem {remaining} jogo(s) sem resultado final. "
                "Volta a clicar depois de terminarem."
            )
        saved_results, created = _daily_save_results(
            f"{target_day}_results.json", results_payload
        )
        predictions = pregame.get("predictions", [])
        odds_rows = _daily_flatten_odds(
            pregame.get("dunkapi_responses", []),
            {str(g["game_id"]): {
                "home_team": g["home_team"], "away_team": g["away_team"],
                "starts_at": g["starts_at_utc"],
            } for g in pregame.get("games", [])}
        )
        zip_bytes = _daily_zip_bytes(pregame, odds_rows, saved_results)
        return (
            f"NBA_LAB_PRO_{target_day}_COMPLETO.zip",
            zip_bytes,
            f"Resultados finais associados a {len(saved_results.get('results', []))} jogos. "
            "A captura pré-jogo original foi preservada.",
        )

    # No completed pending report: capture the selected date once.
    existing = _daily_read_json(f"{today_key}_pregame.json")
    if existing is not None:
        raise RuntimeError(
            f"Já existe uma captura pré-jogo para {today_key}. "
            "Ainda não há um relatório anterior pronto para atualizar com resultados."
        )
    if not current_games:
        raise RuntimeError("Não há jogos carregados para a data selecionada.")

    capture_moment = datetime.datetime.now(datetime.timezone.utc)
    eligible_games = []
    excluded_games = []
    for game in current_games:
        kickoff_check = datetime.datetime.fromtimestamp(
            float(game["timestamp"]), tz=datetime.timezone.utc
        )
        if str(game.get("status", "NS")).upper() != "NS" or kickoff_check <= capture_moment:
            excluded_games.append({
                "game_id": str(game.get("id")),
                "home_team": game.get("home"),
                "away_team": game.get("away"),
                "status": game.get("status"),
                "reason": "Jogo já iniciado, terminado ou sem estado pré-jogo confirmado",
            })
        else:
            eligible_games.append(game)
    if not eligible_games:
        raise RuntimeError(
            "Não há jogos ainda por começar com estado pré-jogo confirmado. "
            "Não vou criar previsões retroativas."
        )

    now_utc = capture_moment.isoformat()
    predictions = _daily_build_predictions(eligible_games, now_utc)
    games_saved = []
    for game in eligible_games:
        kickoff = datetime.datetime.fromtimestamp(float(game["timestamp"]), tz=datetime.timezone.utc)
        games_saved.append({
            "game_id": str(game["id"]),
            "home_team": game["home"],
            "away_team": game["away"],
            "starts_at_utc": kickoff.isoformat(),
            "game_date_et": kickoff.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date().isoformat(),
            "game_date_portugal": kickoff.astimezone(__import__("zoneinfo").ZoneInfo("Europe/Lisbon")).date().isoformat(),
            "status_at_capture": game.get("status"),
        })
    et_dates = [g["game_date_et"] for g in games_saved]
    dunk_responses = _daily_fetch_dunk_odds(et_dates)
    game_lookup = {
        str(g["game_id"]): {
            "home_team": g["home_team"], "away_team": g["away_team"],
            "starts_at": g["starts_at_utc"],
        } for g in games_saved
    }
    odds_rows = _daily_flatten_odds(dunk_responses, game_lookup)
    pregame_payload = {
        "report_date": today_key,
        "captured_at_utc": now_utc,
        "phase": "pregame",
        "games": games_saved,
        "excluded_games": excluded_games,
        "predictions": predictions,
        "dunkapi_query_dates_et": sorted(set(et_dates)),
        "dunkapi_estimated_credits": 10 * len(set(et_dates)),
        "dunkapi_responses": dunk_responses,
    }
    saved, created = _daily_create_json(f"{today_key}_pregame.json", pregame_payload)
    if not created:
        raise RuntimeError(
            f"Já existia uma captura pré-jogo para {today_key}; não foi substituída."
        )
    zip_bytes = _daily_zip_bytes(saved, odds_rows)
    return (
        f"NBA_LAB_PRO_{today_key}_PREJOGO.zip",
        zip_bytes,
        f"Captura pré-jogo concluída: {len(games_saved)} jogos, "
        f"{len(odds_rows)} registos de odds. Estimativa DunkAPI: "
        f"{pregame_payload['dunkapi_estimated_credits']} créditos.",
    )


def _render_daily_report_button(day, day_games):
    st.markdown("#### Relatório diário")
    st.caption(
        "Primeiro clique: captura pré-jogo M1/M2 + odds DunkAPI. "
        "Num clique posterior, quando os jogos estiverem terminados, recolhe os resultados."
    )
    if st.button(
        "📊 Preparar relatório diário",
        key=f"prepare_daily_report_{day.isoformat()}",
        use_container_width=True,
        type="primary",
    ):
        try:
            with st.spinner("A recolher previsões, odds e/ou resultados. Não feche a aplicação…"):
                filename, content, message = _daily_report_action(day, day_games)
            st.session_state["nba_daily_report_download"] = {
                "filename": filename,
                "content": content,
                "message": message,
            }
            st.success(message)
        except Exception as exc:
            st.error(f"Não foi possível preparar o relatório: {exc}")

    download = st.session_state.get("nba_daily_report_download")
    if download:
        st.download_button(
            "⬇️ Descarregar ZIP do relatório",
            data=download["content"],
            file_name=download["filename"],
            mime="application/zip",
            key=f"download_{download['filename']}",
            use_container_width=True,
        )


selected_game = next((g for g in games if g["id"] == st.session_state.get("nba_selected_game")), None)

if active_page == "Ver calendário":
    with main_area:
        left, right = st.columns([0.9, 2.1], gap="medium")
        with left:
            st.markdown('<div class="section-label">01 · Jogos</div>', unsafe_allow_html=True)
            st.date_input("Data dos jogos", value=selected_date, key="nba_matchday_widget", on_change=persist_matchday, format="DD/MM/YYYY")
            _render_daily_report_button(selected_date, games)
            st.markdown(f'<div class="fixture-help">{selected_date.strftime("%A, %d %B %Y")} · {len(games)} jogo(s)</div>', unsafe_allow_html=True)
            if not API_KEY:
                st.warning("Configura `API_SPORTS_NBA_KEY` nas Secrets.")
            elif not games:
                if any("erro" in item for item in api_debug):
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #2A2112, #171B25);
                        border: 1px solid #8A672D;
                        border-radius: 12px;
                        padding: 24px 16px;
                        text-align: center;
                        margin-top: 8px;
                    ">
                        <div style="font-size:30px;margin-bottom:10px">⚠️</div>
                        <div style="color:#F5D58A;font-size:17px;font-weight:750;margin-bottom:8px">
                            Não foi possível carregar os jogos
                        </div>
                        <div style="color:#C3CBD5;font-size:13px;line-height:1.6">
                            A API NBA não respondeu a tempo. Isto não significa que não existam jogos agendados.
                            <br>O calendário mantém-se disponível; tenta novamente dentro de momentos.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #122337, #101B29);
                        border: 1px solid #29445B;
                        border-radius: 12px;
                        padding: 28px 16px;
                        text-align: center;
                        margin-top: 8px;
                    ">
                        <div style="font-size:32px;margin-bottom:12px">📅</div>
                        <div style="color:#E5F1FC;font-size:18px;font-weight:700;margin-bottom:8px">
                            Sem jogos agendados
                        </div>
                        <div style="color:#9CB2C8;font-size:13px;line-height:1.6">
                            A API respondeu, mas não devolveu jogos para esta data.
                            <br>Seleciona outro dia no calendário para consultar os jogos.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Compact rows; each button is one line when possible.
                with st.container(height=455, border=False):
                    for game in games:
                        is_selected = st.session_state.get("nba_selected_game") == game["id"]
                        time_col, match_col = st.columns([0.23, 0.77], gap="small", vertical_alignment="center")
                        with time_col:
                            st.markdown(
                                f'<div style="font-size:.72rem;font-weight:750;color:{"#55ddb0" if is_selected else "#8fa4ba"};text-align:center;padding:7px 0">{game["time"]}</div>',
                                unsafe_allow_html=True
                            )
                        with match_col:
                            st.markdown('<div class="game-button">', unsafe_allow_html=True)
                            status = game.get("status", "NS")
                            if status == "FT" and game.get("home_score") is not None and game.get("away_score") is not None:
                                label = f"{game['away']} {game['away_score']} @ {game['home']} {game['home_score']} · FINAL"
                            elif status in ("LIVE", "Q1", "Q2", "Q3", "Q4", "HT", "OT"):
                                label = f"🔴 {game['away']} @ {game['home']} · EM DIRETO"
                            else:
                                label = f"▸ **{game['away']} @ {game['home']}**" if is_selected else f"{game['away']} @ {game['home']}"
                            if st.button(label, key=f"choose_{date_str}_{game['id']}",
                                         type="primary" if is_selected else "secondary", use_container_width=True):
                                st.session_state["nba_selected_game"] = game["id"]
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="small-note">Seleciona qualquer jogo para atualizar os mercados à direita.</div>', unsafe_allow_html=True)
        
        with right:
            st.markdown('<div class="section-label">02 · Análise dos mercados</div>', unsafe_allow_html=True)
            if not selected_game:
                st.markdown('<div class="empty-state">Escolhe um jogo na lista à esquerda.<br>Os dois mercados aparecem aqui, lado a lado.</div>', unsafe_allow_html=True)
            else:
                home, away = selected_game["home"], selected_game["away"]
                status_code = selected_game.get("status", "NS")
                if status_code == "FT" and selected_game.get("home_score") is not None and selected_game.get("away_score") is not None:
                    status_label = f"FINAL · {selected_game['home_score']}–{selected_game['away_score']}"
                elif status_code in ("LIVE", "Q1", "Q2", "Q3", "Q4", "HT", "OT"):
                    status_label = "EM DIRETO"
                else:
                    status_label = "PRÉ-JOGO"
                st.markdown(f"""
                <div class="match-header">
                  <div class="match-title">{away} <span style="color:#6c849b;font-weight:500">@</span> {home}</div>
                  <div class="match-meta">📅 {selected_date.strftime("%d/%m/%Y")} &nbsp; · &nbsp; 🕒 {selected_game["time"]} (Portugal) &nbsp; · &nbsp; <span style="color:#68c4ff;font-weight:750">VISITANTE @ CASA</span> &nbsp; · &nbsp; <b>{status_label}</b></div>
                </div>
                """, unsafe_allow_html=True)
        
                if status_code == "FT" and selected_game.get("home_score") is not None and selected_game.get("away_score") is not None:
                    final_home = int(selected_game["home_score"])
                    final_away = int(selected_game["away_score"])
                    final_total = final_home + final_away
                    st.markdown(f"""
                    <div class="final-score-panel">
                      <div class="final-score-kicker">🏁 RESULTADO FINAL</div>
                      <div class="final-score-teams">
                        <div class="final-team"><span>{away}</span><strong>{final_away}</strong></div>
                        <div class="final-vs">@</div>
                        <div class="final-team"><span>{home}</span><strong>{final_home}</strong></div>
                      </div>
                      <div class="final-total">TOTAL DE PONTOS <strong>{final_total}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)
                    saved_row = log_df[
                        log_df["game_id"].map(normalize_game_id) == normalize_game_id(selected_game["id"])
                    ] if not log_df.empty else pd.DataFrame()
                    if not saved_row.empty:
                        saved = saved_row.iloc[-1]
        
                        def outcome_card(title, pick, result, detail=""):
                            result_text = str(result or "Pendente")
                            if result_text == "Acertou":
                                css_class, icon_text = "won", "✓ ACERTOU"
                            elif result_text == "Falhou":
                                css_class, icon_text = "lost", "✕ FALHOU"
                            elif result_text == "Push":
                                css_class, icon_text = "push", "＝ PUSH"
                            else:
                                css_class, icon_text = "pending", "… PENDENTE"
                            return f"""
                            <div class="market-outcome {css_class}">
                              <div class="outcome-heading">{title}</div>
                              <div class="outcome-pick">{pick}</div>
                              <div class="outcome-result">{icon_text}</div>
                              <div class="outcome-detail">{detail}</div>
                            </div>"""
        
                        ml_pick = str(saved.get("moneyline_pick", "—"))
                        ml_detail = f"Vencedor real: {home if final_home > final_away else away if final_away > final_home else 'Empate'}"
                        total_pick = str(saved.get("total_pick", "—"))
                        total_line_value = saved.get("total_line", "—")
                        total_detail = f"Total real: {final_total} pts · Linha: {total_line_value}"
                        outcome1 = outcome_card("MONEYLINE", f"Seleção: {ml_pick}", saved.get("moneyline_result", "Pendente"), ml_detail)
                        outcome2 = outcome_card("OVER / UNDER", f"Seleção: {total_pick}", saved.get("total_result", "Pendente"), total_detail)
                        out1, out2 = st.columns(2, gap="small")
                        with out1:
                            st.markdown(outcome1, unsafe_allow_html=True)
                        with out2:
                            st.markdown(outcome2, unsafe_allow_html=True)
                    else:
                        st.info("O resultado final está disponível, mas não existe uma previsão congelada para este jogo. Não é possível classificar um acerto retroativamente.")
        
                # MARKET 1: Moneyline
                market1, market2 = st.columns(2, gap="small")
                with market1:
                    st.markdown("""
                    <div class="market-card">
                      <div class="market-kicker">Mercado 01 · Moneyline</div>
                      <div class="market-title">🏆 Vencedor do jogo</div>
                      <div class="market-desc">Probabilidade e odd teórica por equipa.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    kickoff_m1 = pd.to_datetime(selected_game["timestamp"], unit="s", utc=True)
                    m1_error = None
                    try:
                        m1_result = predict_moneyline_m1(home, away, kickoff_m1)
                        p_home = float(m1_result["p_home_m1"])
                        p_away = float(m1_result["p_away_m1"])
                        if not (0.0 <= p_home <= 1.0 and 0.0 <= p_away <= 1.0):
                            raise ValueError("M1 devolveu uma probabilidade fora de [0,1].")
                        if abs((p_home + p_away) - 1.0) > 1e-9:
                            raise ValueError("As probabilidades M1 não somam 100%.")
                        fair_home, fair_away = 1 / p_home, 1 / p_away
                    except Exception as exc:
                        p_home = p_away = fair_home = fair_away = None
                        m1_error = str(exc)
                    if m1_error:
                        st.error(f"M1 v1.1 indisponível: {m1_error}")
                    else:
                        a, b = st.columns(2, gap="small")
                        with a:
                            st.metric(f"Casa · {home}", f"{p_home*100:.1f}%", f"Odd justa {fair_home:.2f}")
                        with b:
                            st.metric(f"Visitante · {away}", f"{p_away*100:.1f}%", f"Odd justa {fair_away:.2f}")
                    # Load saved values per game so inputs survive page reloads and redeploys.
                    draft_for_game = load_market_drafts().get(normalize_game_id(selected_game["id"]), {})
                    default_home_odd = float(draft_for_game.get("home_market_odds", 1.95)) if pd.notna(draft_for_game.get("home_market_odds", 1.95)) else 1.95
                    default_away_odd = float(draft_for_game.get("away_market_odds", 1.95)) if pd.notna(draft_for_game.get("away_market_odds", 1.95)) else 1.95
                    default_total_line = float(draft_for_game.get("total_line", 220.5)) if pd.notna(draft_for_game.get("total_line", 220.5)) else 220.5
                    # Use native Streamlit labels so both inputs align and never overlap.
                    odds_home_col, odds_away_col = st.columns(2, gap="small")
                    with odds_home_col:
                        market_odd_home = st.number_input(
                            "Odd Casa", min_value=1.01, value=default_home_odd, step=0.05,
                            help=f"Odd de mercado para {home}",
                            key=f"market_odd_home_{selected_game['id']}"
                        )
                    with odds_away_col:
                        market_odd_away = st.number_input(
                            "Odd Visitante", min_value=1.01, value=default_away_odd, step=0.05,
                            help=f"Odd de mercado para {away}",
                            key=f"market_odd_away_{selected_game['id']}"
                        )
                    if fair_home is not None and fair_away is not None:
                        edge_home = (market_odd_home / fair_home - 1) * 100
                        edge_away = (market_odd_away / fair_away - 1) * 100
                        edge_col1, edge_col2 = st.columns(2, gap="small")
                        with edge_col1:
                            if edge_home > 0:
                                st.markdown(f'<div class="edge-box">EDGE · CASA <b style="float:right">+{edge_home:.1f}%</b></div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="edge-box neutral">EDGE · CASA <b style="float:right">Sem vantagem</b></div>', unsafe_allow_html=True)
                        with edge_col2:
                            if edge_away > 0:
                                st.markdown(f'<div class="edge-box">EDGE · VISITANTE <b style="float:right">+{edge_away:.1f}%</b></div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="edge-box neutral">EDGE · VISITANTE <b style="float:right">Sem vantagem</b></div>', unsafe_allow_html=True)
                    else:
                        st.info("Edge indisponível enquanto o M1 não conseguir atualizar o estado da época.")
                    st.caption("Probabilidades Moneyline produzidas pelo M1 v1.1; resultados anteriores ao dia do jogo atualizam o estado das equipas sem retreinar o classificador.")
        
                # MARKET 2: Total points / Over Under
                with market2:
                    st.markdown("""
                    <div class="market-card total">
                      <div class="market-kicker">MOTOR PRINCIPAL · M2 HGB v1</div>
                      <div class="market-title">🏀 Total de pontos · Over / Under</div>
                      <div class="market-desc">Previsão M2 e probabilidades calculadas para cada linha.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    over_line = st.number_input(
                        "Linha para consulta (pontos)", min_value=150.0, max_value=300.0,
                        value=default_total_line, step=0.5,
                        key=f"total_line_{selected_game['id']}"
                    )
                    kickoff_utc = pd.to_datetime(selected_game["timestamp"], unit="s", utc=True)
                    m2_total, m2_error = predict_total(home, away, kickoff=kickoff_utc)
                    if m2_total is None:
                        st.error(m2_error or "M2 indisponível. Não será apresentada uma estimativa alternativa.")
                        total_reference = None
                    else:
                        total_reference = m2_total
                        p_over, p_under = probability_over(m2_total, over_line)
                        st.metric("Total previsto — M2 HGB v1", f"{m2_total:.1f} pts")
                        prob_col, under_col = st.columns(2, gap="small")
                        with prob_col:
                            st.metric("Probabilidade Over", f"{p_over*100:.1f}%", f"Odd justa {1/p_over:.2f}")
                        with under_col:
                            st.metric("Probabilidade Under", f"{p_under*100:.1f}%", f"Odd justa {1/p_under:.2f}")
        
                        # Show the market ladder around the selected reference line.
                        ladder_lines = [round(over_line + offset, 1) for offset in range(-7, 8)]
                        ladder = []
                        for line_value in ladder_lines:
                            po, pu = probability_over(m2_total, line_value)
                            ladder.append({"Linha": line_value, "Over": f"{po*100:.1f}%", "Under": f"{pu*100:.1f}%", "Odd justa Over": round(1/po, 2), "Odd justa Under": round(1/pu, 2)})
                        st.caption("Escada de probabilidades M2 · linhas de 1 em 1 ponto em torno da linha selecionada")
                        st.dataframe(pd.DataFrame(ladder), use_container_width=True, hide_index=True)
                        gap = m2_total - over_line
                        st.caption(f"Diferença entre previsão M2 e linha: {gap:+.1f} pontos · Distribuição Normal fixa · σ = 15,54")
                        st.caption("M2 HGB v1 congelado · Treinado em 6.150 jogos até 2025 · forma 2026–27 incorporada nas features, sem retreino.")
                        st.caption("Ridge legado: permanece no repositório, mas não é utilizado pela aplicação nem substitui o M2 em caso de erro.")
        
                # Persist the values every time Streamlit reruns after an input change.
                save_market_draft(selected_game["id"], market_odd_home, market_odd_away, over_line)
        
                st.info("As odds e a linha de mercado são guardadas automaticamente quando alteras os campos. A previsão estatística é congelada separadamente a T−5 quando a aplicação executa uma atualização.")
        
        
        st.markdown('<hr><div class="small-note">NBA LAB PRO · Moneyline mantém a implementação anterior. Motor principal de Total de Pontos: M2 HGB v1 congelado. Ridge legado não é chamado pela aplicação. Candidato em validação prospectiva; não assumir rentabilidade.</div>', unsafe_allow_html=True)



if active_page == "Adicionar aposta":
    with main_area:
        if st.session_state.pop("nba_bet_reset_form", False):
            # Limpar o estado dos widgets antes de estes serem criados neste rerun.
            # Não atribuir None ao selectbox do vencedor: o valor tem de pertencer
            # às opções atuais, caso contrário o Streamlit lança KeyError.
            for widget_key in (
                "draft_bet_type",
                "draft_bet_stake",
                "draft_bet_combined",
                "draft_leg_side",
                "draft_leg_odds",
                "draft_leg_line",
                "draft_leg_moneyline_winner",
                "draft_leg_date",
                "draft_leg_game",
            ):
                st.session_state.pop(widget_key, None)
        st.markdown("## Adicionar aposta")
        st.caption("Regista uma aposta simples ou múltipla.")
        st.markdown("### Criar boletim")
        draft_type = st.radio(
            "Tipo de aposta",
            ["Simples", "Múltipla"],
            horizontal=True,
            key="draft_bet_type",
            help="Uma simples contém uma seleção; uma múltipla contém duas ou mais."
        )
        
        d1, d2 = st.columns(2)
        with d1:
            draft_stake = st.number_input(
                "Stake (€)", min_value=0.01, max_value=100000.0,
                value=5.0, step=0.5, key="draft_bet_stake"
            )
        with d2:
            draft_combined = st.number_input(
                "Odd total do bookmaker", min_value=1.01, max_value=100000.0,
                value=1.90, step=0.01, key="draft_bet_combined"
            )
        
        st.divider()
        st.markdown("#### Adicionar uma seleção")

        # A data da seleção é independente da data atualmente aberta no calendário.
        # Assim é possível registar antecipadamente jogos de amanhã ou de qualquer
        # outra data sem alterar a navegação principal do LAB.
        add_bet_date = st.date_input(
            "Data dos jogos",
            value=st.session_state.get("draft_leg_date", selected_date),
            key="draft_leg_date",
            format="DD/MM/YYYY"
        )

        add_bet_games = []
        add_bet_fetch_failed = False

        if not API_KEY:
            st.error("Chave da API NBA não encontrada. Não é possível carregar os jogos.")
            add_bet_fetch_failed = True

        # Se a data escolhida coincide com a do calendário, reutilizar os jogos
        # já carregados. Evita repetir exatamente os mesmos pedidos à API.
        elif add_bet_date == selected_date:
            add_bet_games = [
                {
                    "id": g["id"],
                    "home": g["home"],
                    "away": g["away"],
                    "time": g["time"],
                    "status": g.get("status", "NS")
                }
                for g in games
            ]
            # O calendário já reportou falha para esta data se não conseguiu
            # obter qualquer jogo. Não confundir falha de ligação com dia vazio.
            if not add_bet_games and not api_debug:
                add_bet_fetch_failed = True

        else:
            # Cache por sessão: mudar mercado/odd ou adicionar uma seleção não
            # deve voltar a consultar a API para a mesma data.
            if "bet_games_date_cache" not in st.session_state:
                st.session_state["bet_games_date_cache"] = {}

            cache_key = add_bet_date.isoformat()
            cached_games = st.session_state["bet_games_date_cache"].get(cache_key)

            if cached_games is not None:
                add_bet_games = cached_games
            else:
                add_bet_seen_ids = set()
                successful_api_dates = 0

                # Consultar o dia anterior e o próprio dia para cobrir jogos
                # cuja data UTC passa para o dia seguinte em Portugal.
                for api_date in (
                    add_bet_date - datetime.timedelta(days=1),
                    add_bet_date
                ):
                    api_date_str = api_date.strftime("%Y-%m-%d")
                    api_season = str(
                        api_date.year if api_date.month >= 10 else api_date.year - 1
                    )
                    try:
                        add_bet_payload, add_bet_error = fetch_nba_api_date(
                            api_date_str, api_season, API_KEY
                        )
                        if add_bet_error:
                            st.warning(
                                f"Não foi possível consultar {api_date_str}: "
                                f"{add_bet_error}. A consulta será tentada novamente mais tarde."
                            )
                            continue

                        successful_api_dates += 1
                        for item in add_bet_payload.get("response", []):
                            teams = item.get("teams") or {}
                            home = teams.get("home") or {}
                            away = teams.get("visitors") or teams.get("away") or {}
                            start_time = (item.get("date") or {}).get("start")
                            if not home.get("name") or not away.get("name") or not start_time:
                                continue

                            local_dt = (
                                pd.to_datetime(start_time, utc=True)
                                .tz_convert("Europe/Lisbon")
                            )
                            if local_dt.date() != add_bet_date:
                                continue

                            game_id = str(item.get("id") or start_time)
                            if game_id in add_bet_seen_ids:
                                continue
                            add_bet_seen_ids.add(game_id)

                            add_bet_games.append({
                                "id": game_id,
                                "home": home["name"],
                                "away": away["name"],
                                "time": local_dt.strftime("%H:%M"),
                                "status": normalize_game_status(item.get("status") or {})
                            })

                    except requests.RequestException as exc:
                        add_bet_fetch_failed = True
                        st.warning(
                            f"Falha de ligação ao consultar {api_date_str}: {exc}"
                        )
                    except (ValueError, TypeError) as exc:
                        add_bet_fetch_failed = True
                        st.warning(
                            f"Resposta inválida da API para {api_date_str}: {exc}"
                        )

                # Guardar apenas quando as duas consultas foram processadas
                # sem falha de ligação/API.
                if successful_api_dates == 2 and not add_bet_fetch_failed:
                    st.session_state["bet_games_date_cache"][cache_key] = add_bet_games

        if add_bet_games:
            game_options = {
                f"{add_bet_date.strftime('%d/%m/%Y')} · {g['away']} @ {g['home']} · {g['time']}": g
                for g in add_bet_games
            }
            selected_leg_label = st.selectbox(
                "Jogo", list(game_options.keys()), key="draft_leg_game"
            )
            a1, a2, a3 = st.columns(3)
            with a1:
                draft_market = st.selectbox(
                    "Mercado", ["Over", "Under", "Moneyline"], key="draft_leg_side"
                )
            draft_winner = None
            if draft_market == "Moneyline":
                with a2:
                    winner_options = {
                        f"Casa · {game_options[selected_leg_label]['home']}": "Casa",
                        f"Visitante · {game_options[selected_leg_label]['away']}": "Visitante",
                    }
                    draft_winner_label = st.selectbox(
                        "Vencedor do jogo", list(winner_options.keys()), key="draft_leg_moneyline_winner"
                    )
                    draft_winner = winner_options[draft_winner_label]
                with a3:
                    draft_leg_odds = st.number_input(
                        "Odd da seleção", min_value=1.01, max_value=100.0,
                        value=1.90, step=0.01, key="draft_leg_odds"
                    )
                draft_line = None
            else:
                with a2:
                    draft_line = st.number_input(
                        "Linha de pontos", min_value=100.0, max_value=350.0,
                        value=220.5, step=0.5, key="draft_leg_line"
                    )
                with a3:
                    draft_leg_odds = st.number_input(
                        "Odd da seleção", min_value=1.01, max_value=100.0,
                        value=1.90, step=0.01, key="draft_leg_odds"
                    )
        
            add_col, helper_col = st.columns([1, 3])
            with add_col:
                if st.button("＋ Adicionar seleção", key="add_draft_leg", use_container_width=True):
                    g = game_options[selected_leg_label]
                    st.session_state["bet_slip_draft_legs"].append({
                        "game_id": str(g["id"]),
                        "game_date": add_bet_date.isoformat(),
                        "home_team": g["home"],
                        "away_team": g["away"],
                        "selection": (
                            f"Moneyline · {draft_winner}" if draft_market == "Moneyline" else draft_market
                        ),
                        "line": float(draft_line) if draft_line is not None else None,
                        "odds": float(draft_leg_odds)
                    })
                    st.rerun()
            with helper_col:
                st.caption("A data acima é independente do calendário principal. Podes escolher jogos de amanhã ou de qualquer outro dia e adicionar várias seleções à mesma múltipla.")
        elif add_bet_fetch_failed:
            st.warning(
                "Não foi possível confirmar os jogos desta data porque a API não respondeu. "
                "Isto não significa que não existam jogos. Tenta novamente mais tarde."
            )
        else:
            st.info(
                f"A API respondeu, mas não devolveu jogos NBA para "
                f"{add_bet_date.strftime('%d/%m/%Y')}. Confirma a data ou tenta outro dia."
            )
        
        draft_legs = st.session_state["bet_slip_draft_legs"]
        st.divider()
        st.markdown(f"#### Seleções adicionadas · {len(draft_legs)}")
        if draft_legs:
            draft_view = pd.DataFrame([{
                "Jogo": f"{x['away_team']} @ {x['home_team']}",
                "Data": x["game_date"],
                "Mercado": x["selection"],
                "Linha": x["line"] if x["line"] is not None else "—",
                "Odd": x["odds"]
            } for x in draft_legs])
            st.dataframe(draft_view, use_container_width=True, hide_index=True)
        
            r1, r2, r3 = st.columns([2, 2, 3])
            with r1:
                remove_idx = st.selectbox(
                    "Selecionar para remover",
                    list(range(len(draft_legs))),
                    format_func=lambda i: f"{i+1}. {draft_legs[i]['away_team']} @ {draft_legs[i]['home_team']}",
                    key="remove_draft_leg_idx"
                )
                if st.button("Remover seleção", key="remove_draft_leg", use_container_width=True):
                    st.session_state["bet_slip_draft_legs"].pop(remove_idx)
                    st.rerun()
            with r2:
                if st.button("Limpar rascunho", key="clear_draft_legs", use_container_width=True):
                    st.session_state["bet_slip_draft_legs"] = []
                    st.rerun()
            can_save = len(draft_legs) == 1 if draft_type == "Simples" else len(draft_legs) >= 2
            with r3:
                st.write("")
                if st.button("Guardar boletim", key="save_bet_slip", type="primary",
                             use_container_width=True, disabled=not can_save):
                    sid = uuid.uuid4().hex[:12]
                    slip_row = {
                        "slip_id": sid,
                        "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                        "bet_type": draft_type,
                        "stake": float(draft_stake),
                        "combined_odds": float(draft_combined),
                        "status": "Pendente",
                        "return_amount": None,
                        "profit_loss": None
                    }
                    leg_rows = []
                    for n, leg in enumerate(draft_legs, start=1):
                        leg_rows.append({
                            "selection_id": f"{sid}-{n}",
                            "slip_id": sid,
                            **leg,
                            "away_score": None,
                            "home_score": None,
                            "actual_total": None,
                            "result": "Pendente"
                        })
                    slips_df = pd.concat([slips_df, pd.DataFrame([slip_row])], ignore_index=True)
                    legs_df = pd.concat([legs_df, pd.DataFrame(leg_rows)], ignore_index=True)
                    save_slip_data(slips_df, legs_df)
                    st.session_state["bet_slip_draft_legs"] = []
                    st.session_state["nba_bet_success_toast"] = (
                        f"Boletim #{len(slips_df):03d} registado com sucesso!"
                    )
                    st.session_state["nba_bet_reset_form"] = True
                    st.rerun()
            if not can_save:
                st.caption("Para guardar uma simples, adiciona exatamente uma seleção. Para uma múltipla, adiciona pelo menos duas.")
        else:
            st.caption("Ainda não adicionaste seleções a este boletim.")
        


if active_page == "Ver apostas":
    with main_area:
        st.markdown("## Histórico de apostas")
        st.caption("Todos os teus boletins NBA num só lugar. Consulta o resumo ou abre qualquer recibo.")

        delete_feedback = st.session_state.pop("delete_slip_feedback", None)
        delete_feedback_error = st.session_state.pop("delete_slip_feedback_error", False)
        if delete_feedback:
            if delete_feedback_error:
                st.toast(delete_feedback, icon="⚠️")
            else:
                st.toast(delete_feedback, icon="✅")


        if slips_df.empty:
            st.info("Ainda não existem boletins registados.")
        else:
            if "manage_slip_id" not in st.session_state:
                st.session_state["manage_slip_id"] = None
            if "pending_delete_slip_id" not in st.session_state:
                st.session_state["pending_delete_slip_id"] = None

            ordered_slips = slips_df.copy()
            ordered_slips["_saved_dt"] = pd.to_datetime(ordered_slips["saved_at"], errors="coerce")
            ordered_slips = ordered_slips.sort_values("_saved_dt", ascending=False, na_position="last")

            # Número sequencial estável: o boletim mais antigo é #001 e os novos
            # recebem números progressivamente superiores, mesmo com "Mais recentes" no topo.
            chronological_slips = slips_df.copy()
            chronological_slips["_saved_dt"] = pd.to_datetime(
                chronological_slips["saved_at"], errors="coerce"
            )
            chronological_slips = chronological_slips.sort_values(
                "_saved_dt", ascending=True, na_position="first", kind="stable"
            )
            slip_number_map = {
                str(sid): number
                for number, sid in enumerate(chronological_slips["slip_id"].astype(str), start=1)
            }

            # KPIs reais calculados a partir dos boletins guardados.
            status_series = ordered_slips["status"].fillna("Pendente").astype(str).str.strip().str.lower()
            won_mask = status_series.isin(["ganho", "won", "ganha"])
            lost_mask = status_series.isin(["perdido", "lost", "perdida"])
            pending_mask = status_series.isin(["pendente", "pending", "open", "aberto"])
            total_stake = pd.to_numeric(ordered_slips["stake"], errors="coerce").fillna(0).sum()
            settled_mask = won_mask | lost_mask
            net_result = pd.to_numeric(
                ordered_slips.loc[settled_mask, "profit_loss"], errors="coerce"
            ).fillna(0).sum()
            win_rate = (won_mask.sum() / settled_mask.sum() * 100) if settled_mask.sum() else 0.0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Boletins registados", f"{len(ordered_slips)}")
            k2.metric("Em aberto", f"{int(pending_mask.sum())}")
            k3.metric("Taxa de acerto", f"{win_rate:.1f}%")
            k4.metric("Resultado líquido", f"{net_result:+.2f} €")

            st.markdown("### Os meus boletins")

            # Filtros compactos, inspirados nas áreas My Bets / Bet History de sportsbooks.
            f1, f2, f3, f4, f5 = st.columns([1.1, 1.05, 1.15, 1.35, 1.5])
            with f1:
                status_filter = st.selectbox(
                    "Estado", ["Todos", "Pendentes", "Ganhas", "Perdidas"], key="history_status_filter"
                )
            with f2:
                type_filter = st.selectbox(
                    "Tipo", ["Todos", "Simples", "Múltiplas"], key="history_type_filter"
                )
            with f3:
                period_filter = st.selectbox(
                    "Período", ["Todo o histórico", "Últimos 7 dias", "Últimos 30 dias", "Últimos 90 dias"],
                    key="history_period_filter"
                )
            with f4:
                search_text = st.text_input(
                    "Pesquisar", placeholder="Equipa, mercado ou ID do boletim…", key="history_search"
                )
            with f5:
                sort_filter = st.selectbox(
                    "Ordenar por",
                    ["Mais recentes", "Mais antigas", "Maior odd", "Maior stake"],
                    key="history_sort_filter",
                )

            visible_slips = ordered_slips.copy()
            visible_slips["_status_norm"] = visible_slips["status"].fillna("Pendente").astype(str).str.strip().str.lower()
            visible_slips["_type_norm"] = visible_slips["bet_type"].fillna("").astype(str).str.strip().str.lower()

            if status_filter == "Pendentes":
                visible_slips = visible_slips[visible_slips["_status_norm"].isin(["pendente", "pending", "open", "aberto"])]
            elif status_filter == "Ganhas":
                visible_slips = visible_slips[visible_slips["_status_norm"].isin(["ganho", "won", "ganha"])]
            elif status_filter == "Perdidas":
                visible_slips = visible_slips[visible_slips["_status_norm"].isin(["perdido", "lost", "perdida"])]

            if type_filter == "Simples":
                visible_slips = visible_slips[visible_slips["_type_norm"].isin(["simples", "single"])]
            elif type_filter == "Múltiplas":
                visible_slips = visible_slips[visible_slips["_type_norm"].isin(["múltipla", "multipla", "múltiplas", "multipla"])]

            if period_filter != "Todo o histórico":
                days = {"Últimos 7 dias": 7, "Últimos 30 dias": 30, "Últimos 90 dias": 90}[period_filter]
                cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days - 1)
                visible_slips = visible_slips[visible_slips["_saved_dt"].isna() | (visible_slips["_saved_dt"] >= cutoff)]

            query = search_text.strip().lower()
            if query:
                matched_ids = []
                for _, search_slip in visible_slips.iterrows():
                    search_id = str(search_slip.get("slip_id", "")).lower()
                    search_legs = legs_df[legs_df["slip_id"].astype(str) == str(search_slip["slip_id"])]
                    searchable = " ".join(
                        search_legs[["away_team", "home_team", "selection"]].fillna("").astype(str).values.flatten()
                    ).lower()
                    if query in search_id or query in searchable:
                        matched_ids.append(str(search_slip["slip_id"]))
                visible_slips = visible_slips[visible_slips["slip_id"].astype(str).isin(matched_ids)]

            if sort_filter == "Mais antigas":
                visible_slips = visible_slips.sort_values("_saved_dt", ascending=True, na_position="last")
            elif sort_filter == "Maior odd":
                visible_slips = visible_slips.assign(
                    _sort_value=pd.to_numeric(visible_slips["combined_odds"], errors="coerce")
                ).sort_values("_sort_value", ascending=False, na_position="last")
            elif sort_filter == "Maior stake":
                visible_slips = visible_slips.assign(
                    _sort_value=pd.to_numeric(visible_slips["stake"], errors="coerce")
                ).sort_values("_sort_value", ascending=False, na_position="last")

            st.caption(f"A mostrar {len(visible_slips)} de {len(ordered_slips)} boletim(ns).")

            if visible_slips.empty:
                st.info("Não foram encontrados boletins com estes filtros.")
            else:
                slip_ids = visible_slips["slip_id"].astype(str).tolist()
                chosen_slip_id = st.session_state.get("manage_slip_id")
                if chosen_slip_id not in slip_ids:
                    chosen_slip_id = None
                    st.session_state["manage_slip_id"] = None

                # Cada recibo é apresentado imediatamente abaixo do respetivo boletim.
                for _, slip in visible_slips.iterrows():
                    sid = str(slip["slip_id"])
                    row_number = slip_number_map.get(sid, 0)
                    group = legs_df[legs_df["slip_id"].astype(str) == sid].copy()
                    count = len(group)
                    bet_type = str(slip.get("bet_type") or ("Múltipla" if count > 1 else "Simples"))
                    selected = st.session_state.get("manage_slip_id") == sid
                    raw_date = slip.get("saved_at", "")
                    parsed_date = pd.to_datetime(raw_date, errors="coerce")
                    date_label = parsed_date.strftime("%d-%m-%Y · %H:%M") if pd.notna(parsed_date) else str(raw_date)[:10]

                    status_text = str(slip.get("status", "Pendente"))
                    status_norm = status_text.strip().lower()
                    if status_norm in ("perdido", "lost", "perdida"):
                        status_bg, status_fg, status_border = "#3a2026", "#ff929b", "#71343d"
                        status_label = "PERDIDA"
                    elif status_norm in ("ganho", "won", "ganha"):
                        status_bg, status_fg, status_border = "#153329", "#70ddb0", "#28664d"
                        status_label = "GANHA"
                    else:
                        status_bg, status_fg, status_border = "#2b2b20", "#e6d48b", "#62552b"
                        status_label = "PENDENTE"

                    odd_value = pd.to_numeric(slip.get("combined_odds"), errors="coerce")
                    stake_value = pd.to_numeric(slip.get("stake"), errors="coerce")
                    return_value = pd.to_numeric(slip.get("return_amount"), errors="coerce")
                    odd_label = f"{odd_value:.2f}" if pd.notna(odd_value) else "—"
                    stake_label = f"{stake_value:.2f} €" if pd.notna(stake_value) else "—"
                    return_label = f"{return_value:.2f} €" if pd.notna(return_value) else ("—" if status_label == "PENDENTE" else "0,00 €")
                    is_multiple = count > 1 or "múlt" in bet_type.lower()
                    type_label = "MÚLTIPLA" if is_multiple else "SIMPLES"
                    selections_label = f"{count} seleções" if count != 1 else "1 seleção"
                    if is_multiple:
                        type_bg, type_fg, type_border = "#30234a", "#d6b7ff", "#70539d"
                    else:
                        # Azul/ciano para identificar aposta simples; verde fica reservado a ganhos.
                        type_bg, type_fg, type_border = "#15364a", "#91dcff", "#28749a"

                    with st.container(border=True):
                        header_cols = st.columns([2.8, 1.05, 1.05, 1.0, 0.34], vertical_alignment="center")
                        with header_cols[0]:
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:9px;flex-wrap:wrap;">'
                                f'<span style="padding:4px 9px;border-radius:6px;background:{type_bg};'
                                f'border:1px solid {type_border};color:{type_fg};font-size:.68rem;'
                                f'font-weight:900;letter-spacing:.07em;">{type_label}</span>'
                                f'<span style="color:#8fa8bc;font-size:.73rem;font-weight:750;'
                                f'letter-spacing:.04em;">BOLETIM #{row_number:03d}</span>'
                                f'</div>'
                                f'<div style="font-size:.78rem;color:#b5c5d3;margin-top:8px;">'
                                f'{escape(date_label)} <span style="color:#58748b;">·</span> '
                                f'{escape(selections_label)}</div>',
                                unsafe_allow_html=True,
                            )
                        with header_cols[1]:
                            st.markdown(
                                f'<div style="font-size:.68rem;color:#91a7ba;font-weight:700;'
                                f'letter-spacing:.06em;text-transform:uppercase;">Stake</div>'
                                f'<div style="font-size:.96rem;font-weight:800;color:#edf4fa;margin-top:4px;">'
                                f'{stake_label}</div>',
                                unsafe_allow_html=True,
                            )
                        with header_cols[2]:
                            st.markdown(
                                f'<div style="font-size:.68rem;color:#91a7ba;font-weight:700;'
                                f'letter-spacing:.06em;text-transform:uppercase;">Odd total</div>'
                                f'<div style="font-size:1.08rem;font-weight:900;color:#f4f8fc;margin-top:4px;">'
                                f'{odd_label}</div>',
                                unsafe_allow_html=True,
                            )
                        with header_cols[3]:
                            st.markdown(
                                f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:5px;">'
                                f'<span style="padding:5px 10px;border-radius:7px;background:{status_bg};'
                                f'color:{status_fg};border:1px solid {status_border};font-size:.72rem;'
                                f'font-weight:850;letter-spacing:.04em;">{status_label}</span>'
                                f'<span style="font-size:.72rem;color:#9fb2c4;">Retorno: {return_label}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with header_cols[4]:
                            st.button(
                                "🗑",
                                key=f"delete_slip_{sid}",
                                help=f"Eliminar o Boletim #{row_number:03d}",
                                type="tertiary",
                                use_container_width=False,
                                on_click=lambda slip_id=sid: st.session_state.update(
                                    {"pending_delete_slip_id": slip_id}
                                ),
                            )
                        footer_cols = st.columns([1, 5], vertical_alignment="center")
                        with footer_cols[0]:
                            action_label = "OCULTAR RECIBO  ↑" if selected else "VER RECIBO  ↓"
                            clicked = st.button(
                                action_label,
                                key=f"select_slip_{sid}",
                                type="tertiary",
                                use_container_width=False,
                            )
                            if clicked:
                                st.session_state["manage_slip_id"] = None if selected else sid
                                st.rerun()

                    if st.session_state.get("pending_delete_slip_id") == sid:
                        st.warning(
                            f"Eliminar o Boletim #{row_number:03d}? "
                            f"Esta ação também apaga as {count} seleção(ões) associadas e não pode ser anulada."
                        )
                        confirm_cols = st.columns([1, 1, 5], vertical_alignment="center")
                        with confirm_cols[0]:
                            st.button(
                                "Sim, eliminar",
                                key=f"confirm_delete_slip_{sid}",
                                type="primary",
                                on_click=confirm_delete_slip,
                                args=(sid,),
                            )
                        with confirm_cols[1]:
                            if st.button("Cancelar", key=f"cancel_delete_slip_{sid}"):
                                st.session_state["pending_delete_slip_id"] = None
                                st.rerun()

                    if st.session_state.get("manage_slip_id") == sid:
                        chosen_slip_id = sid
                        chosen_slip = slip
                        chosen_legs = group
                        chosen_idx = slips_df.index[slips_df["slip_id"].astype(str) == sid][0]
        
                        st.markdown(
                            '<div style="display:flex;align-items:center;justify-content:space-between;'
                            'gap:12px;margin:8px 0 12px 0;padding:10px 13px;'
                            'background:linear-gradient(100deg,#14283a,#101c29);'
                            'border:1px solid #29445a;border-radius:10px;">'
                            '<div><div style="font-size:.68rem;font-weight:800;letter-spacing:.09em;'
                            'color:#7fc8f2;text-transform:uppercase;">Recibo da aposta</div>'
                            '<div style="font-size:1rem;font-weight:850;color:#f3f7fb;margin-top:3px;">'
                            f'{escape(str(chosen_slip["bet_type"]))} · {len(chosen_legs)} '
                            f'{"seleção" if len(chosen_legs) == 1 else "seleções"}</div></div>'
                            '<div style="font-size:.75rem;color:#a9bdcf;text-align:right;">'
                            f'{escape(pd.to_datetime(chosen_slip.get("saved_at", ""), errors="coerce").strftime("%d-%m-%Y · %H:%M") if pd.notna(pd.to_datetime(chosen_slip.get("saved_at", ""), errors="coerce")) else str(chosen_slip.get("saved_at", "")))}'
                            '</div></div>',
                            unsafe_allow_html=True,
                        )

                        if not chosen_legs.empty:
                            display_rows = []
                            for _, leg in chosen_legs.iterrows():
                                raw_date = str(leg.get("game_date", ""))
                                parsed_date = pd.to_datetime(raw_date, errors="coerce")
                                date_text = parsed_date.strftime("%d-%m-%Y") if not pd.isna(parsed_date) else raw_date
                                result_text = str(leg.get("result", "Pendente"))
                                result_norm = result_text.strip().lower()
                                if result_norm in ("acertou", "ganho", "won"):
                                    result_bg, result_fg, result_border = "#153b2c", "#70e0aa", "#286c4d"
                                elif result_norm in ("falhou", "perdido", "lost"):
                                    result_bg, result_fg, result_border = "#3b2027", "#ff8792", "#82404b"
                                else:
                                    result_bg, result_fg, result_border = "#3b321b", "#f2d27b", "#806b35"
                                market_text = str(leg.get("selection", "")).strip()
                                market_norm = market_text.casefold()
                                if "over" in market_norm or "mais" in market_norm:
                                    market_class = "market-over"
                                elif "under" in market_norm or "menos" in market_norm:
                                    market_class = "market-under"
                                elif (
                                    "moneyline" in market_norm
                                    or "vencedor" in market_norm
                                    or "winner" in market_norm
                                    or market_norm in {"casa", "fora", "home", "away", "1", "2"}
                                ):
                                    market_class = "market-moneyline"
                                else:
                                    market_class = "market-other"

                                away_score = pd.to_numeric(leg.get("away_score"), errors="coerce")
                                home_score = pd.to_numeric(leg.get("home_score"), errors="coerce")
                                final_score_text = (
                                    f"{int(away_score)}–{int(home_score)}"
                                    if pd.notna(away_score) and pd.notna(home_score) else "—"
                                )
                                display_rows.append(
                                    f'<tr>'
                                    f'<td class="date">{escape(date_text)}</td>'
                                    f'<td class="match"><span>{escape(str(leg.get("away_team", "")))}</span>'
                                    f'<span class="at">@</span><span>{escape(str(leg.get("home_team", "")))}</span></td>'
                                    f'<td><span class="market {market_class}">{escape(market_text)}</span></td>'
                                    f'<td class="num">{escape(str(leg.get("line"))) if pd.notna(leg.get("line")) else "—"}</td>'
                                    f'<td class="num odds">{float(leg.get("odds", 0) or 0):.2f}</td>'
                                    f'<td class="num">{escape(final_score_text)}</td>'
                                    f'<td class="num">{escape(str(int(float(leg.get("actual_total"))))) if pd.notna(leg.get("actual_total")) else "—"}</td>'
                                    f'<td><span class="result" style="background:{result_bg};color:{result_fg};'
                                    f'border:1px solid {result_border};">{escape(result_text)}</span></td>'
                                    f'</tr>'
                                )

                            table_html = (
                                '<div class="receipt-table-wrap"><table class="receipt-table">'
                                '<thead><tr><th>DATA</th><th>CONFRONTO</th><th>MERCADO</th>'
                                '<th class="right">LINHA</th><th class="right">ODD</th>'
                                '<th class="right">RESULTADO FINAL</th><th class="right">TOTAL FINAL</th>'
                                '<th>RESULTADO</th></tr></thead>'
                                '<tbody>' + ''.join(display_rows) + '</tbody></table></div>'
                            )
                            st.markdown(table_html, unsafe_allow_html=True)

                        with st.expander("Editar dados do boletim"):
                            with st.form(f"edit_slip_form_{chosen_slip_id}"):
                                e1, e2 = st.columns(2)
                                with e1:
                                    edit_stake = st.number_input(
                                        "Stake (€)", min_value=0.01,
                                        value=max(0.01, float(chosen_slip["stake"])), step=0.5
                                    )
                                with e2:
                                    edit_combined = st.number_input(
                                        "Odd total", min_value=1.01,
                                        value=max(1.01, float(chosen_slip["combined_odds"])), step=0.01
                                    )
                                update_slip = st.form_submit_button("Guardar alterações", type="primary")
                            if update_slip:
                                slips_df.at[chosen_idx, "stake"] = edit_stake
                                slips_df.at[chosen_idx, "combined_odds"] = edit_combined
                                slips_df, legs_df, _ = reconcile_slips(slips_df, legs_df, result_games_by_id)
                                save_slip_data(slips_df, legs_df)
                                st.success("Boletim atualizado.")
                                st.rerun()
        
                        with st.expander("Corrigir ou eliminar uma seleção"):
                            if chosen_legs.empty:
                                st.caption("Este boletim não tem seleções.")
                            else:
                                leg_ids = chosen_legs["selection_id"].astype(str).tolist()
                                chosen_leg_id = st.selectbox(
                                    "Seleção",
                                    leg_ids,
                                    format_func=lambda x: (
                                        f"{chosen_legs.loc[chosen_legs['selection_id'].astype(str)==x, 'away_team'].iloc[0]} @ "
                                        f"{chosen_legs.loc[chosen_legs['selection_id'].astype(str)==x, 'home_team'].iloc[0]}"
                                    ),
                                    key=f"manage_leg_id_{chosen_slip_id}"
                                )
                                li = legs_df.index[legs_df["selection_id"].astype(str) == chosen_leg_id][0]
                                selected_leg = legs_df.loc[li]
                                with st.form(f"edit_leg_form_{chosen_slip_id}"):
                                    q1, q2, q3 = st.columns(3)
                                    current_selection = str(selected_leg["selection"])
                                    current_is_ml = "moneyline" in current_selection.casefold()
                                    with q1:
                                        edit_market = st.selectbox(
                                            "Mercado", ["Over", "Under", "Moneyline"],
                                            index=2 if current_is_ml else (0 if current_selection == "Over" else 1)
                                        )
                                    edit_winner = None
                                    if edit_market == "Moneyline":
                                        with q2:
                                            edit_winner = st.selectbox(
                                                "Vencedor",
                                                ["Casa", "Visitante"],
                                                index=1 if "visitante" in current_selection.casefold() or "away" in current_selection.casefold() else 0
                                            )
                                        edit_line = None
                                    else:
                                        with q2:
                                            current_line = pd.to_numeric(selected_leg.get("line"), errors="coerce")
                                            edit_line = st.number_input(
                                                "Linha", min_value=100.0, max_value=350.0,
                                                value=float(current_line) if not pd.isna(current_line) else 220.5,
                                                step=0.5
                                            )
                                    with q3:
                                        edit_leg_odds = st.number_input(
                                            "Odd individual", min_value=1.01, max_value=100.0,
                                            value=float(selected_leg["odds"]), step=0.01
                                        )
                                    update_leg = st.form_submit_button("Guardar correção")
                                if update_leg:
                                    legs_df.at[li, "selection"] = (
                                        f"Moneyline · {edit_winner}" if edit_market == "Moneyline" else edit_market
                                    )
                                    legs_df.at[li, "line"] = edit_line
                                    legs_df.at[li, "odds"] = edit_leg_odds
                                    legs_df.at[li, "result"] = "Pendente"
                                    legs_df.at[li, "actual_total"] = None
                                    slips_df, legs_df, _ = reconcile_slips(slips_df, legs_df, result_games_by_id)
                                    save_slip_data(slips_df, legs_df)
                                    st.success("Seleção corrigida e resultado recalculado.")
                                    st.rerun()
        
                                confirm_leg_delete = st.checkbox(
                                    "Confirmo que quero eliminar apenas esta seleção",
                                    key=f"confirm_leg_delete_{chosen_slip_id}"
                                )
                                if st.button(
                                    "Eliminar seleção", key=f"delete_one_leg_{chosen_slip_id}",
                                    disabled=not confirm_leg_delete
                                ):
                                    legs_df = legs_df[legs_df["selection_id"].astype(str) != chosen_leg_id].copy()
                                    if legs_df[legs_df["slip_id"].astype(str) == chosen_slip_id].empty:
                                        slips_df = slips_df[slips_df["slip_id"].astype(str) != chosen_slip_id].copy()
                                        st.session_state["manage_slip_id"] = None
                                    save_slip_data(slips_df, legs_df)
                                    st.success("Seleção eliminada.")
                                    st.rerun()
        
                        with st.expander("Eliminar boletim completo"):
                            st.warning("Esta ação elimina o boletim selecionado e todas as respetivas seleções.")
                            confirm_slip_delete = st.checkbox(
                                "Confirmo a eliminação definitiva deste boletim",
                                key=f"confirm_slip_delete_{chosen_slip_id}"
                            )
                            if st.button(
                                "Eliminar boletim", key=f"delete_whole_slip_{chosen_slip_id}",
                                disabled=not confirm_slip_delete
                            ):
                                slips_df = slips_df[slips_df["slip_id"].astype(str) != chosen_slip_id].copy()
                                legs_df = legs_df[legs_df["slip_id"].astype(str) != chosen_slip_id].copy()
                                st.session_state["manage_slip_id"] = None
                                save_slip_data(slips_df, legs_df)
                                st.success("Boletim e seleções eliminados.")
                                st.rerun()
        
            with st.expander("Gestão de dados · Exportações"):
                st.caption("Pode exportar os boletins e as seleções separadamente. O ZIP completo continua disponível como cópia conjunta.")
                export_col1, export_col2, export_col3 = st.columns(3, gap="small")
                with export_col1:
                    st.download_button(
                        "📄  Boletins CSV",
                        data=slips_df[SLIP_COLUMNS].to_csv(index=False).encode("utf-8-sig"),
                        file_name="nba_bet_slips.csv",
                        mime="text/csv",
                        key="export_slips_compact",
                        help="Exportar apenas os boletins registados",
                    )
                with export_col2:
                    st.download_button(
                        "🎯  Seleções CSV",
                        data=legs_df[LEG_COLUMNS].to_csv(index=False).encode("utf-8-sig"),
                        file_name="nba_bet_selections.csv",
                        mime="text/csv",
                        key="export_legs_compact",
                        help="Exportar apenas as seleções dos boletins",
                    )
                backup_buffer = io.BytesIO()
                with zipfile.ZipFile(backup_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as backup_zip:
                    backup_zip.writestr(
                        "nba_bet_slips.csv",
                        slips_df[SLIP_COLUMNS].to_csv(index=False).encode("utf-8-sig")
                    )
                    backup_zip.writestr(
                        "nba_bet_selections.csv",
                        legs_df[LEG_COLUMNS].to_csv(index=False).encode("utf-8-sig")
                    )
                with export_col3:
                    st.download_button(
                        "📦  Backup ZIP",
                        data=backup_buffer.getvalue(),
                        file_name="nba_bet_backup_completo.zip",
                        mime="application/zip",
                        key="export_full_backup_compact",
                        help="Exportar os dois ficheiros num único backup",
                    )
                if github_configured():
                    st.caption("Persistência GitHub ativa: os dois CSV são sincronizados no arranque e guardados juntos num único commit.")
                else:
                    st.warning("Persistência GitHub não configurada. As apostas não podem ser guardadas com segurança até configurar os Secrets.")

        
        

if active_page == "Estatísticas":
    with main_area:
        st.markdown("## Estatísticas das apostas")
        st.markdown("### Desempenho das apostas reais")
        if slips_df.empty:
            st.info("As estatísticas aparecem depois de registares os primeiros boletins.")
        else:
            settled_slips = slips_df[slips_df["status"].isin(["Ganho", "Perdido", "Push"])]
            total_stake = pd.to_numeric(slips_df["stake"], errors="coerce").fillna(0).sum()
            settled_stake = pd.to_numeric(settled_slips["stake"], errors="coerce").fillna(0).sum()
            total_pnl = pd.to_numeric(settled_slips["profit_loss"], errors="coerce").fillna(0).sum()
            roi = (total_pnl / settled_stake * 100) if settled_stake > 0 else 0.0
        
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Boletins", len(slips_df))
            m2.metric("Avaliados", len(settled_slips))
            m3.metric("Total apostado", f"€{total_stake:.2f}")
            m4.metric("Lucro / prejuízo", f"€{total_pnl:.2f}", f"ROI {roi:.1f}%")
        
            st.divider()
            st.markdown("#### Resultados por seleção")
            all_settled_legs = legs_df[legs_df["result"].isin(["Acertou", "Falhou", "Push"])]
            wins = int((all_settled_legs["result"] == "Acertou").sum()) if not all_settled_legs.empty else 0
            losses = int((all_settled_legs["result"] == "Falhou").sum()) if not all_settled_legs.empty else 0
            pushes = int((all_settled_legs["result"] == "Push").sum()) if not all_settled_legs.empty else 0
            hit_rate = wins / (wins + losses) * 100 if wins + losses else 0.0
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Seleções avaliadas", len(all_settled_legs))
            x2.metric("Acertos", wins)
            x3.metric("Falhas", losses)
            x4.metric("Push", pushes)
            st.caption(
                f"Taxa de acerto: {hit_rate:.1f}% (exclui Push). "
                "A taxa de acerto das seleções não é a rentabilidade dos boletins."
            )
            if pushes:
                st.info(
                    "Em múltiplas com Push, o retorno é estimado com as odds das seleções vencedoras. "
                    "A liquidação real pode depender das regras do bookmaker."
                )
        
        

if active_page == "Ver ROI":
    with main_area:
        st.markdown("## Rentabilidade — ROI")
        if slips_df.empty:
            st.info("Ainda não existem boletins registados.")
        else:
            settled_roi = slips_df[slips_df["status"].isin(["Ganho", "Perdido", "Push"])].copy()
            settled_roi["stake_num"] = pd.to_numeric(settled_roi["stake"], errors="coerce").fillna(0)
            settled_roi["pnl_num"] = pd.to_numeric(settled_roi["profit_loss"], errors="coerce").fillna(0)
            stake_total = settled_roi["stake_num"].sum()
            pnl_total = settled_roi["pnl_num"].sum()
            roi_total = pnl_total / stake_total * 100 if stake_total > 0 else 0.0
            pending_count = int((slips_df["status"] == "Pendente").sum())
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("ROI", f"{roi_total:.2f}%")
            r2.metric("Lucro / prejuízo", f"€{pnl_total:.2f}")
            r3.metric("Total apostado (avaliado)", f"€{stake_total:.2f}")
            r4.metric("Boletins pendentes", pending_count)
            if not settled_roi.empty:
                settled_roi["Mês"] = pd.to_datetime(settled_roi["saved_at"], errors="coerce").dt.to_period("M").astype(str)
                monthly = settled_roi.groupby("Mês", as_index=False).agg(Apostado=("stake_num", "sum"), Resultado=("pnl_num", "sum"))
                monthly["ROI (%)"] = monthly.apply(lambda row: row["Resultado"] / row["Apostado"] * 100 if row["Apostado"] > 0 else 0, axis=1)
                st.markdown("### Evolução mensal")
                st.dataframe(monthly, use_container_width=True, hide_index=True)
                st.line_chart(monthly.set_index("Mês")[["ROI (%)"]], use_container_width=True)
            else:
                st.info("O ROI será calculado quando existirem boletins avaliados.")

if active_page == "Histórico":
    with main_area:
        st.markdown("## Resultados e avaliação automática")
        st.caption("Os resultados finais vêm da API-Sports NBA. As previsões são congeladas automaticamente a partir de T−5 quando a aplicação executa uma atualização.")
        
        log_df = load_log()
        if log_df.empty:
            st.info("Ainda não existem previsões congeladas. Mantém a aplicação aberta ou atualiza-a perto do início dos jogos.")
        else:
            finished_ml = log_df[log_df["moneyline_result"].isin(["Acertou", "Falhou"])]
            finished_total = log_df[log_df["total_result"].isin(["Acertou", "Falhou"])]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                rate = (finished_ml["moneyline_result"] == "Acertou").mean() * 100 if not finished_ml.empty else 0
                st.metric("Acerto Moneyline", f"{rate:.1f}%", f"{len(finished_ml)} avaliados")
            with c2:
                rate = (finished_total["total_result"] == "Acertou").mean() * 100 if not finished_total.empty else 0
                st.metric("Acerto Over/Under", f"{rate:.1f}%", f"{len(finished_total)} avaliados")
            with c3:
                st.metric("Previsões congeladas", int(len(log_df)))
            with c4:
                st.metric("Resultados finais", int(log_df["game_status"].eq("Final").sum()))
        
            history = log_df.copy()
            history["Resultado"] = history.apply(
                lambda r: f"{int(r['home_score'])}–{int(r['away_score'])}" if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")) else "—",
                axis=1
            )
            history["Total real"] = pd.to_numeric(history["actual_total"], errors="coerce")
            history["Linha de referência/registada"] = pd.to_numeric(history["total_line"], errors="coerce")
            history["Prob. casa"] = pd.to_numeric(history["p_home"], errors="coerce").map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            history["Prob. visitante"] = pd.to_numeric(history["p_away"], errors="coerce").map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
            display_cols = [
                "game_date", "home_team", "away_team", "Resultado",
                "moneyline_pick", "moneyline_result", "Prob. casa", "Prob. visitante",
                "total_reference", "Linha de referência/registada", "total_pick", "Total real", "total_result", "capture_mode"
            ]
            st.caption("As novas previsões usam a referência fixa de 220,5. Os registos históricos anteriores são preservados tal como estavam, incluindo linhas que possam ter sido introduzidas manualmente.")
            st.dataframe(
                history[display_cols].rename(columns={
                    "game_date": "Data", "home_team": "Casa", "away_team": "Visitante",
                    "moneyline_pick": "Moneyline previsto", "moneyline_result": "Moneyline",
                    "total_reference": "Total referência", "total_pick": "Over/Under previsto",
                    "total_result": "Over/Under", "capture_mode": "Registo"
                }),
                use_container_width=True, hide_index=True
            )
            st.download_button(
                "Descarregar backup CSV",
                data=log_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="nba_prediction_log_backup.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.warning("Limitação importante: o registo automático só acontece quando o Streamlit executa uma atualização. Para garantir o congelamento exato em T−5, é necessário um agendador externo ou manter uma atualização periódica ativa. A linha de referência fixa de 220,5 serve apenas para avaliar a previsão estatística de forma consistente; não representa uma linha de bookmaker. As apostas reais são registadas separadamente com a linha e odd efetivamente escolhidas.")
            pending_final = log_df[~log_df["game_status"].astype(str).str.lower().eq("final")]
            if not pending_final.empty:
                st.caption(f"Há {len(pending_final)} previsões ainda sem resultado final confirmado pela API. Os resultados são reconciliados pelas datas guardadas no histórico em cada atualização.")