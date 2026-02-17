import json
import os
import re
import tempfile
import time
import html
import base64
from difflib import SequenceMatcher
from datetime import datetime, date
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlparse

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
except Exception:
    go = None

from extrair_dados_fundos import get_historical_data, get_xid_from_isin


# Configurações básicas e caminhos de cache/relatórios
BASE_URL = "https://markets.ft.com"
FT_SEARCH_API_URL = f"{BASE_URL}/data/searchapi/searchsecurities"
BASE_START_DATE = "2000/01/01"
CACHE_DIR = Path("cache")
NAMES_FILE = CACHE_DIR / "fund_names.json"
PORTFOLIO_FILE = Path("carteiras.json")
REPORTS_DIR = Path("reports")
FUNDS_CATALOG_FILE = Path("fundos_completos_fundname_isin_roa.csv")
# Tenta várias moedas antes de falhar; sem sufixo é fallback final.
XID_CURRENCY_FALLBACKS = ["USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", None]
XID_LOOKUP_RETRIES = 2
CACHE_MAX_AGE_DAYS = 1
DEFAULT_CHART_COLORS = [
    "#00A8E8",
    "#F45D48",
    "#2EC4B6",
    "#FF9F1C",
    "#A06CD5",
    "#7AE582",
    "#FF5D8F",
    "#6C8EAD",
    "#FFD166",
    "#4CC9F0",
]
ISIN_REGEX = re.compile(r"[A-Z]{2}[A-Z0-9]{10}")
SYMBOL_REGEX = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,19}(?::[A-Z0-9]{2,8}){0,2}")


@st.cache_data(show_spinner=False)
def load_funds_catalog(path_str: str) -> pd.DataFrame:
    """Carrega base de fundos (FundName, Isin, DistributionRoaFee)."""
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame(columns=["FundName", "Isin", "DistributionRoaFee"])

    last_exc: Exception | None = None
    for sep in (";", ",", "\t", None):
        try:
            if sep is None:
                df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
            else:
                df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            break
        except Exception as exc:
            last_exc = exc
    else:
        raise RuntimeError(f"Falha ao ler catálogo de fundos: {last_exc}") from None

    normalized = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    fund_col = normalized.get("fundname")
    isin_col = normalized.get("isin")
    roa_col = normalized.get("distributionroafee")
    if not fund_col or not isin_col or not roa_col:
        raise RuntimeError(
            "Catálogo de fundos inválido. Colunas esperadas: FundName, Isin, DistributionRoaFee."
        )

    out = df[[fund_col, isin_col, roa_col]].copy()
    out.columns = ["FundName", "Isin", "DistributionRoaFee"]
    out["FundName"] = out["FundName"].astype(str).str.strip()
    out["Isin"] = out["Isin"].astype(str).str.strip().str.upper()
    out["DistributionRoaFee"] = pd.to_numeric(out["DistributionRoaFee"], errors="coerce")
    out = out[(out["FundName"] != "") & (out["Isin"] != "")]
    out = out.drop_duplicates(subset=["Isin"], keep="first")
    out["FundOption"] = out["FundName"] + " | " + out["Isin"]
    return out.sort_values("FundName").reset_index(drop=True)


def _extract_isin_from_option(option: str, option_to_isin: dict[str, str]) -> str | None:
    """Converte opção selecionada em identificador (ISIN ou symbol)."""
    text = str(option).strip()
    if not text:
        return None
    if text in option_to_isin:
        return option_to_isin[text]
    upper = text.upper()

    if "|" in text:
        candidate = text.split("|")[-1].strip().upper()
        if ISIN_REGEX.fullmatch(candidate) or SYMBOL_REGEX.fullmatch(candidate):
            return candidate

    isin_match = ISIN_REGEX.search(upper)
    if isin_match:
        return isin_match.group(0)

    symbol_match = re.search(r"[A-Z0-9][A-Z0-9.\-]{0,19}(?::[A-Z0-9]{2,8}){0,2}", upper)
    if symbol_match and SYMBOL_REGEX.fullmatch(symbol_match.group(0)):
        return symbol_match.group(0)
    return upper


def _normalize_typed_identifier(text: str) -> str:
    return str(text).strip().upper()


def _is_valid_typed_identifier(text: str) -> bool:
    value = _normalize_typed_identifier(text)
    return bool(ISIN_REGEX.fullmatch(value) or SYMBOL_REGEX.fullmatch(value))


def _is_manual_search_candidate(text: str) -> bool:
    """Define quando a entrada digitada deve aparecer como opção manual no dropdown."""
    value = _normalize_typed_identifier(text)
    if not value:
        return False
    if ISIN_REGEX.fullmatch(value):
        return True
    if not SYMBOL_REGEX.fullmatch(value):
        return False
    if ":" in value:
        return True
    # Para símbolo sem ":": exige dígito (ex.: PETR4) para evitar ruído
    # em buscas por nome de fundo (ex.: "pimco").
    return any(ch.isdigit() for ch in value)


def _guess_manual_asset_type(identifier: str) -> str:
    """Heurística para fallback de categoria em entradas manuais."""
    value = _normalize_typed_identifier(identifier)
    if ISIN_REGEX.fullmatch(value):
        return "funds"
    if ":" in value:
        suffix = value.split(":")[-1]
        if suffix == "USD":
            return "etfs"
        return "equities"
    return "equities"


def _parse_funds_input(
    funds_df: pd.DataFrame,
    option_to_isin: dict[str, str],
    isin_to_roa: dict[str, float],
) -> tuple[list[str], np.ndarray, list[str], list[str], dict[str, float]]:
    """Extrai isins/pesos/classes/opções selecionadas e mapa de ROA por ISIN."""
    isins: list[str] = []
    weights: list[float] = []
    classes: list[str] = []
    selected_options: list[str] = []
    roa_by_isin: dict[str, float] = {}

    for _, row in funds_df.iterrows():
        raw_option = row.get("Ativo", row.get("Fundo", ""))
        if pd.isna(raw_option):
            continue
        option = str(raw_option).strip()
        if not option:
            continue
        raw_weight = row.get("Alocacao_%", np.nan)
        if pd.isna(raw_weight):
            continue
        try:
            weight = float(raw_weight)
        except Exception:
            continue
        if not np.isfinite(weight):
            continue

        isin = _extract_isin_from_option(option, option_to_isin)
        if not isin:
            continue

        raw_cls = row.get("Classe", row.get("Categoria", ""))
        cls = "" if pd.isna(raw_cls) else str(raw_cls).strip()
        class_name = cls if cls else "Outros"

        isins.append(isin)
        weights.append(weight)
        classes.append(class_name)
        selected_options.append(option)

        roa = _to_float(isin_to_roa.get(isin))
        if np.isfinite(roa):
            roa_by_isin[isin] = float(roa)

    return isins, np.array(weights, dtype=float), classes, selected_options, roa_by_isin


def _resolve_option_roa(option: str, option_to_isin: dict[str, str], isin_to_roa: dict[str, float]) -> float | None:
    """Retorna ROA do fundo selecionado (ou None quando indisponível)."""
    isin = _extract_isin_from_option(option, option_to_isin)
    if not isin:
        return None
    value = _to_float(isin_to_roa.get(isin))
    if not np.isfinite(value):
        return None
    return float(value)


def _fmt_optional_number(value: object, decimals: int = 2) -> str:
    """Formata número finito ou retorna vazio."""
    num = _to_float(value)
    if not np.isfinite(num):
        return ""
    return f"{num:.{decimals}f}"


def _search_catalog_options(catalog_df: pd.DataFrame, query: str, limit: int = 20) -> list[str]:
    """Busca opções por nome/ISIN no catálogo."""
    q = str(query).strip()
    if not q or catalog_df.empty:
        return []
    mask = (
        catalog_df["FundName"].str.contains(q, case=False, na=False)
        | catalog_df["Isin"].str.contains(q, case=False, na=False)
    )
    return catalog_df.loc[mask, "FundOption"].head(limit).tolist()


def _normalize_asset_type(asset_class: str | None) -> str | None:
    value = str(asset_class or "").strip().lower()
    if value in {"equity", "equities"}:
        return "equities"
    if value in {"fund", "funds"}:
        return "funds"
    if value in {"etf", "etfs"}:
        return "etfs"
    return None


def _fallback_category_from_asset_type(asset_type: str | None) -> str:
    normalized = _normalize_asset_type(asset_type) or "funds"
    if normalized == "equities":
        return "Stock"
    if normalized == "etfs":
        return "ETF"
    return "Fundo"


def _is_light_theme_active() -> bool:
    """Resolve tema efetivo considerando escolha do usuário."""
    choice = str(st.session_state.get("ui_theme_option", "Escuro")).strip().lower()
    return choice == "claro"


def _local_fund_match_score(name: str, isin: str, query: str) -> float:
    """Score simples para ranquear fundos do CSV por similaridade."""
    q = str(query).strip().upper()
    if not q:
        return 0.0
    name_u = str(name).upper()
    isin_u = str(isin).upper()

    score = 0.0
    if isin_u.startswith(q):
        score += 800
    elif q in isin_u:
        score += 520

    if name_u.startswith(q):
        score += 460
    elif q in name_u:
        score += 300

    score += SequenceMatcher(None, q, name_u).ratio() * 120
    return score


def _search_local_funds_options(query: str, limit: int = 8) -> list[dict]:
    """Busca por similaridade na base local de fundos (CSV)."""
    q = str(query).strip()
    if not q:
        return []

    try:
        catalog_df = load_funds_catalog(str(FUNDS_CATALOG_FILE))
    except Exception:
        return []
    if catalog_df.empty:
        return []

    ranked: list[tuple[float, str, str]] = []
    for _, row in catalog_df.iterrows():
        name = str(row.get("FundName", "")).strip()
        isin = str(row.get("Isin", "")).strip().upper()
        if not name or not isin:
            continue
        score = _local_fund_match_score(name, isin, q)
        if score >= 250:
            ranked.append((score, name, isin))

    ranked.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    seen: set[str] = set()
    for score, name, isin in ranked:
        if isin in seen:
            continue
        seen.add(isin)
        out.append(
            {
                "asset_type": "funds",
                "asset_label": "Funds",
                "name": name,
                "symbol": isin,  # mantém ISIN para parsing posterior
                "xid": "",
                "label": f"[FUND] {name} | {isin}",
                "source": "csv",
                "score": score,
            }
        )
        if len(out) >= limit:
            break
    return out


@st.cache_data(show_spinner=False, ttl=300)
def search_ft_securities(query: str, limit_per_asset: int = 8) -> list[dict]:
    """Busca sugestões no FT (Funds/Equities/ETFs) com symbol e xid."""
    import requests

    q = str(query).strip()
    if not q:
        return []

    r = requests.get(
        FT_SEARCH_API_URL,
        params={"query": q},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()
    raw_items = payload.get("data", {}).get("security", [])
    if not isinstance(raw_items, list):
        return []

    buckets: dict[str, list[dict]] = {"equities": [], "funds": [], "etfs": []}
    seen: set[tuple[str, str]] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        asset_type = _normalize_asset_type(item.get("assetClass"))
        symbol = str(item.get("symbol", "")).strip().upper()
        name = str(item.get("name", "")).strip()
        xid = str(item.get("xid", "")).strip()
        if not asset_type or not symbol or not xid:
            continue
        key = (asset_type, symbol)
        if key in seen:
            continue
        seen.add(key)
        buckets[asset_type].append(
            {
                "asset_type": asset_type,
                "asset_label": asset_type.capitalize(),
                "name": name or symbol,
                "symbol": symbol,
                "xid": xid,
                "label": f"[FT {asset_type.capitalize()}] {name or symbol} | {symbol}",
            }
        )

    out: list[dict] = []
    for asset_type in ("equities", "funds", "etfs"):
        out.extend(buckets[asset_type][:limit_per_asset])
    return out


def _ft_searchbox_options(
    query: str,
    key_prefix: str,
    limit_per_asset: int = 6,
    asset_filter: str = "Todos",
) -> list[str]:
    """Opções para autocomplete em tempo real com flag por classe."""
    items = search_ft_securities(query, limit_per_asset=limit_per_asset)
    options: list[str] = []
    mapping: dict[str, dict] = {}
    by_asset = {"equities": [], "etfs": [], "funds": []}
    for item in items:
        asset = _normalize_asset_type(item.get("asset_type"))
        symbol = str(item.get("symbol", "")).strip().upper()
        if not asset or asset not in by_asset or not symbol:
            continue

        # Regras de exibição:
        # - Fundos: somente os cotados em USD (sufixo final :USD)
        # - ETFs: somente os cotados em USD (sufixo final :USD)
        # - Equities: somente bolsas NSQ e NYS
        if asset == "funds" and symbol.split(":")[-1] != "USD":
            continue
        if asset == "etfs" and symbol.split(":")[-1] != "USD":
            continue
        if asset == "equities" and symbol.split(":")[-1] not in {"NSQ", "NYS"}:
            continue

        if asset in by_asset:
            by_asset[asset].append(item)

    flag_map = {"equities": "[EQ]", "etfs": "[ETF]", "funds": "[FUND]"}
    selected_assets = {
        "Todos": ("equities", "etfs", "funds"),
        "Stocks": ("equities",),
        "ETFs": ("etfs",),
        "Fundos": ("funds",),
    }.get(asset_filter, ("equities", "etfs", "funds"))

    # Quando filtro inclui fundos, combina também sugestões do CSV local
    # (melhor cobertura para fundos não bem rankeados pela API do FT).
    local_fund_items = _search_local_funds_options(query, limit=max(limit_per_asset, 8))
    local_fund_isins = {
        str(item.get("symbol", "")).split(":")[0].upper()
        for item in local_fund_items
        if str(item.get("symbol", "")).strip()
    }

    for asset_key in ("equities", "etfs", "funds"):
        if asset_key not in selected_assets:
            continue

        if asset_key == "funds":
            for item in local_fund_items:
                opt = f"[FUND] {item['name']} | {item['symbol']}"
                if opt in mapping:
                    continue
                options.append(opt)
                mapping[opt] = item

        for item in by_asset[asset_key]:
            if asset_key == "funds":
                # Evita duplicar fundo FT quando já temos o mesmo ISIN do CSV.
                base_symbol = str(item.get("symbol", "")).split(":")[0].upper()
                if base_symbol in local_fund_isins:
                    continue
            opt = f"{flag_map[asset_key]} {item['name']} | {item['symbol']}"
            options.append(opt)
            mapping[opt] = item

    # Opção manual para funcionar mesmo quando o filtro esconde os resultados do FT.
    typed = _normalize_typed_identifier(query)
    if _is_manual_search_candidate(typed):
        manual_opt = f"[DIGITE] {typed}"
        if manual_opt not in mapping:
            options.insert(0, manual_opt)
            mapping[manual_opt] = {
                "asset_type": _guess_manual_asset_type(typed),
                "asset_label": "Manual",
                "name": typed,
                "symbol": typed,
                "xid": "",
                "manual": True,
            }

    st.session_state[f"{key_prefix}_ft_map"] = mapping
    return options


def _live_ft_searchbox(
    label: str,
    placeholder: str,
    key_prefix: str,
    asset_filter: str = "Todos",
) -> dict | None:
    """Barra única com dropdown de sugestões; retorna item FT selecionado."""
    try:
        from streamlit_searchbox import st_searchbox
    except Exception:
        st.error(
            "Autocomplete em tempo real requer `streamlit-searchbox`. "
            "Instale com: `pip install -r requirements.txt`."
        )
        return None

    if label:
        st.caption(label)

    is_light = _is_light_theme_active()
    style_overrides = None
    if is_light:
        style_overrides = {
            "wrapper": {
                "backgroundColor": "#ffffff",
                "border": "0",
                "borderRadius": "10px",
                "padding": "0",
            },
            "searchbox": {
                "control": {
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #cbd5e1",
                    "borderColor": "#cbd5e1",
                    "borderRadius": "10px",
                    "boxShadow": "none",
                    "outline": "none",
                    "minHeight": "44px",
                    "&:hover": {"borderColor": "#94a3b8"},
                },
                "valueContainer": {"backgroundColor": "#ffffff"},
                "input": {"color": "#111827"},
                "singleValue": {"color": "#111827"},
                "placeholder": {"color": "#6b7280"},
                "menu": {
                    "backgroundColor": "#ffffff",
                    "border": "1px solid #cbd5e1",
                    "boxShadow": "none",
                    "marginTop": "0",
                },
                "menuList": {
                    "backgroundColor": "#ffffff",
                    "paddingTop": "0",
                    "paddingBottom": "0",
                    "marginTop": "0",
                },
                "option": {
                    "color": "#111827",
                    "backgroundColor": "#ffffff",
                    "highlightColor": "#f1f5f9",
                },
                "optionEmpty": "hidden",
            },
            "clear": {"fill": "#64748b", "stroke": "#64748b"},
            "dropdown": {"fill": "#64748b"},
        }
    else:
        style_overrides = {
            "wrapper": {
                "backgroundColor": "transparent",
                "border": "0",
                "borderRadius": "10px",
                "padding": "0",
            },
            "searchbox": {
                "control": {
                    "backgroundColor": "#0f172a",
                    "border": "1px solid #223041",
                    "borderColor": "#223041",
                    "borderRadius": "10px",
                    "boxShadow": "none",
                    "outline": "none",
                    "minHeight": "44px",
                    "&:hover": {"borderColor": "#475569"},
                },
                "valueContainer": {"backgroundColor": "#0f172a"},
                "input": {"color": "#e5e7eb"},
                "singleValue": {"color": "#e5e7eb"},
                "placeholder": {"color": "#94a3b8"},
                "menu": {"backgroundColor": "#0f172a", "border": "1px solid #334155"},
                "menuList": {
                    "backgroundColor": "#0f172a",
                    "paddingTop": "0",
                    "paddingBottom": "0",
                    "marginTop": "0",
                },
                "option": {
                    "color": "#e5e7eb",
                    "backgroundColor": "#0f172a",
                    "highlightColor": "#1f2937",
                },
                "optionEmpty": "hidden",
            },
            "clear": {"fill": "#94a3b8", "stroke": "#94a3b8"},
            "dropdown": {"fill": "#94a3b8"},
        }

    component_key = f"{key_prefix}_searchbox_{'light' if is_light else 'dark'}"
    selected = st_searchbox(
        search_function=lambda q: _ft_searchbox_options(q, key_prefix, asset_filter=asset_filter),
        label=None,
        placeholder=placeholder,
        clear_on_submit=False,
        style_overrides=style_overrides,
        key=component_key,
    )
    if not selected:
        return None
    mapping = st.session_state.get(f"{key_prefix}_ft_map", {})
    item = mapping.get(str(selected))
    if isinstance(item, dict):
        return item
    return None


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_ft_morningstar_category(
    identifier: str,
    symbol_hint: str | None = None,
    asset_type_hint: str | None = None,
) -> str:
    """Busca Morningstar category no Summary do FT; fallback para Fundo/ETF/Stock."""
    import requests
    from bs4 import BeautifulSoup

    fallback = _fallback_category_from_asset_type(asset_type_hint)
    symbol = str(symbol_hint or "").strip().upper()
    asset_type = _normalize_asset_type(asset_type_hint) or "funds"

    if not symbol:
        try:
            symbol, resolved_asset = resolve_ft_tearsheet_context(identifier)
            asset_type = _normalize_asset_type(resolved_asset) or asset_type
            fallback = _fallback_category_from_asset_type(asset_type)
        except Exception:
            return fallback

    summary_url = f"{BASE_URL}/data/{asset_type}/tearsheet/summary?s={quote_plus(symbol)}"
    try:
        r = requests.get(summary_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
    except Exception:
        return fallback

    soup = BeautifulSoup(r.text, "html.parser")
    tables = soup.select(
        "table.mod-ui-table.mod-ui-table--two-column.mod-profile-and-investment-app__table--profile, "
        "table.mod-profile-and-investment-app__table--profile, "
        "table.mod-ui-table.mod-ui-table--two-column"
    )

    for table in tables:
        for tr in table.select("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            label = re.sub(r"\s+", " ", cells[0].get_text(" ", strip=True)).strip().lower()
            if label == "morningstar category":
                value = cells[1].get_text(" ", strip=True)
                return value or fallback
    return fallback


@st.cache_data(show_spinner=False, ttl=86400)
def resolve_ft_symbol_for_isin(isin: str) -> str:
    """Resolve simbolo FT no formato ISIN:MOEDA para montar links estaveis."""
    import requests

    isin = str(isin).strip().upper()
    if not isin:
        return ""

    with requests.Session() as session:
        for currency in XID_CURRENCY_FALLBACKS:
            try:
                _, referer, _ = get_xid_from_isin(session, isin, currency=currency)
                parsed = urlparse(referer)
                symbol_values = parse_qs(parsed.query).get("s", [])
                if symbol_values and symbol_values[0]:
                    return symbol_values[0]
                if currency:
                    return f"{isin}:{currency}"
                return isin
            except Exception:
                continue
    return isin


@st.cache_data(show_spinner=False, ttl=86400)
def resolve_ft_tearsheet_context(identifier: str) -> tuple[str, str]:
    """Resolve symbol FT e tipo de tearsheet (funds/equities/etfs)."""
    import requests

    instrument = str(identifier).strip().upper()
    if not instrument:
        return "", "funds"

    with requests.Session() as session:
        for currency in XID_CURRENCY_FALLBACKS:
            try:
                _, referer, _ = get_xid_from_isin(session, instrument, currency=currency)
                parsed = urlparse(referer)
                path = parsed.path.lower()
                if "/data/equities/" in path:
                    asset_type = "equities"
                elif "/data/etfs/" in path:
                    asset_type = "etfs"
                else:
                    asset_type = "funds"

                symbol_values = parse_qs(parsed.query).get("s", [])
                symbol = symbol_values[0] if symbol_values and symbol_values[0] else instrument
                return symbol, asset_type
            except Exception:
                continue

    return instrument, "funds"


def make_ft_summary_link_from_isin(isin: str) -> str:
    symbol, asset_type = resolve_ft_tearsheet_context(isin)
    return f"{BASE_URL}/data/{asset_type}/tearsheet/summary?s={quote_plus(symbol)}"


def make_ft_risk_link_from_isin(isin: str) -> str:
    symbol, asset_type = resolve_ft_tearsheet_context(isin)
    return f"{BASE_URL}/data/{asset_type}/tearsheet/risk?s={quote_plus(symbol)}"


def _to_float(value: str | None) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace("%", "").replace("+", "").replace(",", ".")
    if not text:
        return np.nan
    try:
        return float(text)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False, ttl=43200)
def fetch_ft_risk_snapshot(isin: str) -> dict:
    """Extrai benchmark e medidas de risco 1Y/3Y/5Y da pagina Risk do FT."""
    import requests
    from bs4 import BeautifulSoup

    symbol, asset_type = resolve_ft_tearsheet_context(isin)
    risk_url = f"{BASE_URL}/data/{asset_type}/tearsheet/risk?s={quote_plus(symbol)}"
    r = requests.get(risk_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    out: dict[str, str | float | None] = {
        "risk_url": risk_url,
        "benchmark": None,
    }
    bench = soup.select_one(".mod-risk-measures__benchmark--dark")
    if bench:
        out["benchmark"] = bench.get_text(" ", strip=True)

    panel_map = {
        "1Y": "modriskmeasures1y-panel",
        "3Y": "modriskmeasures3y-panel",
        "5Y": "modriskmeasures5y-panel",
    }
    metrics_to_keep = {
        "alpha": "Alpha",
        "beta": "Beta",
        "informationratio": "Information ratio",
        "rsquared": "R squared",
        "sharperatio": "Sharpe ratio",
        "standarddeviation": "Standard deviation",
    }

    for horizon, panel_id in panel_map.items():
        panel = soup.find(id=panel_id)
        if not panel:
            continue
        for tr in panel.select("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            raw_name = cells[0].get_text(" ", strip=True)
            norm_name = re.sub(r"[^a-z]", "", raw_name.lower())
            if norm_name not in metrics_to_keep:
                continue
            fund_val = cells[1].get_text(" ", strip=True)
            cat_val = cells[2].get_text(" ", strip=True) if len(cells) >= 3 else None
            metric_name = metrics_to_keep[norm_name]
            out[f"{horizon}_{metric_name}"] = fund_val
            out[f"{horizon}_{metric_name}_float"] = _to_float(fund_val)
            if cat_val is not None:
                out[f"{horizon}_{metric_name}_cat"] = cat_val
                out[f"{horizon}_{metric_name}_cat_float"] = _to_float(cat_val)
    return out


def _today_str() -> str:
    """Data de hoje no formato esperado pelo endpoint do FT."""
    return datetime.today().strftime("%Y/%m/%d")


def _cache_path(isin: str) -> Path:
    """Arquivo de cache por ISIN."""
    safe_isin = "".join(ch for ch in isin if ch.isalnum() or ch in ("-", "_"))
    return CACHE_DIR / f"{safe_isin}_FT_Historical.csv"


def load_or_fetch_history(
    isin: str,
    start_date: str = BASE_START_DATE,
    end_date: str | None = None,
    max_age_days: int = 7,
    ft_hint: dict | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Busca histórico (com cache) e tenta salvar nome do fundo."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(isin)
    names = {}
    if NAMES_FILE.exists():
        with NAMES_FILE.open("r", encoding="utf-8") as f:
            names = json.load(f)

    if end_date is None:
        end_date = _today_str()

    # Cache válido usa direto
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days <= max_age_days:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            # Se nao tiver nome salvo, tenta buscar apenas o nome
            if isin not in names:
                try:
                    import requests
                    session = requests.Session()
                    fund_name = None
                    for currency in XID_CURRENCY_FALLBACKS:
                        try:
                            _, _, fund_name = get_xid_from_isin(session, isin, currency=currency)
                            if fund_name:
                                break
                        except Exception:
                            continue
                    if fund_name:
                        names[isin] = fund_name
                        with NAMES_FILE.open("w", encoding="utf-8") as f:
                            json.dump(names, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                finally:
                    try:
                        session.close()
                    except Exception:
                        pass
            return df, names.get(isin)

    # Cache inválido: baixa do FT e atualiza cache
    session = None
    try:
        import requests
        session = requests.Session()

        if ft_hint:
            hint_xid = str(ft_hint.get("xid", "")).strip()
            hint_symbol = str(ft_hint.get("symbol", "")).strip().upper()
            hint_asset = _normalize_asset_type(ft_hint.get("asset_type")) or "funds"
            hint_name = str(ft_hint.get("name", "")).strip() or None
            if hint_xid and hint_symbol:
                try:
                    referer = f"{BASE_URL}/data/{hint_asset}/tearsheet/historical?s={quote_plus(hint_symbol)}"
                    df = fetch_history_chunked(session, hint_xid, start_date, end_date, referer)
                    df.to_csv(cache_file, index=False)
                    if hint_name:
                        names[isin] = hint_name
                        with NAMES_FILE.open("w", encoding="utf-8") as f:
                            json.dump(names, f, ensure_ascii=False, indent=2)
                    return df, hint_name
                except Exception:
                    # Fallback para resolução tradicional (ISIN/symbol -> XID)
                    pass

        last_exc: Exception | None = None
        for attempt in range(1, XID_LOOKUP_RETRIES + 1):
            for currency in XID_CURRENCY_FALLBACKS:
                try:
                    xid, referer, fund_name = get_xid_from_isin(session, isin, currency=currency)
                    df = fetch_history_chunked(session, xid, start_date, end_date, referer)
                    df.to_csv(cache_file, index=False)
                    if fund_name:
                        names[isin] = fund_name
                        with NAMES_FILE.open("w", encoding="utf-8") as f:
                            json.dump(names, f, ensure_ascii=False, indent=2)
                    return df, fund_name
                except Exception as exc:
                    last_exc = exc
                    continue

            # FT pode responder "Search" temporariamente; troca sessao e tenta de novo.
            if attempt < XID_LOOKUP_RETRIES:
                try:
                    session.close()
                except Exception:
                    pass
                session = requests.Session()
                time.sleep(0.8 * attempt)

        raise RuntimeError(
            f"Falha ao buscar dados para {isin} com moedas alternativas. Ultimo erro: {last_exc}"
        ) from None
    except Exception as exc:
        if cache_file.exists():
            st.warning(f"Falha ao atualizar {isin}. Usando cache local. Detalhe: {exc}")
            return pd.read_csv(cache_file, parse_dates=["Date"]), names.get(isin)
        raise
    finally:
        if session:
            session.close()


def fetch_history_with_retries(
    session,
    xid: str,
    start_date: str,
    end_date: str,
    referer: str,
    retries: int = 2,
    backoff_seconds: float = 1.0
) -> pd.DataFrame:
    """Retry simples com backoff para instabilidades do FT."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return get_historical_data(session, xid, start_date, end_date, referer)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                import time
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Falha ao buscar dados após {retries} tentativas: {last_exc}") from None


def fetch_history_chunked(
    session,
    xid: str,
    start_date: str,
    end_date: str | None,
    referer: str,
    chunk_days: int = 1825
) -> pd.DataFrame:
    """Divide o histórico em blocos de dias para evitar erros 500 do FT."""
    if end_date is None:
        end_date = _today_str()

    start_dt = datetime.strptime(start_date, "%Y/%m/%d").date()
    end_dt = datetime.strptime(end_date, "%Y/%m/%d").date()

    all_parts: list[pd.DataFrame] = []
    current_end = end_dt

    while current_end >= start_dt:
        current_start = max(start_dt, current_end - pd.Timedelta(days=chunk_days))
        start_str = current_start.strftime("%Y/%m/%d")
        end_str = current_end.strftime("%Y/%m/%d")

        df_part = fetch_history_with_retries(session, xid, start_str, end_str, referer)
        if not df_part.empty:
            all_parts.append(df_part)

        # anda para trás um dia antes do início atual
        current_end = current_start - pd.Timedelta(days=1)

    if not all_parts:
        raise RuntimeError("Nao foi possivel obter historico por blocos.")

    df = pd.concat(all_parts, ignore_index=True)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
    return df


def build_price_matrix(
    isins: list[str],
    start_date: date,
    max_age_days: int,
    ft_hints: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Monta matriz de preços (colunas por fundo) e mapa ISIN->nome."""
    price_series = []
    names = {}
    for isin in isins:
        try:
            hint = (ft_hints or {}).get(isin)
            df, fund_name = load_or_fetch_history(
                isin,
                start_date=start_date.strftime("%Y/%m/%d"),
                max_age_days=max_age_days,
                ft_hint=hint,
            )
            if fund_name:
                names[isin] = fund_name
            df = df[["Date", "Close"]].dropna()
            df = df[df["Date"] >= pd.Timestamp(start_date)]
            df = df.set_index("Date").sort_index()
            df = df.rename(columns={"Close": isin})
            price_series.append(df)
        except Exception as exc:
            st.warning(f"Falha ao buscar dados para {isin}. Detalhe: {exc}")
            continue

    if not price_series:
        return pd.DataFrame(), {}

    prices = pd.concat(price_series, axis=1, join="inner").dropna(how="any")
    return prices, names


def _filter_failed_isins(
    isins: list[str],
    weights: np.ndarray,
    classes: list[str],
    prices: pd.DataFrame
) -> tuple[list[str], np.ndarray, list[str]]:
    """Remove ISINs sem dados e ajusta pesos/classes."""
    if prices.empty:
        return isins, weights, classes

    available = set(prices.columns)
    kept_isins = []
    kept_weights = []
    kept_classes = []
    dropped = []
    for isin, weight, cls in zip(isins, weights.tolist(), classes):
        if isin in available:
            kept_isins.append(isin)
            kept_weights.append(weight)
            kept_classes.append(cls)
        else:
            dropped.append(isin)

    if dropped:
        st.warning("ISINs removidos por falta de dados: " + ", ".join(dropped))

    return kept_isins, np.array(kept_weights), kept_classes


def _local_load_portfolios() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            with PORTFOLIO_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _local_save_portfolios(portfolios: dict):
    with PORTFOLIO_FILE.open("w", encoding="utf-8") as f:
        json.dump(portfolios, f, ensure_ascii=False, indent=2)


def _github_persistence_config() -> dict:
    """Configuração opcional para persistir carteiras no GitHub (Streamlit Cloud)."""
    try:
        section = st.secrets.get("github_persistence", {})
    except Exception:
        section = {}
    if not isinstance(section, dict):
        section = {}

    def _secret_pick(section_key: str, root_key: str, default: str = "") -> str:
        sec_val = section.get(section_key, "")
        if sec_val not in (None, ""):
            return str(sec_val).strip()
        try:
            root_val = st.secrets.get(root_key, "")
        except Exception:
            root_val = ""
        if root_val in (None, ""):
            return default
        return str(root_val).strip()

    repo = _secret_pick("repo", "GITHUB_PERSISTENCE_REPO", "")
    token = _secret_pick("token", "GITHUB_PERSISTENCE_TOKEN", "")
    branch = _secret_pick("branch", "GITHUB_PERSISTENCE_BRANCH", "main") or "main"
    file_path = _secret_pick("file_path", "GITHUB_PERSISTENCE_FILE", "carteiras.json") or "carteiras.json"
    commit_message = _secret_pick(
        "commit_message",
        "GITHUB_PERSISTENCE_COMMIT_MESSAGE",
        "Atualiza carteiras salvas",
    )
    enabled_raw = _secret_pick("enabled", "GITHUB_PERSISTENCE_ENABLED", "")
    enabled = str(enabled_raw).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        enabled = bool(repo and token)

    return {
        "enabled": enabled,
        "repo": repo,
        "token": token,
        "branch": branch,
        "file_path": file_path,
        "commit_message": commit_message,
    }


def _github_api_headers(token: str) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _load_portfolios_from_github(cfg: dict) -> dict | None:
    """Retorna dicionário de carteiras do GitHub ou None em falha."""
    if not cfg.get("enabled"):
        return None

    import requests

    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['file_path']}"
    try:
        resp = requests.get(
            url,
            headers=_github_api_headers(cfg["token"]),
            params={"ref": cfg["branch"]},
            timeout=20,
        )
        if resp.status_code == 404:
            st.session_state["portfolio_persistence_last_error"] = ""
            return {}
        resp.raise_for_status()
        payload = resp.json()
        content_b64 = str(payload.get("content", "")).strip()
        if not content_b64:
            st.session_state["portfolio_persistence_last_error"] = ""
            return {}
        content_txt = base64.b64decode(content_b64).decode("utf-8")
        data = json.loads(content_txt) if content_txt.strip() else {}
        if not isinstance(data, dict):
            raise ValueError("Formato inválido de carteiras no GitHub (esperado objeto JSON).")
        st.session_state["portfolio_persistence_last_error"] = ""
        return data
    except Exception as exc:
        st.session_state["portfolio_persistence_last_error"] = str(exc)
        return None


def _save_portfolios_to_github(cfg: dict, portfolios: dict) -> bool:
    """Persiste carteiras no GitHub; retorna True quando gravou com sucesso."""
    if not cfg.get("enabled"):
        return False

    import requests

    url = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['file_path']}"
    headers = _github_api_headers(cfg["token"])
    sha = None

    try:
        get_resp = requests.get(
            url,
            headers=headers,
            params={"ref": cfg["branch"]},
            timeout=20,
        )
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")
        elif get_resp.status_code != 404:
            get_resp.raise_for_status()

        content_txt = json.dumps(portfolios, ensure_ascii=False, indent=2)
        body = {
            "message": cfg["commit_message"],
            "content": base64.b64encode(content_txt.encode("utf-8")).decode("utf-8"),
            "branch": cfg["branch"],
        }
        if sha:
            body["sha"] = sha

        put_resp = requests.put(url, headers=headers, json=body, timeout=25)
        put_resp.raise_for_status()
        st.session_state["portfolio_persistence_last_error"] = ""
        return True
    except Exception as exc:
        st.session_state["portfolio_persistence_last_error"] = str(exc)
        return False


def load_saved_portfolios() -> dict:
    """Carrega carteiras salvas (GitHub quando configurado; fallback local)."""
    cfg = _github_persistence_config()
    if cfg.get("enabled"):
        remote = _load_portfolios_from_github(cfg)
        if isinstance(remote, dict):
            _local_save_portfolios(remote)
            st.session_state["portfolio_persistence_backend"] = "github"
            return remote
        st.session_state["portfolio_persistence_backend"] = "local_fallback"
        return _local_load_portfolios()

    st.session_state["portfolio_persistence_backend"] = "local"
    return _local_load_portfolios()


def save_all_portfolios(portfolios: dict):
    """Persiste o dicionário completo de carteiras."""
    safe_portfolios = portfolios if isinstance(portfolios, dict) else {}
    cfg = _github_persistence_config()
    saved_remote = False
    if cfg.get("enabled"):
        saved_remote = _save_portfolios_to_github(cfg, safe_portfolios)

    _local_save_portfolios(safe_portfolios)

    if cfg.get("enabled"):
        st.session_state["portfolio_persistence_backend"] = "github" if saved_remote else "local_fallback"
    else:
        st.session_state["portfolio_persistence_backend"] = "local"


def save_portfolio(name: str, data: dict):
    """Salva ou atualiza carteira no arquivo local."""
    portfolios = load_saved_portfolios()
    portfolios[name] = data
    save_all_portfolios(portfolios)


def delete_portfolio(name: str):
    """Remove carteira pelo nome quando existir."""
    portfolios = load_saved_portfolios()
    if name in portfolios:
        portfolios.pop(name, None)
        save_all_portfolios(portfolios)


def _plot_and_save(fig, path: Path):
    """Salva figura do matplotlib como imagem."""
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clf()


def _short_label(text: str, max_len: int = 22) -> str:
    """Reduz nomes longos para caber nos gráficos."""
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _unique_labels(labels: list[str]) -> list[str]:
    """Evita labels duplicados adicionando sufixo (2), (3)..."""
    seen = {}
    result = []
    for label in labels:
        if label not in seen:
            seen[label] = 1
            result.append(label)
        else:
            seen[label] += 1
            result.append(f"{label} ({seen[label]})")
    return result


def _table_image(
    df: pd.DataFrame,
    path: Path,
    title: str,
    font_size: int = 8,
    header_color: str = "#1f4e79",
    row_alt_color: str = "#f3f6fa",
    figsize: tuple[float, float] | None = None,
    show_index: bool = True,
    wrap_chars: int | None = None,
    title_pad: int = 18,
    top_margin: float = 0.86
):
    """Renderiza DataFrame como imagem de tabela para o PDF."""
    import matplotlib.pyplot as plt
    import textwrap

    df_display = df.copy()
    if wrap_chars:
        df_display = df_display.applymap(lambda x: textwrap.fill(str(x), wrap_chars))

    row_line_counts = []
    for row in df_display.values:
        line_count = 1
        for cell in row:
            line_count = max(line_count, str(cell).count("\n") + 1)
        row_line_counts.append(line_count)
    max_lines = max(row_line_counts) if row_line_counts else 1

    if figsize is None:
        rows = max(3, len(df_display) + 1)  # +1 header
        height = 0.6 + 0.35 * rows + 0.25 * (max_lines - 1)
        figsize = (10, height)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", color=header_color, pad=title_pad)
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        rowLabels=df_display.index if show_index else None,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.2 + 0.35 * (max_lines - 1))
    # Ajusta altura das linhas com base no conteúdo
    for i, line_count in enumerate(row_line_counts, start=1):
        for j in range(len(df_display.columns)):
            cell = table[(i, j)]
            cell.set_height(cell.get_height() * (1 + 0.35 * (line_count - 1)))
    fig.subplots_adjust(top=top_margin)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(row_alt_color)
    _plot_and_save(fig, path)


def _plot_interactive_lines(
    df: pd.DataFrame,
    title: str,
    yaxis_title: str,
    color_map: dict[str, str] | None = None,
    key: str | None = None,
):
    """Renderiza linhas interativas com legenda clicável (toggle por fundo)."""
    if go is None:
        st.line_chart(df)
        return

    light = _is_light_theme_active()
    text_color = "#0f172a" if light else "#e5e7eb"
    grid_color = "rgba(100,116,139,0.26)" if light else "rgba(148,163,184,0.18)"
    paper_color = "#ffffff" if light else "#0b1220"
    fig = go.Figure()
    for col in df.columns:
        line_color = (color_map or {}).get(str(col))
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=str(col),
                line={"width": 2.2, "color": line_color} if line_color else {"width": 2.2},
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.0, "y": 0.99, "xanchor": "left", "yanchor": "top"},
        template="plotly_white" if light else "plotly_dark",
        height=430,
        margin={"l": 20, "r": 20, "t": 95, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "left",
            "x": 0,
            "font": {"color": text_color, "size": 12},
        },
        hovermode="x unified",
        yaxis_title=yaxis_title,
        paper_bgcolor=paper_color,
        plot_bgcolor=paper_color,
        font={"color": text_color, "size": 13},
        title_font={"color": text_color, "size": 22},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        tickfont={"color": text_color, "size": 12},
        title_font={"color": text_color, "size": 12},
        color=text_color,
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        tickfont={"color": text_color, "size": 12},
        title_font={"color": text_color, "size": 12},
        color=text_color,
        zeroline=False,
    )
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def _plot_corr_heatmap(corr: pd.DataFrame, title: str, key: str | None = None):
    """Heatmap de correlação em escala divergente (-1 a +1)."""
    if go is None:
        st.dataframe(corr, use_container_width=True)
        return

    light = _is_light_theme_active()
    text_color = "#0f172a" if light else "#e5e7eb"
    paper_color = "#ffffff" if light else "#0b1220"
    heat = go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        zmin=-1,
        zmax=1,
        colorscale="RdBu_r",
        colorbar={
            "title": {"text": "Corr", "font": {"color": text_color}},
            "tickfont": {"color": text_color},
        },
    )
    fig = go.Figure(data=[heat])
    fig.update_layout(
        title=title,
        template="plotly_white" if light else "plotly_dark",
        height=500,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis={"tickangle": -35, "tickfont": {"color": text_color, "size": 12}},
        yaxis={"tickfont": {"color": text_color, "size": 12}},
        paper_bgcolor=paper_color,
        plot_bgcolor=paper_color,
        font={"color": text_color, "size": 13},
        title_font={"color": text_color, "size": 22},
    )
    # Rótulo por célula com contraste dinâmico:
    # células próximas de zero (claras) usam texto escuro;
    # extremos (vermelho/azul fortes) usam texto claro.
    for i, row_name in enumerate(corr.index.tolist()):
        for j, col_name in enumerate(corr.columns.tolist()):
            value = float(corr.iloc[i, j])
            label_color = "#0f172a" if abs(value) < 0.50 else "#f8fafc"
            fig.add_annotation(
                x=col_name,
                y=row_name,
                text=f"{value:.2f}",
                showarrow=False,
                font={"color": label_color, "size": 11},
            )
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def _show_dataframe_themed(df: pd.DataFrame, use_container_width: bool = True):
    """Renderiza tabela simples; o tema é controlado via CSS global."""
    st.table(df)


def _render_ft_links_table(links_df: pd.DataFrame):
    """Renderiza tabela de links FT com HTML para respeitar tema claro/escuro."""
    if links_df is None or links_df.empty:
        st.info("Sem links para exibir.")
        return

    rows_html: list[str] = []
    for _, row in links_df.iterrows():
        fundo = html.escape(str(row.get("Fundo", "")))
        isin = html.escape(str(row.get("ISIN", "")))
        summary = html.escape(str(row.get("FT Summary", "")))
        risk = html.escape(str(row.get("FT Risk", "")))
        rows_html.append(
            (
                "<tr>"
                f"<td>{fundo}</td>"
                f"<td>{isin}</td>"
                f"<td><a href=\"{summary}\" target=\"_blank\" rel=\"noopener noreferrer\">{summary}</a></td>"
                f"<td><a href=\"{risk}\" target=\"_blank\" rel=\"noopener noreferrer\">{risk}</a></td>"
                "</tr>"
            )
        )

    st.markdown(
        (
            "<div class=\"ft-links-wrap\">"
            "<table class=\"ft-links-table\">"
            "<thead><tr><th>Fundo</th><th>ISIN</th><th>FT Summary</th><th>FT Risk</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_assets_grid_table(funds_df: pd.DataFrame, option_to_isin: dict[str, str], isin_to_roa: dict[str, float]):
    """Tabela visual estável para Ativo/Alocação/Categoria/ROA."""
    if funds_df is None or funds_df.empty:
        return

    rows_html: list[str] = []
    for _, row in funds_df.iterrows():
        ativo = str(row.get("Ativo", "")).strip()
        aloc_raw = pd.to_numeric(row.get("Alocacao_%", 0.0), errors="coerce")
        aloc = 0.0 if pd.isna(aloc_raw) else float(aloc_raw)
        classe = str(row.get("Classe", "")).strip()
        roa = _resolve_option_roa(ativo, option_to_isin, isin_to_roa)
        roa_txt = _fmt_optional_number(roa, decimals=2)
        rows_html.append(
            (
                "<tr>"
                f"<td>{html.escape(ativo)}</td>"
                f"<td>{aloc:.2f}</td>"
                f"<td>{html.escape(classe)}</td>"
                f"<td>{roa_txt}</td>"
                "</tr>"
            )
        )

    st.markdown(
        (
            "<div class=\"assets-grid-wrap\">"
            "<table class=\"assets-grid-table\">"
            "<thead><tr><th>Ativo</th><th>Alocação (%)</th><th>Categoria</th><th>ROA (%)</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _collect_fund_colors(labels: list[str], key_prefix: str) -> dict[str, str]:
    """Permite ao usuário customizar cores por fundo."""
    colors: dict[str, str] = {}
    with st.expander("Cores", expanded=False):
        st.caption("Dica: clique na legenda do gráfico para ocultar/exibir fundos.")
        for idx, label in enumerate(labels):
            default = DEFAULT_CHART_COLORS[idx % len(DEFAULT_CHART_COLORS)]
            safe = re.sub(r"[^a-zA-Z0-9]+", "_", str(label)).strip("_").lower() or f"fund_{idx}"
            key = f"{key_prefix}_color_{safe}_{idx}"
            colors[str(label)] = st.color_picker(str(label), value=default, key=key)
    return colors


def compute_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial_value: float,
    rebalance_monthly: bool
):
    """Simula carteira mensal (rebalanceada ou buy&hold)."""
    # Trabalhamos apenas com preços mensais (sem diário).
    monthly_prices = prices.resample("M").last().dropna(how="any")
    monthly_returns_assets = monthly_prices.pct_change().dropna(how="any")

    if rebalance_monthly:
        # Rebalanceia todo fim de mês
        portfolio_monthly_returns = monthly_returns_assets.dot(weights)
        portfolio_value = (1 + portfolio_monthly_returns).cumprod() * initial_value
    else:
        # Buy & hold: fixa número de cotas no início
        first_prices = monthly_prices.iloc[0]
        shares = (initial_value * weights) / first_prices
        portfolio_value = (monthly_prices * shares).sum(axis=1)
        portfolio_monthly_returns = portfolio_value.pct_change().dropna()

    rolling_max = portfolio_value.cummax()
    drawdown = portfolio_value / rolling_max - 1

    years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1
    vol_annual = portfolio_monthly_returns.std() * np.sqrt(12)

    return {
        "portfolio_value": portfolio_value,
        "monthly_returns": portfolio_monthly_returns,
        "drawdown": drawdown,
        "cagr": cagr,
        "vol_annual": vol_annual,
    }


def annual_returns(portfolio_value: pd.Series) -> pd.Series:
    """Retornos ano a ano."""
    yearly = portfolio_value.resample("Y").last()
    yearly_ret = yearly.pct_change().dropna()
    yearly_ret.index = yearly_ret.index.year
    return yearly_ret


def monthly_table(portfolio_value: pd.Series) -> pd.DataFrame:
    """Tabela Year x Month de retornos mensais."""
    monthly = portfolio_value.resample("M").last()
    monthly_ret = monthly.pct_change()
    table = monthly_ret.to_frame(name="ret")
    table["Year"] = table.index.year
    table["Month"] = table.index.strftime("%b")
    pivot = table.pivot(index="Year", columns="Month", values="ret")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=month_order)
    return pivot


def trailing_returns(portfolio_value: pd.Series) -> pd.DataFrame:
    """Retornos acumulados/anualizados para janelas (1M, 3M, 1Y...)."""
    last_value = portfolio_value.iloc[-1]
    last_date = portfolio_value.index[-1]
    horizons = {
        "1M": 1,
        "3M": 3,
        "6M": 6,
        "1Y": 12,
        "3Y": 36,
        "5Y": 60,
        "10Y": 120,
    }

    results = {}
    monthly = portfolio_value.resample("M").last()
    for label, months in horizons.items():
        if len(monthly) <= months:
            results[label] = np.nan
            continue
        start_value = monthly.iloc[-(months + 1)]
        if "Y" in label:
            years = months / 12
            results[label] = (last_value / start_value) ** (1 / years) - 1
        else:
            results[label] = (last_value / start_value) - 1

    results["Total"] = (last_value / monthly.iloc[0]) - 1
    results_df = pd.DataFrame(results, index=["Retorno"])
    return results_df


# -------- UI (Streamlit) --------
st.set_page_config(
    page_title="Simulador de Fundos Offshore",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "ui_theme_option" not in st.session_state:
    st.session_state.ui_theme_option = "Escuro"
if st.session_state.get("ui_theme_option") not in {"Escuro", "Claro"}:
    st.session_state.ui_theme_option = "Escuro"

theme_spacer_col, theme_control_col = st.columns([9.8, 2.2])
with theme_control_col:
    st.caption("Tema")
    st.radio(
        "Tema",
        ["Escuro", "Claro"],
        horizontal=True,
        key="ui_theme_option",
        label_visibility="collapsed",
    )

_light_theme = _is_light_theme_active()
try:
    plt.style.use("default" if _light_theme else "dark_background")
except Exception:
    pass

if _light_theme:
    _app_grad_a = "rgba(37,99,235,0.10)"
    _app_grad_b = "rgba(16,185,129,0.08)"
    _app_base_bg = "#f7fbff"
    _hero_border = "rgba(18,36,58,0.14)"
    _hero_bg = "linear-gradient(135deg, rgba(240,247,255,0.92), rgba(236,255,248,0.92))"
    _hero_sub = "rgba(22,31,45,0.74)"
    _hero_title = "#16345c"
    _theme_overrides = """
[data-testid="stAppViewContainer"] {
  background-color: #f7fbff !important;
}
html, body, [data-testid="stAppViewContainer"] {
  color-scheme: light !important;
}
:root {
  --gdg-bg-cell: #f8fafc !important;
  --gdg-bg-header: #e2e8f0 !important;
  --gdg-border-color: #94a3b8 !important;
  --gdg-text-dark: #0f172a !important;
  --gdg-text-medium: #334155 !important;
  --gdg-text-light: #475569 !important;
  --gdg-text-group-header: #0f172a !important;
  --gdg-accent-color: #e11d48 !important;
  --gdg-accent-fg: #ffffff !important;
  --gdg-accent-light: #fde8ef !important;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] div[data-testid="stMarkdownContainer"] {
  color: #1f2937 !important;
}
[data-testid="stHeader"] {
  background: rgba(248,250,252,0.92) !important;
}
[data-testid="stSidebar"] {
  background-color: #e9edf3 !important;
}
section[data-testid="stSidebar"] * {
  color: #1f2937 !important;
}
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] div[role="radiogroup"] label,
[data-testid="stRadio"] div[role="radiogroup"] label *,
[data-testid="stCaptionContainer"] p {
  color: #1f2937 !important;
  opacity: 1 !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] input {
  accent-color: #e11d48 !important;
}
[data-baseweb="segmented-control"] {
  background-color: #f8fafc !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
}
[data-baseweb="segmented-control-group"] {
  background-color: #f8fafc !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
}
[data-baseweb="segmented-control"] button,
[data-baseweb="segmented-control-group"] button {
  color: #334155 !important;
  background: #f8fafc !important;
  border: none !important;
}
[data-baseweb="segmented-control"] button[aria-pressed="true"],
[data-baseweb="segmented-control-group"] button[aria-pressed="true"] {
  color: #ffffff !important;
  background: #e11d48 !important;
}
[data-baseweb="segmented-control"] button:not([aria-pressed="true"]):hover,
[data-baseweb="segmented-control-group"] button:not([aria-pressed="true"]):hover {
  background: #eef2f7 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
.stDateInput input,
.stTextArea textarea {
  background-color: #ffffff !important;
  color: #111827 !important;
  border-color: #cbd5e1 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: #111827 !important;
  border-color: #cbd5e1 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
  color: #111827 !important;
}
[data-testid="stSelectbox"] svg {
  fill: #64748b !important;
}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
  background-color: #ffffff !important;
  border: 1px solid #b7c5d7 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
[data-testid="stTextInput"] [data-baseweb="input"] > div,
[data-testid="stDateInput"] [data-baseweb="input"] > div,
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stDateInput"] [data-baseweb="base-input"] {
  background-color: #ffffff !important;
  border-color: #cbd5e1 !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within > div,
[data-testid="stDateInput"] [data-baseweb="input"]:focus-within > div,
[data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within,
[data-testid="stDateInput"] [data-baseweb="base-input"]:focus-within {
  border-color: #94a3b8 !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input {
  color: #111827 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stDateInput"] input:focus {
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus-visible,
[data-testid="stDateInput"] input:focus-visible {
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stDateInput"] button,
[data-testid="stDateInput"] [data-baseweb="button"] {
  background-color: #ffffff !important;
  color: #334155 !important;
  border-left: 1px solid #cbd5e1 !important;
}
[data-testid="stNumberInputContainer"] {
  background-color: #ffffff !important;
  border-color: #cbd5e1 !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
[data-testid="stNumberInputContainer"].focused {
  border-color: #94a3b8 !important;
  box-shadow: none !important;
}
[data-testid="stNumberInputField"] {
  background-color: #ffffff !important;
  color: #111827 !important;
}
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"] {
  background-color: #f8fafc !important;
  color: #1f2937 !important;
  border-left: 1px solid #cbd5e1 !important;
}
[data-testid="stNumberInputStepDown"]:hover,
[data-testid="stNumberInputStepUp"]:hover {
  background-color: #f1f5f9 !important;
}
.stButton > button {
  background-color: #ffffff !important;
  color: #1f2937 !important;
  border: 1px solid #cbd5e1 !important;
}
.stButton > button:hover {
  background-color: #f3f4f6 !important;
  color: #111827 !important;
}
iframe[title*="searchbox"] {
  border: 0 !important;
  outline: 0 !important;
  background: #ffffff !important;
  box-shadow: none !important;
}
[data-baseweb="popover"] {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
}
[data-baseweb="menu"] {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
}
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li,
[data-baseweb="menu"] div {
  background: #ffffff !important;
  color: #111827 !important;
}
[data-baseweb="popover"] *,
[role="listbox"] *,
li[role="option"] * {
  color: #111827 !important;
}
[role="listbox"] {
  background: #ffffff !important;
}
li[role="option"] {
  background: #ffffff !important;
}
li[role="option"][aria-selected="true"] {
  background: #eef2ff !important;
}
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  overflow: hidden !important;
  --gdg-bg-cell: #ffffff !important;
  --gdg-bg-header: #e2e8f0 !important;
  --gdg-border-color: #cbd5e1 !important;
  --gdg-text-dark: #0f172a !important;
  --gdg-text-medium: #1f2937 !important;
  --gdg-text-light: #334155 !important;
  --gdg-text-group-header: #0f172a !important;
}
[data-testid="stDataFrame"] [role="grid"],
[data-testid="stDataEditor"] [role="grid"],
[data-testid="stDataFrame"] [role="grid"] *,
[data-testid="stDataEditor"] [role="grid"] * {
  color: #0f172a !important;
}
[data-testid="stTable"] table {
  width: 100% !important;
  border-collapse: collapse !important;
  background: #ffffff !important;
  color: #0f172a !important;
}
[data-testid="stTable"] th {
  background: #e2e8f0 !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
}
[data-testid="stTable"] td {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
}
[data-testid="stDataFrameGlideDataEditor"],
[data-testid="stDataEditor"] [data-testid="stDataFrameGlideDataEditor"],
[data-testid="stDataFrame"] [data-testid="stDataFrameGlideDataEditor"],
.stDataFrameGlideDataEditor {
  background: #ffffff !important;
  color-scheme: light !important;
  --gdg-bg-cell: #ffffff !important;
  --gdg-bg-cell-medium: #f8fafc !important;
  --gdg-bg-header: #e2e8f0 !important;
  --gdg-bg-header-hovered: #dbeafe !important;
  --gdg-border-color: #cbd5e1 !important;
  --gdg-text-dark: #0f172a !important;
  --gdg-text-medium: #1f2937 !important;
  --gdg-text-light: #334155 !important;
  --gdg-text-group-header: #0f172a !important;
  --gdg-text-header: #0f172a !important;
  --gdg-link-color: #1d4ed8 !important;
  --gdg-fg-icon-header: #475569 !important;
  --gdg-bg-icon-header: #ffffff !important;
}
[data-testid="stDataFrameGlideDataEditor"] [role="grid"],
[data-testid="stDataFrameGlideDataEditor"] [role="grid"] * {
  color: #0f172a !important;
}
[data-testid="stDataFrameGlideDataEditor"] a {
  color: #1d4ed8 !important;
}
.assets-grid-wrap {
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  overflow: auto !important;
  background: #ffffff !important;
  margin-top: 0.45rem !important;
}
.assets-grid-table {
  width: 100% !important;
  border-collapse: collapse !important;
  font-size: 0.95rem !important;
  color: #0f172a !important;
}
.assets-grid-table thead th {
  background: #e2e8f0 !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  text-align: left !important;
  padding: 0.48rem 0.58rem !important;
}
.assets-grid-table tbody td {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  padding: 0.40rem 0.58rem !important;
}
.ft-links-wrap {
  border: 1px solid #cbd5e1 !important;
  border-radius: 10px !important;
  overflow: auto !important;
  background: #ffffff !important;
}
.ft-links-table {
  width: 100% !important;
  border-collapse: collapse !important;
  font-size: 0.92rem !important;
  color: #0f172a !important;
}
.ft-links-table thead th {
  background: #e2e8f0 !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  text-align: left !important;
  padding: 0.45rem 0.55rem !important;
}
.ft-links-table tbody td {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  padding: 0.38rem 0.55rem !important;
  vertical-align: top !important;
}
.ft-links-table a {
  color: #1d4ed8 !important;
  text-decoration: none !important;
}
.ft-links-table a:hover {
  text-decoration: underline !important;
}
.asset-inline-name {
  color: #0f172a !important;
  font-weight: 600 !important;
  background: transparent !important;
}
.alloc-total {
  margin: 0.35rem 0 0.15rem 0 !important;
  font-size: 0.92rem !important;
  font-weight: 600 !important;
}
.alloc-total.ok { color: #15803d !important; }
.alloc-total.low { color: #b45309 !important; }
.alloc-total.high { color: #b91c1c !important; }
"""
else:
    _app_grad_a = "rgba(22,108,255,0.14)"
    _app_grad_b = "rgba(0,196,140,0.10)"
    _app_base_bg = "#0a1018"
    _hero_border = "rgba(255,255,255,0.08)"
    _hero_bg = "linear-gradient(135deg, rgba(11,26,44,0.86), rgba(9,17,28,0.86))"
    _hero_sub = "rgba(230,240,255,0.82)"
    _hero_title = "#eaf1ff"
    _theme_overrides = """
[data-testid="stAppViewContainer"] {
  background-color: #0a1018 !important;
}
html, body, [data-testid="stAppViewContainer"] {
  color-scheme: dark !important;
}
:root {
  --gdg-bg-cell: #0b1220 !important;
  --gdg-bg-header: #111827 !important;
  --gdg-border-color: #334155 !important;
  --gdg-text-dark: #e5e7eb !important;
  --gdg-text-medium: #cbd5e1 !important;
  --gdg-text-light: #94a3b8 !important;
  --gdg-text-group-header: #e5e7eb !important;
  --gdg-accent-color: #e11d48 !important;
  --gdg-accent-fg: #f8fafc !important;
  --gdg-accent-light: #3f1d2b !important;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] div[data-testid="stMarkdownContainer"] {
  color: #e5e7eb !important;
}
[data-testid="stSidebar"] {
  background-color: #0f1724 !important;
}
section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
}
[data-testid="stHeader"] {
  background: rgba(10,16,24,0.96) !important;
}
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] div[role="radiogroup"] label,
[data-testid="stRadio"] div[role="radiogroup"] label *,
[data-testid="stCaptionContainer"] p {
  color: #e5e7eb !important;
  opacity: 1 !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] input {
  accent-color: #e11d48 !important;
}
div[data-baseweb="segmented-control"] *,
div[data-baseweb="segmented-control-group"] *,
[data-testid="stSegmentedControl"] * {
  color: #e5e7eb !important;
}
[data-baseweb="segmented-control"] {
  background-color: #0f172a !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
}
[data-baseweb="segmented-control-group"] {
  background-color: #0f172a !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
}
[data-baseweb="segmented-control"] button,
[data-baseweb="segmented-control-group"] button {
  color: #cbd5e1 !important;
  background: #0f172a !important;
  border: none !important;
}
[data-baseweb="segmented-control"] button[aria-pressed="true"],
[data-baseweb="segmented-control-group"] button[aria-pressed="true"] {
  color: #f8fafc !important;
  background: #e11d48 !important;
}
[data-baseweb="segmented-control"] button:not([aria-pressed="true"]):hover,
[data-baseweb="segmented-control-group"] button:not([aria-pressed="true"]):hover {
  background: #1f2937 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
.stDateInput input,
.stTextArea textarea {
  background-color: #111827 !important;
  color: #e5e7eb !important;
  border-color: #374151 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background-color: #111827 !important;
  color: #e5e7eb !important;
  border-color: #374151 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {
  color: #e5e7eb !important;
}
[data-testid="stSelectbox"] svg {
  fill: #94a3b8 !important;
}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
  background-color: #111827 !important;
  border: 1px solid #475569 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
[data-testid="stTextInput"] [data-baseweb="input"] > div,
[data-testid="stDateInput"] [data-baseweb="input"] > div,
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stDateInput"] [data-baseweb="base-input"] {
  background-color: #111827 !important;
  border-color: #374151 !important;
  border: 1px solid #475569 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within > div,
[data-testid="stDateInput"] [data-baseweb="input"]:focus-within > div,
[data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within,
[data-testid="stDateInput"] [data-baseweb="base-input"]:focus-within {
  border-color: #64748b !important;
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input {
  color: #e5e7eb !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stDateInput"] input:focus {
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus-visible,
[data-testid="stDateInput"] input:focus-visible {
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stDateInput"] button,
[data-testid="stDateInput"] [data-baseweb="button"] {
  background-color: #111827 !important;
  color: #e5e7eb !important;
  border-left: 1px solid #475569 !important;
}
[data-testid="stNumberInputContainer"] {
  background-color: #111827 !important;
  border-color: #374151 !important;
  border: 1px solid #475569 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
[data-testid="stNumberInputContainer"].focused {
  border-color: #64748b !important;
  box-shadow: none !important;
}
[data-testid="stNumberInputField"] {
  background-color: #111827 !important;
  color: #e5e7eb !important;
}
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"] {
  background-color: #111827 !important;
  color: #e5e7eb !important;
  border-left: 1px solid #334155 !important;
}
[data-testid="stNumberInputStepDown"]:hover,
[data-testid="stNumberInputStepUp"]:hover {
  background-color: #1f2937 !important;
}
[data-baseweb="popover"] * {
  color: #e5e7eb !important;
}
.stButton > button {
  background-color: #111827 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
}
.stButton > button:hover {
  background-color: #1f2937 !important;
  color: #f9fafb !important;
}
.stButton > button:disabled {
  background-color: #1f2937 !important;
  color: #94a3b8 !important;
  border: 1px solid #334155 !important;
  opacity: 1 !important;
}
iframe[title*="searchbox"] {
  border: 0 !important;
  outline: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
[data-baseweb="popover"] {
  background: #111827 !important;
  border: 1px solid #334155 !important;
}
[data-baseweb="menu"] {
  background: #111827 !important;
  border: 1px solid #334155 !important;
}
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li,
[data-baseweb="menu"] div {
  background: #111827 !important;
  color: #e5e7eb !important;
}
[data-baseweb="popover"] *,
[role="listbox"] *,
li[role="option"] * {
  color: #e5e7eb !important;
}
[role="listbox"] {
  background: #111827 !important;
}
li[role="option"] {
  background: #111827 !important;
}
li[role="option"][aria-selected="true"] {
  background: #1f2937 !important;
}
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
  background: #0b1220 !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
  overflow: hidden !important;
  --gdg-bg-cell: #0b1220 !important;
  --gdg-bg-header: #111827 !important;
  --gdg-border-color: #334155 !important;
  --gdg-text-dark: #e5e7eb !important;
  --gdg-text-medium: #cbd5e1 !important;
  --gdg-text-light: #94a3b8 !important;
  --gdg-text-group-header: #e5e7eb !important;
}
[data-testid="stDataFrame"] [role="grid"],
[data-testid="stDataEditor"] [role="grid"],
[data-testid="stDataFrame"] [role="grid"] *,
[data-testid="stDataEditor"] [role="grid"] * {
  color: #e5e7eb !important;
}
[data-testid="stTable"] table {
  width: 100% !important;
  border-collapse: collapse !important;
  background: #0b1220 !important;
  color: #e5e7eb !important;
}
[data-testid="stTable"] th {
  background: #111827 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
}
[data-testid="stTable"] td {
  background: #0b1220 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
}
[data-testid="stDataFrameGlideDataEditor"],
[data-testid="stDataEditor"] [data-testid="stDataFrameGlideDataEditor"],
[data-testid="stDataFrame"] [data-testid="stDataFrameGlideDataEditor"],
.stDataFrameGlideDataEditor {
  background: #0b1220 !important;
  color-scheme: dark !important;
  --gdg-bg-cell: #0b1220 !important;
  --gdg-bg-cell-medium: #0f172a !important;
  --gdg-bg-header: #111827 !important;
  --gdg-bg-header-hovered: #1f2937 !important;
  --gdg-border-color: #334155 !important;
  --gdg-text-dark: #e5e7eb !important;
  --gdg-text-medium: #cbd5e1 !important;
  --gdg-text-light: #94a3b8 !important;
  --gdg-text-group-header: #e5e7eb !important;
  --gdg-text-header: #e5e7eb !important;
  --gdg-link-color: #60a5fa !important;
  --gdg-fg-icon-header: #cbd5e1 !important;
  --gdg-bg-icon-header: #0f172a !important;
}
[data-testid="stDataFrameGlideDataEditor"] [role="grid"],
[data-testid="stDataFrameGlideDataEditor"] [role="grid"] * {
  color: #e5e7eb !important;
}
[data-testid="stDataFrameGlideDataEditor"] a {
  color: #60a5fa !important;
}
.assets-grid-wrap {
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
  overflow: auto !important;
  background: #0b1220 !important;
  margin-top: 0.45rem !important;
}
.assets-grid-table {
  width: 100% !important;
  border-collapse: collapse !important;
  font-size: 0.95rem !important;
  color: #e5e7eb !important;
}
.assets-grid-table thead th {
  background: #111827 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
  text-align: left !important;
  padding: 0.48rem 0.58rem !important;
}
.assets-grid-table tbody td {
  background: #0b1220 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
  padding: 0.40rem 0.58rem !important;
}
.ft-links-wrap {
  border: 1px solid #334155 !important;
  border-radius: 10px !important;
  overflow: auto !important;
  background: #0b1220 !important;
}
.ft-links-table {
  width: 100% !important;
  border-collapse: collapse !important;
  font-size: 0.92rem !important;
  color: #e5e7eb !important;
}
.ft-links-table thead th {
  background: #111827 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
  text-align: left !important;
  padding: 0.45rem 0.55rem !important;
}
.ft-links-table tbody td {
  background: #0b1220 !important;
  color: #e5e7eb !important;
  border: 1px solid #334155 !important;
  padding: 0.38rem 0.55rem !important;
  vertical-align: top !important;
}
.ft-links-table a {
  color: #60a5fa !important;
  text-decoration: none !important;
}
.ft-links-table a:hover {
  text-decoration: underline !important;
}
.asset-inline-name {
  color: #e5e7eb !important;
  font-weight: 600 !important;
  background: transparent !important;
}
.alloc-total {
  margin: 0.35rem 0 0.15rem 0 !important;
  font-size: 0.92rem !important;
  font-weight: 600 !important;
}
.alloc-total.ok { color: #4ade80 !important; }
.alloc-total.low { color: #fbbf24 !important; }
.alloc-total.high { color: #f87171 !important; }
"""

_style_template = """
<style>
.stApp {
  background:
    radial-gradient(1000px 500px at 0% 0%, __APP_GRAD_A__, transparent 60%),
    radial-gradient(1000px 600px at 100% 0%, __APP_GRAD_B__, transparent 55%),
    __APP_BASE_BG__;
}
.block-container { padding-top: 1.6rem; }
h1, h2, h3, .stMarkdown p, .stCaption, label { font-family: "Segoe UI", "Trebuchet MS", sans-serif; }
.hero-card {
  border: 1px solid __HERO_BORDER__;
  border-radius: 16px;
  padding: 1rem 1.2rem;
  background: __HERO_BG__;
  margin-bottom: 1rem;
}
.hero-title {
  margin: 0;
  font-size: 2.1rem;
  font-weight: 700;
  letter-spacing: 0.3px;
  color: __HERO_TITLE__;
}
.hero-subtitle {
  margin-top: .35rem;
  color: __HERO_SUB__;
  font-size: .98rem;
}
</style>
"""
_style = (
    _style_template
    .replace("__APP_GRAD_A__", _app_grad_a)
    .replace("__APP_GRAD_B__", _app_grad_b)
    .replace("__APP_BASE_BG__", _app_base_bg)
    .replace("__HERO_BORDER__", _hero_border)
    .replace("__HERO_BG__", _hero_bg)
    .replace("__HERO_SUB__", _hero_sub)
    .replace("__HERO_TITLE__", _hero_title)
)
_style = _style.replace("</style>", f"{_theme_overrides}\n</style>")
st.markdown(_style, unsafe_allow_html=True)
st.markdown(
    """
<div class="hero-card">
  <p class="hero-title">Carteiras Offshore</p>
  <p class="hero-subtitle">Monte carteiras e compare ativos com busca por nome ticker ou ISIN (Fundos).</p>
</div>
""",
    unsafe_allow_html=True,
)
app_mode = st.radio(
    "Escolha a ferramenta",
    ["Simulador de Carteiras", "Comparador de Ativos"],
    horizontal=True,
    key="app_mode_radio",
)

# Heartbeat para o launcher encerrar quando a aba for fechada.
try:
    from streamlit import st_autorefresh
    st_autorefresh(interval=5000, key="heartbeat")
except Exception:
    try:
        st.autorefresh(interval=5000, key="heartbeat")
    except Exception:
        pass

try:
    heartbeat_env = os.environ.get("PORTFOLIO_APP_HEARTBEAT")
    heartbeat_path = Path(heartbeat_env) if heartbeat_env else Path(tempfile.gettempdir()) / "portfolio_app.heartbeat"
    heartbeat_path.write_text(str(time.time()), encoding="utf-8")
except Exception:
    pass

with st.sidebar:
    # Inputs de simulação e carteira salva
    st.subheader("Parâmetros")
    portfolios = load_saved_portfolios()
    portfolio_names = ["(nova)"] + sorted(portfolios.keys())
    selected_portfolio = st.selectbox("Carregar carteira", portfolio_names)

    if "loaded_portfolio" not in st.session_state:
        st.session_state.loaded_portfolio = "(nova)"

    if selected_portfolio != st.session_state.loaded_portfolio and selected_portfolio != "(nova)":
        saved = portfolios[selected_portfolio]
        st.session_state.portfolio_name = saved.get("name", selected_portfolio)
        st.session_state.initial_value = saved.get("initial_value", 10000.0)
        st.session_state.start_date = datetime.strptime(saved.get("start_date"), "%Y-%m-%d").date()
        st.session_state.rebalance_monthly = saved.get("rebalance_monthly", True)
        st.session_state.loaded_portfolio = selected_portfolio

    if selected_portfolio == "(nova)" and st.session_state.loaded_portfolio != "(nova)":
        st.session_state.loaded_portfolio = "(nova)"

    portfolio_name = st.text_input("Nome da carteira", value=st.session_state.get("portfolio_name", "Carteira Padrão"))
    initial_value = st.number_input(
        "Valor inicial (aporte único)",
        min_value=1.0,
        value=float(st.session_state.get("initial_value", 10000.0)),
        step=100.0
    )
    start_date = st.date_input(
        "Data inicial da simulação",
        value=st.session_state.get("start_date", date(2015, 1, 1))
    )
    rebalance_monthly = st.checkbox(
        "Rebalanceamento mensal",
        value=bool(st.session_state.get("rebalance_monthly", True))
    )

    st.divider()
    with st.expander("Gerenciar carteiras", expanded=False):
        current_portfolios = dict(portfolios)
        storage_backend = str(st.session_state.get("portfolio_persistence_backend", "local")).strip().lower()
        cfg_persist = _github_persistence_config()
        if cfg_persist.get("enabled"):
            if storage_backend == "github":
                st.caption("Persistência: GitHub (online).")
            else:
                st.caption("Persistência: fallback local (falha no GitHub).")
                last_err = str(st.session_state.get("portfolio_persistence_last_error", "")).strip()
                if last_err:
                    st.caption(f"Erro GitHub: {last_err[:180]}")
        else:
            st.caption("Persistência: local (arquivo da sessão).")
            st.caption("Para persistir online, configure `github_persistence` em `st.secrets`.")

        saved_names = sorted(current_portfolios.keys())
        if saved_names:
            delete_target = st.selectbox(
                "Excluir carteira salva",
                saved_names,
                key="portfolio_delete_target",
            )
            if st.button("Apagar carteira", key="portfolio_delete_btn", use_container_width=True):
                delete_portfolio(delete_target)
                if st.session_state.get("loaded_portfolio") == delete_target:
                    st.session_state.loaded_portfolio = "(nova)"
                st.success(f"Carteira '{delete_target}' removida.")
                st.rerun()
        else:
            st.caption("Nenhuma carteira salva.")

        export_json = json.dumps(current_portfolios, ensure_ascii=False, indent=2)
        st.download_button(
            "Baixar carteiras (JSON)",
            data=export_json.encode("utf-8"),
            file_name="carteiras.json",
            mime="application/json",
            key="portfolio_download_btn",
            use_container_width=True,
        )

        import_mode = st.radio(
            "Importação",
            ["Mesclar", "Substituir"],
            horizontal=True,
            key="portfolio_import_mode",
        )
        upload_file = st.file_uploader(
            "Importar carteiras.json",
            type=["json"],
            key="portfolio_upload_json",
        )
        if upload_file is not None and st.button(
            "Importar carteiras",
            key="portfolio_import_btn",
            use_container_width=True,
        ):
            try:
                incoming = json.loads(upload_file.getvalue().decode("utf-8"))
                if not isinstance(incoming, dict):
                    raise ValueError("O arquivo precisa conter um objeto JSON no topo.")
                if import_mode == "Substituir":
                    merged = incoming
                else:
                    merged = dict(current_portfolios)
                    merged.update(incoming)
                save_all_portfolios(merged)
                st.success(f"{len(incoming)} carteira(s) importada(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao importar carteiras: {exc}")

max_cache_age = CACHE_MAX_AGE_DAYS

if "ft_identifier_hints" not in st.session_state:
    st.session_state.ft_identifier_hints = {}

st.subheader("Ativos e alocação")
catalog_error = None
try:
    catalog_df = load_funds_catalog(str(FUNDS_CATALOG_FILE))
except Exception as exc:
    catalog_df = pd.DataFrame(columns=["FundName", "Isin", "DistributionRoaFee", "FundOption"])
    catalog_error = str(exc)

if catalog_error:
    st.warning(f"Catálogo de fundos indisponível: {catalog_error}")

catalog_options = catalog_df["FundOption"].tolist()
option_to_isin = dict(zip(catalog_df["FundOption"], catalog_df["Isin"]))
isin_to_option = dict(zip(catalog_df["Isin"], catalog_df["FundOption"]))
isin_to_roa = dict(zip(catalog_df["Isin"], catalog_df["DistributionRoaFee"]))

if app_mode == "Comparador de Ativos":
    st.subheader("Comparador de Ativos")
    st.caption("Compare risco e retorno entre fundos, ações e ETFs com dados do Financial Times.")

    if "comparison_list" not in st.session_state:
        st.session_state.comparison_list = []

    with st.container(border=True):
        cmp_asset_filter = st.radio(
            "Buscar em",
            ["Todos", "Fundos", "ETFs", "Stocks"],
            horizontal=True,
            key="cmp_asset_filter",
        )
        picked_ft = _live_ft_searchbox(
            "Adicionar para comparação (nome, ISIN ou ticker)",
            "Explorar.",
            key_prefix="cmp_ft",
            asset_filter=cmp_asset_filter,
        )
        cmp_col1, cmp_col2 = st.columns([1, 6])
        cmp_clear = cmp_col1.button("Limpar", key="cmp_clear_btn_top", use_container_width=True)
        cmp_col2.caption("Selecione os ativos")

        if cmp_clear:
            st.session_state.comparison_list = []
            st.session_state.pop("cmp_ft_searchbox", None)
            st.session_state.pop("cmp_ft_searchbox_light", None)
            st.session_state.pop("cmp_ft_searchbox_dark", None)
            st.session_state.pop("cmp_last_selected", None)
            st.rerun()

        if picked_ft:
            symbol_value = str(picked_ft["symbol"]).strip().upper()
            parsed_value = _extract_isin_from_option(symbol_value, option_to_isin={}) or symbol_value
            st.session_state.ft_identifier_hints[symbol_value] = {
                "xid": picked_ft["xid"],
                "symbol": symbol_value,
                "asset_type": picked_ft["asset_type"],
                "name": picked_ft["name"],
            }
            st.session_state.ft_identifier_hints[parsed_value] = st.session_state.ft_identifier_hints[symbol_value]
            if parsed_value not in st.session_state.comparison_list:
                st.session_state.comparison_list.append(parsed_value)
            st.session_state.pop("cmp_ft_searchbox", None)
            st.session_state.pop("cmp_ft_searchbox_light", None)
            st.session_state.pop("cmp_ft_searchbox_dark", None)
            st.session_state.pop("cmp_last_selected", None)
            st.rerun()

    cmp_choices = st.multiselect(
        "Ativos para comparar",
        options=st.session_state.comparison_list,
        default=st.session_state.comparison_list,
        key="cmp_multiselect_top",
    )

    if st.button("Comparar ativos", key="cmp_run_btn_top"):
        cmp_df = pd.DataFrame(
            {"Ativo": cmp_choices, "Alocacao_%": [1.0] * len(cmp_choices), "Classe": [""] * len(cmp_choices)}
        )
        cmp_isins, _, _, _, _ = _parse_funds_input(cmp_df, option_to_isin, isin_to_roa)
        cmp_isins = list(dict.fromkeys(cmp_isins))

        if len(cmp_isins) < 2:
            st.warning("Selecione pelo menos 2 ativos para comparar.")
            st.stop()

        with st.spinner("Buscando dados..."):
            cmp_hints = {
                identifier: st.session_state.ft_identifier_hints.get(identifier, {})
                for identifier in cmp_isins
                if identifier in st.session_state.ft_identifier_hints
            }
            cmp_prices, cmp_names = build_price_matrix(cmp_isins, start_date, max_cache_age, ft_hints=cmp_hints)
            risk_rows = []
            for isin in cmp_isins:
                risk = fetch_ft_risk_snapshot(isin)
                risk_rows.append(
                    {
                        "ISIN": isin,
                        "Fundo": cmp_names.get(isin, isin),
                        "Benchmark FT": risk.get("benchmark"),
                        "Sharpe 1Y (FT)": risk.get("1Y_Sharpe ratio_float"),
                        "Sharpe 3Y (FT)": risk.get("3Y_Sharpe ratio_float"),
                        "Sharpe 5Y (FT)": risk.get("5Y_Sharpe ratio_float"),
                        "Vol 1Y (FT, %)": risk.get("1Y_Standard deviation_float"),
                        "Vol 3Y (FT, %)": risk.get("3Y_Standard deviation_float"),
                        "Vol 5Y (FT, %)": risk.get("5Y_Standard deviation_float"),
                        "Link FT Summary": make_ft_summary_link_from_isin(isin),
                        "Link FT Risk": make_ft_risk_link_from_isin(isin),
                    }
                )

        if cmp_prices.empty:
            st.error("Sem dados em comum para os ativos selecionados.")
            st.stop()

        cmp_monthly = cmp_prices.resample("M").last().dropna(how="any")
        cmp_ret = cmp_monthly.pct_change().dropna(how="any")
        cmp_base100 = (cmp_monthly / cmp_monthly.iloc[0]) * 100

        cmp_stats = []
        for isin in cmp_monthly.columns:
            series = cmp_monthly[isin]
            ret = cmp_ret[isin]
            years = (series.index[-1] - series.index[0]).days / 365.25
            cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
            vol = ret.std() * np.sqrt(12)
            sharpe = (ret.mean() * 12) / vol if vol and np.isfinite(vol) and vol > 0 else np.nan
            dd = (series / series.cummax() - 1).min()
            total = series.iloc[-1] / series.iloc[0] - 1
            cmp_stats.append(
                {
                    "ISIN": isin,
                    "Fundo": cmp_names.get(isin, isin),
                    "Retorno total (%)": total * 100,
                    "CAGR (%)": cagr * 100,
                    "Vol anual (%)": vol * 100,
                    "Sharpe (histórico)": sharpe,
                    "Max Drawdown (%)": dd * 100,
                    "Link FT Summary": make_ft_summary_link_from_isin(isin),
                    "Link FT Risk": make_ft_risk_link_from_isin(isin),
                }
            )

        cmp_labels = [_short_label(cmp_names.get(isin, isin), 30) for isin in cmp_base100.columns]
        cmp_labels = _unique_labels(cmp_labels)
        cmp_map = {isin: label for isin, label in zip(cmp_base100.columns, cmp_labels)}
        cmp_base100 = cmp_base100.rename(columns=cmp_map)
        cmp_color_map = _collect_fund_colors(cmp_base100.columns.tolist(), key_prefix="compare")
        _plot_interactive_lines(
            cmp_base100,
            title="Comparação de desempenho (base 100)",
            yaxis_title="Base 100",
            color_map=cmp_color_map,
            key="chart_compare_base100",
        )

        stats_df = pd.DataFrame(cmp_stats).sort_values("CAGR (%)", ascending=False)
        risk_df = pd.DataFrame(risk_rows)
        st.subheader("Métricas históricas (calculadas)")
        st.dataframe(
            stats_df.round(2),
            use_container_width=True,
            column_config={
                "Link FT Summary": st.column_config.LinkColumn("Link FT Summary"),
                "Link FT Risk": st.column_config.LinkColumn("Link FT Risk"),
            },
        )
        st.subheader("Risco FT (Risk tab)")
        st.dataframe(
            risk_df,
            use_container_width=True,
            column_config={
                "Link FT Summary": st.column_config.LinkColumn("Link FT Summary"),
                "Link FT Risk": st.column_config.LinkColumn("Link FT Risk"),
            },
        )
    st.stop()

default_isins = []
default_fund_options = []
default_weights = []
default_classes = []
if selected_portfolio != "(nova)":
    saved = portfolios[selected_portfolio]
    default_isins = saved.get("isins", default_isins)
    default_fund_options = saved.get("fund_options", default_fund_options)
    default_weights = saved.get("weights", default_weights)
    default_classes = saved.get("classes", default_classes)

def _pad_list(values: list, size: int, fill):
    values = list(values)
    if len(values) >= size:
        return values[:size]
    return values + [fill] * (size - len(values))

if not default_fund_options and default_isins:
    default_fund_options = [isin_to_option.get(isin, isin) for isin in default_isins]

default_count = max(len(default_fund_options), len(default_weights), len(default_classes), 0)
default_fund_options = _pad_list(default_fund_options, default_count, "")
default_weights = _pad_list(default_weights, default_count, 0.0)
default_classes = _pad_list(default_classes, default_count, "")

known_options = set(catalog_options)
missing_options = [opt for opt in default_fund_options if opt and opt not in known_options]
fund_options_for_ui = catalog_options + missing_options

default_data = pd.DataFrame(
    {
        "Ativo": pd.Series(default_fund_options, dtype="string"),
        "Alocacao_%": pd.Series(default_weights, dtype="float"),
        "Classe": pd.Series(default_classes, dtype="string"),
    }
)

table_key = f"fund_table::{selected_portfolio}"
if "funds_editor_version" not in st.session_state:
    st.session_state.funds_editor_version = 0
if st.session_state.get("fund_table_key") != table_key:
    st.session_state.funds_table = default_data.copy()
    st.session_state.fund_table_key = table_key
    st.session_state.funds_editor_version = int(st.session_state.get("funds_editor_version", 0)) + 1

with st.container(border=True):
    sim_asset_filter = st.radio(
        "Buscar em",
        ["Todos", "Fundos", "ETFs", "Stocks"],
        horizontal=True,
        key="sim_asset_filter",
    )
    picked_ft = _live_ft_searchbox(
        "Adicionar ativo (nome, ISIN ou ticker)",
        "Explorar.",
        key_prefix="sim_ft",
        asset_filter=sim_asset_filter,
    )
    add_col1, add_col2 = st.columns([1, 6])
    add_clear = add_col1.button("Limpar", key="sim_clear_btn", use_container_width=True)
    add_col2.caption("Selecione os ativos")
    if add_clear:
        st.session_state.funds_table = pd.DataFrame(columns=["Ativo", "Alocacao_%", "Classe"])
        st.session_state.funds_editor_version = int(st.session_state.get("funds_editor_version", 0)) + 1
        st.session_state.pop("sim_ft_searchbox", None)
        st.session_state.pop("sim_ft_searchbox_light", None)
        st.session_state.pop("sim_ft_searchbox_dark", None)
        st.session_state.pop("sim_last_selected", None)
        st.rerun()

    if picked_ft:
        symbol_value = str(picked_ft["symbol"]).strip().upper()
        parsed_value = _extract_isin_from_option(symbol_value, option_to_isin={}) or symbol_value
        auto_category = fetch_ft_morningstar_category(
            parsed_value,
            symbol_hint=symbol_value,
            asset_type_hint=str(picked_ft.get("asset_type", "")),
        )
        st.session_state.ft_identifier_hints[symbol_value] = {
            "xid": picked_ft["xid"],
            "symbol": symbol_value,
            "asset_type": picked_ft["asset_type"],
            "name": picked_ft["name"],
        }
        st.session_state.ft_identifier_hints[parsed_value] = st.session_state.ft_identifier_hints[symbol_value]
        new_row = pd.DataFrame(
            [
                {
                    "Ativo": parsed_value,
                    "Alocacao_%": 0.0,
                    "Classe": auto_category,
                }
            ]
        )
        st.session_state.funds_table = pd.concat([st.session_state.funds_table, new_row], ignore_index=True)
        st.session_state.funds_editor_version = int(st.session_state.get("funds_editor_version", 0)) + 1
        st.session_state.pop("sim_ft_searchbox", None)
        st.session_state.pop("sim_ft_searchbox_light", None)
        st.session_state.pop("sim_ft_searchbox_dark", None)
        st.session_state.pop("sim_last_selected", None)
        st.rerun()

if "funds_table" not in st.session_state:
    st.session_state.funds_table = default_data.copy()
    st.session_state.funds_editor_version = int(st.session_state.get("funds_editor_version", 0)) + 1

table_for_ui = st.session_state.funds_table.copy()
for _col in ("Ativo", "Classe"):
    if _col in table_for_ui.columns:
        table_for_ui[_col] = (
            table_for_ui[_col]
            .astype("string")
            .fillna("")
            .astype(str)
            .replace({"<NA>": "", "nan": ""})
        )
if "Alocacao_%" in table_for_ui.columns:
    table_for_ui["Alocacao_%"] = pd.to_numeric(table_for_ui["Alocacao_%"], errors="coerce").fillna(0.0)
_roa_values = table_for_ui["Ativo"].apply(
    lambda x: _resolve_option_roa(str(x), option_to_isin, isin_to_roa)
)
table_for_ui["ROA_%"] = _roa_values.apply(lambda v: _fmt_optional_number(v, decimals=2))

if table_for_ui.empty:
    st.info("Selecione um ativo na busca para começar a montar a carteira.")
    funds_df = pd.DataFrame(columns=["Ativo", "Alocacao_%", "Classe"])
else:
    _render_assets_grid_table(table_for_ui, option_to_isin, isin_to_roa)
    st.caption("ROA (%) é aplicado apenas para fundos presentes no catálogo local.")
    st.caption("Edite alocação e categoria abaixo. Para excluir um ativo, use o botão Remover.")

    hdr_col1, hdr_col2, hdr_col3, hdr_col4 = st.columns([4.8, 2.0, 2.6, 1.1])
    hdr_col1.caption("Ativo")
    hdr_col2.caption("Alocação (%)")
    hdr_col3.caption("Categoria")
    hdr_col4.caption(" ")

    remove_idx = None
    updated_rows: list[dict] = []
    for idx, row in table_for_ui.reset_index(drop=True).iterrows():
        ativo_val = str(row.get("Ativo", "")).strip()
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", ativo_val).strip("_").lower() or f"row_{idx}"
        key_suffix = f"{table_key}_{idx}_{safe}"

        row_col1, row_col2, row_col3, row_col4 = st.columns([4.8, 2.0, 2.6, 1.1])
        row_col1.markdown(
            f"<span class='asset-inline-name'>{html.escape(ativo_val)}</span>",
            unsafe_allow_html=True,
        )

        raw_weight = pd.to_numeric(row.get("Alocacao_%", 0.0), errors="coerce")
        weight_default = 0.0 if pd.isna(raw_weight) else float(raw_weight)
        weight_value = row_col2.number_input(
            "Alocação (%)",
            min_value=0.0,
            step=0.1,
            format="%.2f",
            value=weight_default,
            key=f"sim_alloc_{key_suffix}",
            label_visibility="collapsed",
        )
        class_value = row_col3.text_input(
            "Categoria",
            value=str(row.get("Classe", "")).strip(),
            key=f"sim_class_{key_suffix}",
            label_visibility="collapsed",
        )
        remove_clicked = row_col4.button("Remover", key=f"sim_remove_{key_suffix}", use_container_width=True)
        if remove_clicked and remove_idx is None:
            remove_idx = idx

        updated_rows.append(
            {
                "Ativo": ativo_val,
                "Alocacao_%": float(weight_value),
                "Classe": str(class_value).strip(),
            }
        )

    if remove_idx is not None:
        updated_rows = [r for i, r in enumerate(updated_rows) if i != remove_idx]
        st.session_state.funds_table = pd.DataFrame(updated_rows, columns=["Ativo", "Alocacao_%", "Classe"])
        st.rerun()

    st.session_state.funds_table = pd.DataFrame(updated_rows, columns=["Ativo", "Alocacao_%", "Classe"])
    funds_df = st.session_state.funds_table.copy()

    total_alloc = float(pd.to_numeric(funds_df.get("Alocacao_%", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    alloc_gap = total_alloc - 100.0
    if abs(alloc_gap) <= 0.01:
        alloc_cls = "ok"
        alloc_msg = f"Alocação total: {total_alloc:.2f}% (ok)"
    elif alloc_gap < 0:
        alloc_cls = "low"
        alloc_msg = f"Alocação total: {total_alloc:.2f}% (faltam {abs(alloc_gap):.2f}%)"
    else:
        alloc_cls = "high"
        alloc_msg = f"Alocação total: {total_alloc:.2f}% (excesso de {abs(alloc_gap):.2f}%)"
    st.markdown(f"<div class='alloc-total {alloc_cls}'>{alloc_msg}</div>", unsafe_allow_html=True)

col_run, col_save, col_export = st.columns(3)
run = col_run.button("Simular carteira")
save = col_save.button("Salvar carteira")
export_pdf = col_export.button("Exportar PDF")

if save:
    # Salva a carteira localmente
    isins, weights_arr, classes, selected_options, _ = _parse_funds_input(
        funds_df,
        option_to_isin,
        isin_to_roa,
    )
    data = {
        "name": portfolio_name,
        "isins": isins,
        "fund_options": selected_options,
        "weights": weights_arr.tolist(),
        "classes": classes,
        "initial_value": float(initial_value),
        "start_date": start_date.isoformat(),
        "rebalance_monthly": bool(rebalance_monthly),
    }
    save_portfolio(portfolio_name, data)
    st.success(f"Carteira '{portfolio_name}' salva.")

if run:
    # Executa a simulação
    isins, weights, classes, selected_options, roa_by_isin = _parse_funds_input(
        funds_df,
        option_to_isin,
        isin_to_roa,
    )

    if not isins:
        st.error("Informe ao menos um ativo.")
        st.stop()

    if len(isins) != len(weights):
        st.error("Ativo e alocação precisam ter o mesmo tamanho.")
        st.stop()

    if weights.sum() <= 0:
        st.error("A soma das alocações deve ser maior que 0.")
        st.stop()

    weights = weights / weights.sum()

    with st.spinner("Buscando dados..."):
        ft_hints = {
            identifier: st.session_state.ft_identifier_hints.get(identifier, {})
            for identifier in isins
            if identifier in st.session_state.ft_identifier_hints
        }
        prices, names = build_price_matrix(isins, start_date, max_cache_age, ft_hints=ft_hints)
        if prices.empty:
            st.error("Sem dados suficientes para o intervalo informado.")
            st.stop()

        isins, weights, classes = _filter_failed_isins(isins, weights, classes, prices)
        if len(isins) == 0 or len(weights) == 0:
            st.error("Nenhum ISIN válido retornou dados.")
            st.stop()

        weights = weights / weights.sum()
        result = compute_portfolio(prices, weights, initial_value, rebalance_monthly)
        st.session_state.last_prices = prices
        st.session_state.last_result = result
        st.session_state.last_isins = isins
        st.session_state.last_weights = weights
        st.session_state.last_classes = classes
        st.session_state.last_names = names
        st.session_state.last_start_date = start_date
        st.session_state.last_initial_value = initial_value
        st.session_state.last_rebalance = rebalance_monthly
        st.session_state.last_portfolio_name = portfolio_name
        st.session_state.last_selected_options = selected_options

        roa_values = np.array([roa_by_isin.get(isin, np.nan) for isin in isins], dtype=float)
        valid_roa = np.isfinite(roa_values)
        portfolio_roa = float(np.dot(weights[valid_roa], roa_values[valid_roa])) if valid_roa.any() else np.nan
        st.session_state.last_portfolio_roa = portfolio_roa

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR", f"{result['cagr']*100:.2f}%")
    col2.metric("Volatilidade anual", f"{result['vol_annual']*100:.2f}%")
    col3.metric("Max Drawdown", f"{result['drawdown'].min()*100:.2f}%")
    roa_value = st.session_state.get("last_portfolio_roa", np.nan)
    roa_value_txt = _fmt_optional_number(roa_value, decimals=2)
    col4.metric("ROA ponderado", "-" if roa_value_txt == "" else f"{roa_value_txt}%")

    st.subheader("Correlação entre ativos")
    monthly_prices = prices.resample("M").last().dropna(how="any")
    monthly_returns_assets = monthly_prices.pct_change().dropna(how="any")
    corr = monthly_returns_assets.corr().round(2)
    short_names = [_short_label(names.get(isin, isin)) for isin in isins]
    short_names = _unique_labels(short_names)
    label_map = {isin: short for isin, short in zip(isins, short_names)}
    corr = corr.rename(index=label_map, columns=label_map)
    yearly_prices = prices.resample("Y").last().dropna(how="any")
    yearly_returns_assets = yearly_prices.pct_change().dropna(how="any")
    corr_y = yearly_returns_assets.corr().round(2)
    corr_y = corr_y.rename(index=label_map, columns=label_map)
    corr_col1, corr_col2 = st.columns(2)
    with corr_col1:
        _plot_corr_heatmap(corr, title="Mensal", key="corr_monthly")
    with corr_col2:
        _plot_corr_heatmap(corr_y, title="Anual", key="corr_yearly")

    st.subheader("Evolução da carteira")
    _plot_interactive_lines(
        result["portfolio_value"].to_frame(name="Carteira"),
        title="Evolução da carteira",
        yaxis_title="Valor",
        color_map={"Carteira": "#60a5fa"},
        key="chart_portfolio_value",
    )

    st.subheader("Drawdown (%)")
    drawdown_pct = result["drawdown"] * 100
    _plot_interactive_lines(
        drawdown_pct.to_frame(name="Drawdown (%)"),
        title="Drawdown (%)",
        yaxis_title="%",
        color_map={"Drawdown (%)": "#ef4444"},
        key="chart_drawdown_pct",
    )

    st.subheader("Retorno mês a mês (tabela)")
    monthly_pivot = monthly_table(result["portfolio_value"]) * 100
    _show_dataframe_themed(monthly_pivot.round(2), use_container_width=True)

    st.subheader("Retorno anual")
    yearly = annual_returns(result["portfolio_value"]) * 100
    _show_dataframe_themed(yearly.to_frame(name="Retorno_%").round(2), use_container_width=True)

    st.subheader("Retornos anualizados (1, 3, 5, 10 anos e total)")
    trailing = trailing_returns(result["portfolio_value"]) * 100
    _show_dataframe_themed(trailing.round(2), use_container_width=True)

    st.subheader("Preços (fundos) - mensal")
    monthly_prices_named = monthly_prices.rename(columns=label_map)
    fund_color_map = _collect_fund_colors(monthly_prices_named.columns.tolist(), key_prefix="portfolio")
    _plot_interactive_lines(
        monthly_prices_named,
        title="Preços dos fundos (mensal)",
        yaxis_title="Preço",
        color_map=fund_color_map,
        key="chart_prices_monthly",
    )

    st.subheader("Desempenho relativo dos fundos (base 100)")
    perf = (monthly_prices / monthly_prices.iloc[0]) * 100
    perf = perf.rename(columns=label_map)
    _plot_interactive_lines(
        perf,
        title="Desempenho relativo (base 100)",
        yaxis_title="Base 100",
        color_map=fund_color_map,
        key="chart_perf_base100",
    )

    links_df = pd.DataFrame(
        {
            "Fundo": [names.get(isin, isin) for isin in isins],
            "ISIN": isins,
            "FT Summary": [make_ft_summary_link_from_isin(isin) for isin in isins],
            "FT Risk": [make_ft_risk_link_from_isin(isin) for isin in isins],
        }
    )
    st.subheader("Links Financial Times")
    _render_ft_links_table(links_df)

    st.subheader("Alocação")
    fund_labels = [_short_label(names.get(isin, isin)) for isin in isins]
    fund_labels = _unique_labels(fund_labels)
    alloc_by_fund = pd.Series(weights, index=fund_labels)
    alloc_col1, alloc_col2 = st.columns(2)
    fig_f, ax_f = plt.subplots(figsize=(3.6, 2.8))
    ax_f.pie(alloc_by_fund.values, labels=alloc_by_fund.index, autopct="%.1f%%")
    ax_f.set_title("Por fundo")
    alloc_col1.pyplot(fig_f)

    alloc_by_class = pd.Series(weights, index=classes).groupby(level=0).sum()
    fig_c, ax_c = plt.subplots(figsize=(3.6, 2.8))
    ax_c.pie(alloc_by_class.values, labels=alloc_by_class.index, autopct="%.1f%%")
    ax_c.set_title("Por classe")
    alloc_col2.pyplot(fig_c)

if export_pdf:
    # Exporta relatório em PDF com gráficos e tabelas
    try:
        from fpdf import FPDF
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception:
        st.error("Para exportar PDF, instale: pip install fpdf2 matplotlib pillow")
        st.stop()

    if "last_result" not in st.session_state:
        isins, weights, classes, _, _ = _parse_funds_input(
            funds_df,
            option_to_isin,
            isin_to_roa,
        )
        classes = [cls if cls else "Outros" for cls in classes]
        if weights.sum() <= 0:
            st.error("A soma das alocações deve ser maior que 0.")
            st.stop()
        weights = weights / weights.sum()
        ft_hints = {
            identifier: st.session_state.ft_identifier_hints.get(identifier, {})
            for identifier in isins
            if identifier in st.session_state.ft_identifier_hints
        }
        prices, names = build_price_matrix(isins, start_date, max_cache_age, ft_hints=ft_hints)
        if prices.empty:
            st.error("Sem dados suficientes para o intervalo informado.")
            st.stop()

        isins, weights, classes = _filter_failed_isins(isins, weights, classes, prices)
        if len(isins) == 0 or len(weights) == 0:
            st.error("Nenhum ISIN válido retornou dados.")
            st.stop()

        weights = weights / weights.sum()
        result = compute_portfolio(prices, weights, initial_value, rebalance_monthly)
    else:
        isins = st.session_state.last_isins
        weights = st.session_state.last_weights
        classes = st.session_state.last_classes
        names = st.session_state.last_names
        result = st.session_state.last_result
        prices = st.session_state.last_prices
        start_date = st.session_state.last_start_date
        initial_value = st.session_state.last_initial_value
        rebalance_monthly = st.session_state.last_rebalance
        portfolio_name = st.session_state.last_portfolio_name

    pdf_filename = f"{portfolio_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    # Export sempre em visual claro, independente do tema da UI.
    with plt.style.context("default"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Gráfico: Evolução da carteira
            fig1, ax1 = plt.subplots(figsize=(8, 3))
            result["portfolio_value"].plot(ax=ax1, color="#1f77b4")
            ax1.set_title("Evolução da carteira")
            ax1.set_ylabel("Valor")
            img_portfolio = tmpdir_path / "portfolio.png"
            _plot_and_save(fig1, img_portfolio)

            # Gráfico: Drawdown (em %)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            (result["drawdown"] * 100).plot(ax=ax2, color="#d62728")
            ax2.set_title("Drawdown (%)")
            ax2.set_ylabel("%")
            img_dd = tmpdir_path / "drawdown.png"
            _plot_and_save(fig2, img_dd)

            # Gráfico: Retornos mensais (barra) com rótulos por ano
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            monthly_pct = result["monthly_returns"].mul(100)
            ax3.bar(range(len(monthly_pct)), monthly_pct.values, color="#2ca02c")
            ax3.set_title("Retornos mensais (%)")
            ax3.set_ylabel("%")
            ax3.grid(axis="y", linestyle="--", alpha=0.4)
            year_ticks = []
            year_labels = []
            for i, dt in enumerate(monthly_pct.index):
                if dt.month == 1:
                    year_ticks.append(i)
                    year_labels.append(str(dt.year))
            ax3.set_xticks(year_ticks)
            ax3.set_xticklabels(year_labels, rotation=0, fontsize=8)
            img_monthly = tmpdir_path / "monthly.png"
            _plot_and_save(fig3, img_monthly)

            # Heatmap de correlação (mensal)
            monthly_prices = prices.resample("M").last().dropna(how="any")
            monthly_returns_assets = monthly_prices.pct_change().dropna(how="any")
            corr = monthly_returns_assets.corr()
            short_names = [_short_label(names.get(isin, isin)) for isin in isins]
            short_names = _unique_labels(short_names)
            label_map = {isin: short for isin, short in zip(isins, short_names)}
            corr = corr.rename(index=label_map, columns=label_map)
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            corr_vmin = float(np.nanmin(corr.values))
            corr_vmax = float(np.nanmax(corr.values))
            if np.isclose(corr_vmin, corr_vmax):
                corr_vmin = max(0.0, corr_vmin - 0.01)
                corr_vmax = min(1.0, corr_vmax + 0.01)
            cax = ax4.imshow(corr.values, cmap="RdYlGn", vmin=corr_vmin, vmax=corr_vmax)
            ax4.set_title("Correlação (mensal)")
            ax4.set_xticks(range(len(corr.columns)))
            ax4.set_yticks(range(len(corr.index)))
            ax4.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
            ax4.set_yticklabels(corr.index, fontsize=7)
            ax4.set_xticks(np.arange(-0.5, len(corr.columns), 1), minor=True)
            ax4.set_yticks(np.arange(-0.5, len(corr.index), 1), minor=True)
            ax4.grid(which="minor", color="white", linestyle="-", linewidth=1)
            ax4.tick_params(which="minor", bottom=False, left=False)
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    ax4.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
            fig4.colorbar(cax, ax=ax4, fraction=0.046, pad=0.04)
            img_corr = tmpdir_path / "corr.png"
            _plot_and_save(fig4, img_corr)

            # Heatmap de correlação (anual)
            yearly_prices = prices.resample("Y").last().dropna(how="any")
            yearly_returns_assets = yearly_prices.pct_change().dropna(how="any")
            corr_y = yearly_returns_assets.corr().rename(index=label_map, columns=label_map)
            fig4b, ax4b = plt.subplots(figsize=(6, 4))
            corr_y_vmin = float(np.nanmin(corr_y.values))
            corr_y_vmax = float(np.nanmax(corr_y.values))
            if np.isclose(corr_y_vmin, corr_y_vmax):
                corr_y_vmin = max(0.0, corr_y_vmin - 0.01)
                corr_y_vmax = min(1.0, corr_y_vmax + 0.01)
            cax2 = ax4b.imshow(corr_y.values, cmap="RdYlGn", vmin=corr_y_vmin, vmax=corr_y_vmax)
            ax4b.set_title("Correlação (anual)")
            ax4b.set_xticks(range(len(corr_y.columns)))
            ax4b.set_yticks(range(len(corr_y.index)))
            ax4b.set_xticklabels(corr_y.columns, rotation=45, ha="right", fontsize=7)
            ax4b.set_yticklabels(corr_y.index, fontsize=7)
            ax4b.set_xticks(np.arange(-0.5, len(corr_y.columns), 1), minor=True)
            ax4b.set_yticks(np.arange(-0.5, len(corr_y.index), 1), minor=True)
            ax4b.grid(which="minor", color="white", linestyle="-", linewidth=1)
            ax4b.tick_params(which="minor", bottom=False, left=False)
            for i in range(len(corr_y.index)):
                for j in range(len(corr_y.columns)):
                    ax4b.text(j, i, f"{corr_y.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
            fig4b.colorbar(cax2, ax=ax4b, fraction=0.046, pad=0.04)
            img_corr_y = tmpdir_path / "corr_y.png"
            _plot_and_save(fig4b, img_corr_y)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()
            # Header band
            pdf.set_fill_color(32, 80, 129)
            pdf.rect(0, 0, 210, 20, "F")
            pdf.set_font("Arial", style="B", size=16)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 12, f"Relatorio - {portfolio_name}", ln=True)
            pdf.set_text_color(0, 0, 0)
            # Bloco de informações em formato de tabela
            info_df = pd.DataFrame(
                {
                    "Campo": [
                        "Data inicial",
                        "Valor inicial",
                        "Rebalanceamento mensal",
                        "CAGR",
                        "Volatilidade anual",
                        "Max Drawdown",
                    ],
                    "Valor": [
                        start_date.isoformat(),
                        f"{initial_value:.2f}",
                        "Sim" if rebalance_monthly else "Não",
                        f"{result['cagr']*100:.2f}%",
                        f"{result['vol_annual']*100:.2f}%",
                        f"{result['drawdown'].min()*100:.2f}%",
                    ],
                }
            )

            info_img = tmpdir_path / "info.png"
            _table_image(
                info_df,
                info_img,
                "Resumo da simulação",
                font_size=9,
                figsize=(6.5, 2.5),
                show_index=False,
                title_pad=14,
                top_margin=0.90
            )
            pdf.image(str(info_img), x=10, w=110)
            pdf.ln(4)

            # Fundos e alocacao em tabela
            funds_df_pdf = pd.DataFrame(
                {
                    "ISIN": isins,
                    "Fundo": [names.get(i, i) for i in isins],
                    "Alocacao_%": [f"{w*100:.2f}%" for w in weights],
                    "Classe": classes,
                }
            )
            funds_img = tmpdir_path / "fundos.png"
            _table_image(
                funds_df_pdf,
                funds_img,
                "Alocação",
                font_size=8,
                figsize=(7.2, 3.1),
                show_index=False,
                wrap_chars=28,
                title_pad=20,
                top_margin=0.88
            )
            pdf.image(str(funds_img), x=10, w=170)

            pdf.ln(4)
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 8, "Graficos", ln=True)
            pdf.set_font("Arial", size=11)

            pdf.image(str(img_portfolio), w=180)
            pdf.ln(2)
            pdf.image(str(img_dd), w=180)
            pdf.ln(2)
            pdf.image(str(img_monthly), w=180)
            pdf.ln(2)
            pdf.image(str(img_corr), w=140)
            pdf.ln(2)
            pdf.image(str(img_corr_y), w=140)

            # Gráfico: desempenho dos fundos (base 100)
            perf = (monthly_prices / monthly_prices.iloc[0]) * 100
            perf = perf.rename(columns=label_map)
            fig5, ax5 = plt.subplots(figsize=(8, 3))
            perf.plot(ax=ax5)
            ax5.set_title("Desempenho relativo dos fundos (base 100)")
            ax5.set_ylabel("Base 100")
            img_perf = tmpdir_path / "perf.png"
            _plot_and_save(fig5, img_perf)
            pdf.ln(2)
            pdf.image(str(img_perf), w=180)

            # Tabelas (retornos) como imagens formatadas
            pdf.add_page()
            monthly_pivot = (monthly_table(result["portfolio_value"]) * 100).round(2)
            yearly = (annual_returns(result["portfolio_value"]) * 100).round(2)
            trailing = (trailing_returns(result["portfolio_value"]) * 100).round(2)

            # Retorno anual em formato horizontal (anos como colunas)
            yearly_row = pd.DataFrame([yearly.values], columns=[str(y) for y in yearly.index], index=["Retorno_%"])

            monthly_img = tmpdir_path / "table_monthly.png"
            yearly_img = tmpdir_path / "table_yearly.png"
            trailing_img = tmpdir_path / "table_trailing.png"

            _table_image(
                monthly_pivot.fillna(""),
                monthly_img,
                "Retorno mensal (em %)",
                font_size=7,
                title_pad=16,
                top_margin=0.90
            )
            _table_image(
                yearly_row.fillna(""),
                yearly_img,
                "Retorno anual (em %)",
                font_size=9,
                figsize=(10, 1.8),
                show_index=False,
                title_pad=14,
                top_margin=0.88
            )
            _table_image(
                trailing.fillna(""),
                trailing_img,
                "Retornos anualizados (em %)",
                font_size=9,
                figsize=(10, 1.8),
                show_index=False,
                title_pad=14,
                top_margin=0.88
            )

            # Usa alturas reais das imagens para evitar sobreposição
            def _img_h(path: Path, w_mm: float) -> float:
                from PIL import Image
                with Image.open(path) as im:
                    w_px, h_px = im.size
                return (h_px / w_px) * w_mm

            w = 190
            pdf.image(str(monthly_img), w=w)
            pdf.ln(2)
            y_pos = pdf.get_y()
            pdf.image(str(yearly_img), x=10, y=y_pos, w=w)
            pdf.ln(_img_h(yearly_img, w) + 2)
            pdf.image(str(trailing_img), x=10, w=w)

            # Pies de alocacao (na pagina 2, abaixo das tabelas)
            labels = [_short_label(names.get(i, i)) for i in isins]
            labels = _unique_labels(labels)
            alloc_by_fund = pd.Series(weights, index=labels)
            alloc_by_class = pd.Series(weights, index=classes).groupby(level=0).sum()

            fig_p1, ax_p1 = plt.subplots(figsize=(4.2, 3.0))
            ax_p1.pie(alloc_by_fund.values, labels=alloc_by_fund.index, autopct="%.1f%%")
            ax_p1.set_title("Alocacao por fundo")
            img_pfund = tmpdir_path / "pie_funds.png"
            _plot_and_save(fig_p1, img_pfund)

            fig_p2, ax_p2 = plt.subplots(figsize=(4.2, 3.0))
            ax_p2.pie(alloc_by_class.values, labels=alloc_by_class.index, autopct="%.1f%%")
            ax_p2.set_title("Alocacao por classe")
            img_pclass = tmpdir_path / "pie_class.png"
            _plot_and_save(fig_p2, img_pclass)

            y_pos = pdf.get_y() + 8
            pdf.image(str(img_pfund), x=15, y=y_pos, w=78)
            pdf.image(str(img_pclass), x=115, y=y_pos, w=78)

            pdf_raw = pdf.output(dest="S")
            if isinstance(pdf_raw, (bytes, bytearray)):
                pdf_bytes = bytes(pdf_raw)
            else:
                # Compatibilidade com versões que retornam string.
                pdf_bytes = str(pdf_raw).encode("latin-1")

    st.success("Relatório pronto para download.")
    st.download_button(
        "Baixar PDF",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True,
        key=f"download_pdf_{pdf_filename}",
    )





