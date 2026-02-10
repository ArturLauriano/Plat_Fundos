import json
import os
import re
import tempfile
import time
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
BASE_START_DATE = "2000/01/01"
CACHE_DIR = Path("cache")
NAMES_FILE = CACHE_DIR / "fund_names.json"
PORTFOLIO_FILE = Path("carteiras.json")
REPORTS_DIR = Path("reports")
FUNDS_CATALOG_FILE = Path("fundos_completos_fundname_isin_roa.csv")
# Tenta várias moedas antes de falhar; sem sufixo é fallback final.
XID_CURRENCY_FALLBACKS = ["USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", None]
XID_LOOKUP_RETRIES = 2
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
    """Converte opção selecionada em ISIN."""
    text = str(option).strip()
    if not text:
        return None
    if text in option_to_isin:
        return option_to_isin[text]
    match = re.search(r"[A-Z]{2}[A-Z0-9]{10}", text.upper())
    if match:
        return match.group(0)
    return text.upper()


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
        raw_option = row.get("Fundo", "")
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

        raw_cls = row.get("Classe", "")
        raw_custom = row.get("Classe_Digitada", "")
        cls = "" if pd.isna(raw_cls) else str(raw_cls).strip()
        custom = "" if pd.isna(raw_custom) else str(raw_custom).strip()
        if cls == "Digitar" and custom:
            class_name = custom
        else:
            class_name = cls if cls else "RF Low Duration"

        isins.append(isin)
        weights.append(weight)
        classes.append(class_name)
        selected_options.append(option)

        roa = isin_to_roa.get(isin)
        if roa is not None and np.isfinite(roa):
            roa_by_isin[isin] = float(roa)

    return isins, np.array(weights, dtype=float), classes, selected_options, roa_by_isin


def _resolve_option_roa(option: str, option_to_isin: dict[str, str], isin_to_roa: dict[str, float]) -> float | None:
    """Retorna ROA do fundo selecionado (ou None quando indisponível)."""
    isin = _extract_isin_from_option(option, option_to_isin)
    if not isin:
        return None
    value = isin_to_roa.get(isin)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


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


def make_ft_summary_link_from_isin(isin: str) -> str:
    symbol = resolve_ft_symbol_for_isin(isin)
    return f"{BASE_URL}/data/funds/tearsheet/summary?s={quote_plus(symbol)}"


def make_ft_risk_link_from_isin(isin: str) -> str:
    symbol = resolve_ft_symbol_for_isin(isin)
    return f"{BASE_URL}/data/funds/tearsheet/risk?s={quote_plus(symbol)}"


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

    symbol = resolve_ft_symbol_for_isin(isin)
    risk_url = f"{BASE_URL}/data/funds/tearsheet/risk?s={quote_plus(symbol)}"
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
    max_age_days: int = 7
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
    max_age_days: int
) -> tuple[pd.DataFrame, dict]:
    """Monta matriz de preços (colunas por fundo) e mapa ISIN->nome."""
    price_series = []
    names = {}
    for isin in isins:
        try:
            df, fund_name = load_or_fetch_history(
                isin,
                start_date=start_date.strftime("%Y/%m/%d"),
                max_age_days=max_age_days
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


def load_saved_portfolios() -> dict:
    """Carrega carteiras salvas do arquivo local."""
    if PORTFOLIO_FILE.exists():
        with PORTFOLIO_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_portfolio(name: str, data: dict):
    """Salva ou atualiza carteira no arquivo local."""
    portfolios = load_saved_portfolios()
    portfolios[name] = data
    with PORTFOLIO_FILE.open("w", encoding="utf-8") as f:
        json.dump(portfolios, f, ensure_ascii=False, indent=2)


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
        template="plotly_dark",
        height=430,
        margin={"l": 20, "r": 20, "t": 95, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08, "xanchor": "left", "x": 0},
        hovermode="x unified",
        yaxis_title=yaxis_title,
    )
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def _plot_corr_heatmap(corr: pd.DataFrame, title: str, key: str | None = None):
    """Heatmap de correlação em escala divergente (-1 a +1)."""
    if go is None:
        st.dataframe(corr, use_container_width=True)
        return
    heat = go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        zmin=-1,
        zmax=1,
        colorscale="RdBu_r",
        text=np.round(corr.values, 2),
        texttemplate="%{text:.2f}",
        colorbar={"title": "Corr"},
    )
    fig = go.Figure(data=[heat])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis={"tickangle": -35},
    )
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


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
st.set_page_config(page_title="Simulador de Fundos Offshore", layout="wide")
st.markdown(
    """
<style>
.stApp {
  background:
    radial-gradient(1000px 500px at 0% 0%, rgba(22,108,255,0.14), transparent 60%),
    radial-gradient(1000px 600px at 100% 0%, rgba(0,196,140,0.10), transparent 55%),
    #0a1018;
}
.block-container { padding-top: 1.6rem; }
h1, h2, h3, .stMarkdown p, .stCaption, label { font-family: "Segoe UI", "Trebuchet MS", sans-serif; }
.hero-card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  background: linear-gradient(135deg, rgba(11,26,44,0.86), rgba(9,17,28,0.86));
  margin-bottom: 1rem;
}
.hero-title {
  margin: 0;
  font-size: 2.1rem;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.hero-subtitle {
  margin-top: .35rem;
  color: rgba(230,240,255,0.82);
  font-size: .98rem;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="hero-card">
  <p class="hero-title">Simulador de Fundos Offshore</p>
  <p class="hero-subtitle">Monte carteiras e compare fundos com busca por nome ou ISIN.</p>
</div>
""",
    unsafe_allow_html=True,
)
app_mode = st.radio(
    "Escolha a ferramenta",
    ["Simulador de Carteiras", "Comparador de Fundos"],
    horizontal=True,
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
    max_cache_age = st.number_input("Atualizar cache (dias)", min_value=1, max_value=90, value=7, step=1)
    rebalance_monthly = st.checkbox(
        "Rebalanceamento mensal",
        value=bool(st.session_state.get("rebalance_monthly", True))
    )

st.subheader("Fundos e alocação")
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

if app_mode == "Comparador de Fundos":
    st.subheader("Comparador de Fundos")
    st.caption("Compare risco e retorno entre fundos com métricas históricas e dados de risco do Financial Times.")

    if "comparison_list" not in st.session_state:
        st.session_state.comparison_list = []

    with st.container(border=True):
        cmp_query = st.text_input(
            "Adicionar para comparação (nome ou ISIN)",
            value="",
            placeholder="Digite o nome do fundo ou um ISIN",
            key="cmp_add_query_top",
        )
        cmp_suggestions = _search_catalog_options(catalog_df, cmp_query)
        cmp_typed_isin = str(cmp_query).strip().upper()
        cmp_isin_valid = bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}", cmp_typed_isin))
        cmp_typed_entry = f"{cmp_typed_isin} (ISIN digitado)"
        if cmp_isin_valid and cmp_typed_entry not in cmp_suggestions:
            cmp_suggestions = [cmp_typed_entry] + cmp_suggestions

        cmp_pick = st.selectbox(
            "Sugestões comparação",
            options=cmp_suggestions if cmp_suggestions else [""],
            index=0,
            label_visibility="collapsed",
            key="cmp_add_pick_top",
        )

        cmp_col1, cmp_col2, cmp_col3 = st.columns([1, 1, 6])
        cmp_add = cmp_col1.button("Adicionar", key="cmp_add_btn_top", use_container_width=True)
        cmp_clear = cmp_col2.button("Limpar", key="cmp_clear_btn_top", use_container_width=True)
        cmp_col3.caption("Selecione pelo menos 2 fundos para comparar.")

        if cmp_clear:
            st.session_state.comparison_list = []
        if cmp_add:
            value = str(cmp_pick).strip()
            if value:
                if value.endswith("(ISIN digitado)"):
                    value = cmp_typed_isin
                if value not in st.session_state.comparison_list:
                    st.session_state.comparison_list.append(value)

    cmp_choices = st.multiselect(
        "Fundos para comparar",
        options=st.session_state.comparison_list,
        default=st.session_state.comparison_list,
        key="cmp_multiselect_top",
    )

    if st.button("Comparar fundos", key="cmp_run_btn_top"):
        cmp_df = pd.DataFrame(
            {"Fundo": cmp_choices, "Alocacao_%": [1.0] * len(cmp_choices), "Classe": [""] * len(cmp_choices)}
        )
        cmp_isins, _, _, _, _ = _parse_funds_input(cmp_df, option_to_isin, isin_to_roa)
        cmp_isins = list(dict.fromkeys(cmp_isins))

        if len(cmp_isins) < 2:
            st.warning("Selecione pelo menos 2 fundos para comparar.")
            st.stop()

        with st.spinner("Buscando históricos e risco dos fundos..."):
            cmp_prices, cmp_names = build_price_matrix(cmp_isins, start_date, max_cache_age)
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
            st.error("Sem dados em comum para os fundos selecionados.")
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
class_options = [
    "RF Low Duration",
    "RF Unconstrained",
    "RF HY",
    "RF EM",
    "Balanced",
    "Long and Short",
    "Global Equities",
    "US Equities",
    "Digitar",
]
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

default_count = max(len(default_fund_options), len(default_weights), len(default_classes), 1)
default_fund_options = _pad_list(default_fund_options, default_count, "")
default_weights = _pad_list(default_weights, default_count, 0.0)
default_classes = _pad_list(default_classes, default_count, "RF Low Duration")

known_options = set(catalog_options)
missing_options = [opt for opt in default_fund_options if opt and opt not in known_options]
fund_options_for_ui = catalog_options + missing_options

default_data = pd.DataFrame(
    {
        "Fundo": pd.Series(default_fund_options, dtype="string"),
        "Alocacao_%": pd.Series(default_weights, dtype="float"),
        "Classe": pd.Series(default_classes, dtype="string"),
        "Classe_Digitada": pd.Series([""] * default_count, dtype="string"),
    }
)

table_key = f"fund_table::{selected_portfolio}"
if st.session_state.get("fund_table_key") != table_key:
    st.session_state.funds_table = default_data.copy()
    st.session_state.fund_table_key = table_key

with st.container(border=True):
    add_query = st.text_input(
        "Adicionar fundo (nome ou ISIN)",
        value="",
        placeholder="Ex.: BlackRock... ou LU1387591990",
        key="fund_add_query",
    )
    suggestion_options = _search_catalog_options(catalog_df, add_query)

    typed_isin = str(add_query).strip().upper()
    typed_isin_valid = bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}", typed_isin))
    typed_entry = f"{typed_isin} (ISIN digitado)"
    if typed_isin_valid and typed_entry not in suggestion_options:
        suggestion_options = [typed_entry] + suggestion_options

    selected_suggestion = st.selectbox(
        "Sugestões",
        options=suggestion_options if suggestion_options else [""],
        index=0,
        label_visibility="collapsed",
        key="fund_add_pick",
    )

    add_col1, add_col2 = st.columns([1, 6])
    add_clicked = add_col1.button("Adicionar", use_container_width=True)
    add_col2.caption("Digite para buscar os mais próximos. Também aceita ISIN fora da base.")

    if add_clicked:
        picked = str(selected_suggestion).strip()
        if not picked:
            st.warning("Digite um nome ou ISIN para adicionar.")
        else:
            if picked.endswith("(ISIN digitado)"):
                value_to_add = typed_isin
            else:
                value_to_add = picked
            new_row = pd.DataFrame(
                [
                    {
                        "Fundo": value_to_add,
                        "Alocacao_%": 0.0,
                        "Classe": "RF Low Duration",
                        "Classe_Digitada": "",
                    }
                ]
            )
            st.session_state.funds_table = pd.concat(
                [st.session_state.funds_table, new_row], ignore_index=True
            )

if "funds_table" not in st.session_state:
    st.session_state.funds_table = default_data.copy()

table_for_ui = st.session_state.funds_table.copy()
table_for_ui["ROA_%"] = table_for_ui["Fundo"].apply(
    lambda x: _resolve_option_roa(str(x), option_to_isin, isin_to_roa)
)
editor_options = list(fund_options_for_ui)
seen_options = set(editor_options)
for value in table_for_ui["Fundo"].astype(str).str.strip().tolist():
    if value and value not in seen_options:
        editor_options.append(value)
        seen_options.add(value)

funds_df = st.data_editor(
    table_for_ui,
    num_rows="dynamic",
    column_config={
        "Fundo": (
            st.column_config.SelectboxColumn("Fundo", options=editor_options)
            if editor_options
            else st.column_config.TextColumn("Fundo (nome ou ISIN)")
        ),
        "Alocacao_%": st.column_config.NumberColumn("Alocação (%)", format="%.2f"),
        "Classe": st.column_config.SelectboxColumn("Classe", options=class_options),
        "Classe_Digitada": st.column_config.TextColumn("Classe (se Digitar)"),
        "ROA_%": st.column_config.NumberColumn("ROA (%)", format="%.2f"),
    },
    disabled=["ROA_%"],
    use_container_width=True,
    key="funds_data_editor",
)
st.session_state.funds_table = funds_df.drop(columns=["ROA_%"], errors="ignore").copy()

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
        st.error("Informe ao menos um fundo.")
        st.stop()

    if len(isins) != len(weights):
        st.error("Fundo e alocação precisam ter o mesmo tamanho.")
        st.stop()

    if weights.sum() <= 0:
        st.error("A soma das alocações deve ser maior que 0.")
        st.stop()

    weights = weights / weights.sum()

    with st.spinner("Baixando histórico e calculando..."):
        prices, names = build_price_matrix(isins, start_date, max_cache_age)
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
    col4.metric("ROA ponderado", "-" if not np.isfinite(roa_value) else f"{roa_value:.2f}%")

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
    corr_col1.markdown("**Mensal**")
    corr_col1.dataframe(
        corr.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}"),
        use_container_width=True,
    )
    corr_col2.markdown("**Anual**")
    corr_col2.dataframe(
        corr_y.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}"),
        use_container_width=True,
    )

    st.subheader("Evolução da carteira")
    st.line_chart(result["portfolio_value"])

    st.subheader("Drawdown (%)")
    drawdown_pct = result["drawdown"] * 100
    st.line_chart(drawdown_pct)

    st.subheader("Retorno mês a mês (tabela)")
    monthly_pivot = monthly_table(result["portfolio_value"]) * 100
    st.dataframe(monthly_pivot.round(2), use_container_width=True)

    st.subheader("Retorno anual")
    yearly = annual_returns(result["portfolio_value"]) * 100
    st.dataframe(yearly.to_frame(name="Retorno_%").round(2), use_container_width=True)

    st.subheader("Retornos anualizados (1, 3, 5, 10 anos e total)")
    trailing = trailing_returns(result["portfolio_value"]) * 100
    st.dataframe(trailing.round(2), use_container_width=True)

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
    st.dataframe(
        links_df,
        use_container_width=True,
        column_config={
            "FT Summary": st.column_config.LinkColumn("FT Summary"),
            "FT Risk": st.column_config.LinkColumn("FT Risk"),
        },
    )

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
        prices, names = build_price_matrix(isins, start_date, max_cache_age)
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





