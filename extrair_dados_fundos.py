import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from pathlib import Path

BASE_URL = "https://markets.ft.com"
REQUEST_TIMEOUT = 12
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
}

DEFAULT_CURRENCY_FALLBACKS: list[str | None] = [
    "USD",
    "EUR",
    "GBP",
    "CHF",
    "JPY",
    "CAD",
    "AUD",
    None,
]


def _extract_price_currency(soup: BeautifulSoup) -> str | None:
    """Extrai Price currency da tabela de perfil, quando presente."""
    for tr in soup.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        if th.get_text(" ", strip=True).lower() == "price currency":
            value = td.get_text(" ", strip=True).upper()
            return value or None
    return None


def _extract_first_tearsheet_link(soup: BeautifulSoup) -> str | None:
    """Na pagina de busca, pega o primeiro link de tearsheet de fundo."""
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        if "/data/funds/tearsheet/" in href:
            return href if href.startswith("http") else f"{BASE_URL}{href}"
    return None


def _extract_xid_and_name(soup: BeautifulSoup, html_text: str) -> tuple[str | None, str | None]:
    """Extrai XID e nome com fallback para layouts diferentes."""
    section = soup.find("section", class_="mod-tearsheet-add-to-watchlist")
    if section and section.get("data-mod-config"):
        try:
            config = json.loads(section["data-mod-config"])
            xid = config.get("xid")
            if xid:
                name_tag = soup.select_one(
                    "h1.mod-tearsheet-overview__header__name, "
                    "h1.mod-tearsheet-overview__header__name--large, "
                    "h1.mod-tearsheet-overview__header__name--small, "
                    "h1"
                )
                fund_name = name_tag.get_text(strip=True) if name_tag else None
                return xid, fund_name
        except Exception:
            pass

    match = re.search(r"\"xid\"\\s*:\\s*\"(\\d+)\"", html_text)
    if match:
        xid = match.group(1)
        name_tag = soup.select_one(
            "h1.mod-tearsheet-overview__header__name, "
            "h1.mod-tearsheet-overview__header__name--large, "
            "h1.mod-tearsheet-overview__header__name--small, "
            "h1"
        )
        fund_name = name_tag.get_text(strip=True) if name_tag else None
        return xid, fund_name
    return None, None


# ============================
# 1. Dado ISIN -> pegar XID
# ============================
def get_xid_from_isin(
    session: requests.Session,
    isin: str,
    currency: str | None = "USD"
) -> tuple[str, str, str | None]:
    isin = isin.strip().upper()
    url = f"{BASE_URL}/data/funds/tearsheet/historical"
    symbol = f"{isin}:{currency}" if currency else isin
    params = {"s": symbol}

    r = session.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    xid, fund_name = _extract_xid_and_name(soup, r.text)
    if xid:
        # Quando o tearsheet mostra moeda original diferente, tenta a classe detectada.
        detected_currency = _extract_price_currency(soup)
        if detected_currency and currency and detected_currency != currency:
            alt_params = {"s": f"{isin}:{detected_currency}"}
            rr = session.get(url, headers=HEADERS, params=alt_params, timeout=REQUEST_TIMEOUT)
            rr.raise_for_status()
            soup_alt = BeautifulSoup(rr.text, "html.parser")
            xid_alt, fund_name_alt = _extract_xid_and_name(soup_alt, rr.text)
            if xid_alt:
                return xid_alt, rr.url, fund_name_alt or fund_name
        return xid, r.url, fund_name

    title_tag = soup.find("h1")
    title_text = title_tag.get_text(" ", strip=True).lower() if title_tag else ""
    if title_text == "search":
        first_link = _extract_first_tearsheet_link(soup)
        if first_link:
            rr = session.get(first_link, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            rr.raise_for_status()
            soup2 = BeautifulSoup(rr.text, "html.parser")
            xid2, fund_name2 = _extract_xid_and_name(soup2, rr.text)
            if xid2:
                return xid2, rr.url, fund_name2
        raise RuntimeError("Página de busca do FT: ISIN/moeda não encontrado.")

    raise RuntimeError("Não foi possível extrair o XID do tearsheet do FT.")


# ==================================
# 2. XID -> historico (AJAX FT)
# ==================================
def get_historical_data(
    session: requests.Session,
    xid: str,
    start_date: str | None,
    end_date: str | None,
    referer: str
) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    url = f"{BASE_URL}/data/equities/ajax/get-historical-prices"

    params = {"symbol": xid}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    headers = {
        **HEADERS,
        "Accept": "application/json, text/plain, */*",
        "Origin": BASE_URL,
        "Referer": referer,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }

    r = session.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    if "application/json" not in content_type.lower():
        snippet = r.text[:500].replace("\n", " ").strip()
        raise RuntimeError(
            "Resposta não-JSON do FT "
            f"(status={r.status_code}, url={r.url}, ct={content_type}): {snippet}"
        )

    payload = r.json()
    html = payload.get("html")

    # Em alguns períodos (ex.: antes do início do fundo), o FT retorna payload sem linhas.
    if not html:
        return empty_df

    soup = BeautifulSoup(f"<table>{html}</table>", "html.parser")

    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue

        raw_date = tds[0].get_text(strip=True)

        row = {
            "Date": raw_date,
            "Open": tds[1].get_text(strip=True),
            "High": tds[2].get_text(strip=True),
            "Low": tds[3].get_text(strip=True),
            "Close": tds[4].get_text(strip=True),
            "Volume": tds[5].get_text(strip=True),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return empty_df

    # Normalizacao
    # A data vem concatenada em dois formatos (longo + curto). Pegamos o formato longo.
    df["Date"] = (
        df["Date"]
        .astype(str)
        .str.extract(r"([A-Za-z]+,\s+[A-Za-z]+\s+\d{2},\s+\d{4})", expand=False)
    )
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        format="%A, %B %d, %Y"
    )
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    return df.sort_values("Date")


# ==================================
# 3. Funcao FINAL (ISIN -> Excel)
# ==================================
def fetch_ft_history_by_isin(
    isin: str,
    start_date: str = "2000/01/01",
    end_date: str = datetime.today().strftime("%Y/%m/%d"),
    output_dir: str = "."
):
    print(f"Buscando XID para ISIN {isin}...")
    session = requests.Session()
    try:
        last_exc: Exception | None = None
        xid = ""
        referer = ""
        for currency in DEFAULT_CURRENCY_FALLBACKS:
            try:
                xid, referer, _ = get_xid_from_isin(session, isin, currency=currency)
                break
            except Exception as exc:
                last_exc = exc
                continue
        if not xid:
            raise RuntimeError(
                f"Não foi possível resolver XID para {isin} com moedas alternativas. "
                f"Último erro: {last_exc}"
            ) from None

        print(f"XID encontrado: {xid}")

        print("Baixando histórico...")
        df = get_historical_data(session, xid, start_date, end_date, referer)

        output_path = Path(output_dir) / f"{isin}_FT_Historical.xlsx"
        df.to_excel(output_path, index=False)

        print(f"Exportado para: {output_path}")
        return df
    finally:
        session.close()


# ============================
# Exemplo de uso
# ============================
if __name__ == "__main__":
    today = datetime.today()
    end = today.strftime("%Y/%m/%d")
    fetch_ft_history_by_isin("IE00BDT57T44", start_date="2000/01/01", end_date=end)


