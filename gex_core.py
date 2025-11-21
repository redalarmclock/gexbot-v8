from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


DERIBIT_BASE_URL = os.getenv("DERIBIT_BASE_URL", "https://www.deribit.com")
DERIBIT_CURRENCY = os.getenv("DERIBIT_CURRENCY", "BTC")

# Simple risk-free rate assumption (annualised)
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))

# Minimum Gamma (in USD) required to trigger a directional signal
# Prevents the bot from flipping bias on noise. 5 Million USD is a conservative floor for BTC.
GAMMA_SIGNAL_THRESHOLD = 5_000_000.0


@dataclass
class OptionPoint:
    instrument_name: str
    strike: float
    option_type: str  # "call" or "put"
    expiration_timestamp: int  # ms since epoch
    open_interest: float
    mark_iv: float  # in percent, e.g. 75 = 75%
    interest_rate: float
    underlying_price: float

    gamma: float          # per 1 option, per 1 underlying unit
    gamma_exposure: float # dealer-side dollar gamma exposure
    delta_exposure: float # dealer-side delta exposure in underlying units


@dataclass
class GexSnapshot:
    spot: float
    now_ms: int
    options: List[OptionPoint]

    # --- Aggregate Totals (Net Book) ---
    net_gamma: float             # Total dealer-side gamma (net, all expiries)
    net_delta: float             # Total dealer-side delta

    # --- Temporal Segmentation ---
    gamma_tactical: float        # Gamma expiring ≤ 24h (tactical)
    gamma_structure: float       # Gamma expiring > 24h (structural)
    
    # --- Levels ---
    magnet: Optional[float]
    flip: Optional[float]
    
    # --- Range Descriptions ---
    near_range: str
    strong_range: str
    
    # --- Environment & Labels (STRUCTURAL-BASED) ---
    gamma_env: str               # "short" / "long" / "flat" (based on structural gamma)
    gamma_env_label: str
    comment: str
    
    # --- High Level Signal ---
    expiry_outlook: str          # Conflict/confluence between Tactical & Structure
    
    # --- Walls & Trends ---
    top_walls: str
    band_walls: str
    band_bias: str
    band_bias_interpretation: str
    delta_trend: str
    delta_trend_long: str
    wide_range_flag: str
    interpretation: str
    bounce_map: str


_last_delta: Optional[float] = None
_delta_history: List[Tuple[int, float]] = []  # (timestamp_ms, net_delta)


# ---------- HTTP helpers ----------


def _get(path: str, params: Dict[str, Any]) -> Any:
    """Thin wrapper around requests.get for Deribit public HTTP API."""
    url = f"{DERIBIT_BASE_URL}/api/v2/{path.lstrip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "result" not in data:
            raise RuntimeError(f"Unexpected Deribit response for {path}: {data}")
        return data["result"]
    except Exception as e:
        # In production, you might want retries here
        raise RuntimeError(f"API Request Failed: {e}")


# ---------- Math helpers ----------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _black_scholes_greeks(
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    rate: float,
    option_type: str,
) -> Tuple[float, float]:
    """
    Return (delta, gamma) for a European option using Black-Scholes.
    """
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    pdf = _norm_pdf(d1)
    gamma = pdf / (spot * vol * sqrt_t)

    cdf_d1 = _norm_cdf(d1)
    if option_type == "call":
        delta = cdf_d1
    else:  # put
        delta = cdf_d1 - 1.0

    return delta, gamma


# ---------- Core data fetch ----------


def fetch_raw_gex_data() -> Dict[str, Any]:
    now_ms = int(time.time() * 1000)

    instruments = _get(
        "public/get_instruments",
        {"currency": DERIBIT_CURRENCY, "kind": "option", "expired": False},
    )

    summaries = _get(
        "public/get_book_summary_by_currency",
        {"currency": DERIBIT_CURRENCY, "kind": "option"},
    )

    index_result = _get("public/get_index", {"currency": DERIBIT_CURRENCY})
    spot = float(index_result.get(DERIBIT_CURRENCY, index_result.get("edp")))

    return {
        "now_ms": now_ms,
        "spot": spot,
        "instruments": instruments,
        "summaries": summaries,
    }


# ---------- Gamma aggregation ----------


def _build_option_points(raw: Dict[str, Any]) -> List[OptionPoint]:
    now_ms = raw["now_ms"]
    spot = raw["spot"]
    instruments = raw["instruments"]
    summaries = raw["summaries"]

    meta_by_name: Dict[str, Dict[str, Any]] = {
        inst["instrument_name"]: inst for inst in instruments
    }

    points: List[OptionPoint] = []

    for summ in summaries:
        name = summ["instrument_name"]
        meta = meta_by_name.get(name)
        if not meta:
            continue

        strike = float(meta["strike"])
        option_type = "call" if meta.get("option_type", "call") == "call" else "put"
        expiration_ts = int(meta["expiration_timestamp"])

        open_interest = float(summ.get("open_interest", 0.0))
        if open_interest <= 0:
            continue

        mark_iv_pct = float(summ.get("mark_iv", 0.0))
        interest_rate = float(summ.get("interest_rate", RISK_FREE_RATE))
        underlying_price = float(summ.get("underlying_price", spot))

        t_years = max((expiration_ts - now_ms) / (365.0 * 24.0 * 3600.0 * 1000.0), 1e-6)
        vol = max(mark_iv_pct / 100.0, 1e-4)

        delta, gamma = _black_scholes_greeks(
            spot=underlying_price,
            strike=strike,
            t_years=t_years,
            vol=vol,
            rate=interest_rate,
            option_type=option_type,
        )

        # Note: Deribit BTC options are coin-margined.
        # Dollar Gamma ≈ gamma * spot^2 * OI
        contract_size = 1.0
        
        # Dealer is short, customer is long
        customer_gamma_exposure = gamma * (underlying_price ** 2) * open_interest * contract_size
        dealer_gamma_exposure = -customer_gamma_exposure

        customer_delta_exposure = delta * open_interest * contract_size
        dealer_delta_exposure = -customer_delta_exposure

        points.append(
            OptionPoint(
                instrument_name=name,
                strike=strike,
                option_type=option_type,
                expiration_timestamp=expiration_ts,
                open_interest=open_interest,
                mark_iv=mark_iv_pct,
                interest_rate=interest_rate,
                underlying_price=underlying_price,
                gamma=gamma,
                gamma_exposure=dealer_gamma_exposure,
                delta_exposure=dealer_delta_exposure,
            )
        )

    return points


def _compute_magnet(gex_by_strike: Dict[float, float]) -> Optional[float]:
    if not gex_by_strike:
        return None
    total_abs = sum(abs(v) for v in gex_by_strike.values())
    if total_abs == 0:
        return None
    return sum(k * abs(v) for k, v in gex_by_strike.items()) / total_abs


def _compute_flip(gex_by_strike: Dict[float, float]) -> Optional[float]:
    if not gex_by_strike:
        return None
    items = sorted(gex_by_strike.items())
    cum = 0.0
    prev_strike = None
    prev_cum = None

    for strike, gex in items:
        prev_strike = strike if prev_strike is None else prev_strike
        prev_cum = cum
        cum += gex

        if prev_cum is not None and (prev_cum == 0 or (prev_cum < 0 < cum) or (prev_cum > 0 > cum)):
            if cum == prev_cum:
                return strike
            w = abs(prev_cum) / (abs(prev_cum) + abs(cum))
            return prev_strike * (1 - w) + strike * w
        prev_strike = strike
    return None


def _compute_ranges(spot: float, gex_by_strike: Dict[float, float], net_gamma: float) -> Tuple[str, str]:
    if not gex_by_strike:
        return "—/—", "—/—"
    suffix = "R" if net_gamma < 0 else "S" if net_gamma > 0 else ""
    items = sorted(gex_by_strike.items())
    max_abs = max(abs(v) for _, v in items)
    if max_abs == 0:
        return "—/—", "—/—"

    # --- 10% TWEAK APPLIED HERE ---
    # Lowered from 0.25 to 0.10 to detect intraday "Edge" walls
    near_threshold = 0.10 * max_abs 
    strong_threshold = 0.5 * max_abs

    def _pick(threshold: float) -> Tuple[Optional[float], Optional[float]]:
        below_candidates = [(spot - k, k) for k, v in items if k <= spot and abs(v) >= threshold]
        above_candidates = [(k - spot, k) for k, v in items if k >= spot and abs(v) >= threshold]
        below = min(below_candidates)[1] if below_candidates else None
        above = min(above_candidates)[1] if above_candidates else None
        return below, above