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
    gamma_exposure: float # dealer-side gamma exposure in underlying units
    delta_exposure: float # dealer-side delta exposure in underlying units


@dataclass
class GexSnapshot:
    spot: float
    now_ms: int
    options: List[OptionPoint]

    net_gamma: float             # dealer-side net gamma exposure (underlying units)
    net_delta: float             # dealer-side net delta exposure (underlying units)
    magnet: Optional[float]      # gamma-weighted "magnet" level
    flip: Optional[float]        # level where cumulative gamma changes sign
    near_range: str              # e.g. "95000R/96000R"
    strong_range: str            # e.g. "94000R/99000R"
    gamma_env: str               # "short" / "long" / "flat"
    gamma_env_label: str         # e.g. "🔴 γ Short"
    comment: str                 # short environment comment


# ---------- HTTP helpers ----------


def _get(path: str, params: Dict[str, Any]) -> Any:
    """Thin wrapper around requests.get for Deribit public HTTP API."""
    url = f"{DERIBIT_BASE_URL}/api/v2/{path.lstrip('/')}"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "result" not in data:
        raise RuntimeError(f"Unexpected Deribit response for {path}: {data}")
    return data["result"]


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
    Gamma is the same for calls and puts.
    """
    # Guard rails for crazy inputs
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
    """
    Fetch live BTC options universe + spot from Deribit public HTTP API.

    Returns a dict with:
      - now_ms
      - spot
      - instruments: metadata from public/get_instruments
      - summaries: book summaries from public/get_book_summary_by_currency
    """
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
    # get_index returns {"BTC": price, "edp": price}
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

    # Map instrument_name -> meta
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
        expiration_ts = int(meta["expiration_timestamp"])  # ms

        open_interest = float(summ.get("open_interest", 0.0))
        if open_interest <= 0:
            continue  # ignore dead series

        mark_iv_pct = float(summ.get("mark_iv", 0.0))
        interest_rate = float(summ.get("interest_rate", RISK_FREE_RATE))
        underlying_price = float(
            summ.get("underlying_price", spot)
        )  # fall back to index

        # Time to expiry in years
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

        # One contract on Deribit BTC options = 1 BTC notional
        contract_size = 1.0

        # Customer is assumed long options; dealers short.
        # So dealer-side exposure is negative of customer greeks.
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


def compute_gex_snapshot(raw: Dict[str, Any]) -> GexSnapshot:
    """Turn raw Deribit data into a single GEX snapshot."""

    spot = float(raw["spot"])
    now_ms = int(raw["now_ms"])
    options = _build_option_points(raw)

    if not options:
        raise RuntimeError("No active BTC options with open interest returned from Deribit")

    # Aggregate gamma / delta
    net_gamma = sum(p.gamma_exposure for p in options)
    net_delta = sum(p.delta_exposure for p in options)

    # Aggregate by strike for magnets / walls
    gex_by_strike: Dict[float, float] = {}
    for p in options:
        gex_by_strike[p.strike] = gex_by_strike.get(p.strike, 0.0) + p.gamma_exposure

    magnet = _compute_magnet(gex_by_strike)
    flip = _compute_flip(gex_by_strike)

    near_range, strong_range = _compute_ranges(spot, gex_by_strike, net_gamma)

    gamma_env, gamma_env_label, comment = _classify_environment(net_gamma)

    return GexSnapshot(
        spot=spot,
        now_ms=now_ms,
        options=options,
        net_gamma=net_gamma,
        net_delta=net_delta,
        magnet=magnet,
        flip=flip,
        near_range=near_range,
        strong_range=strong_range,
        gamma_env=gamma_env,
        gamma_env_label=gamma_env_label,
        comment=comment,
    )


def _compute_magnet(gex_by_strike: Dict[float, float]) -> Optional[float]:
    if not gex_by_strike:
        return None
    total_abs = sum(abs(v) for v in gex_by_strike.values())
    if total_abs == 0:
        return None
    return sum(k * abs(v) for k, v in gex_by_strike.items()) / total_abs


def _compute_flip(gex_by_strike: Dict[float, float]) -> Optional[float]:
    """Approximate price where cumulative dealer gamma changes sign."""
    if not gex_by_strike:
        return None

    items = sorted(gex_by_strike.items())  # (strike, gex)
    cum = 0.0
    prev_strike = None
    prev_cum = None

    for strike, gex in items:
        prev_strike = strike if prev_strike is None else prev_strike
        prev_cum = cum
        cum += gex

        if prev_cum is not None and (prev_cum == 0 or (prev_cum < 0 < cum) or (prev_cum > 0 > cum)):
            # Linear interpolation between previous and current strike
            if cum == prev_cum:
                return strike
            w = abs(prev_cum) / (abs(prev_cum) + abs(cum))
            return prev_strike * (1 - w) + strike * w

        prev_strike = strike

    return None


def _compute_ranges(
    spot: float,
    gex_by_strike: Dict[float, float],
    net_gamma: float,
) -> Tuple[str, str]:
    if not gex_by_strike:
        return "—/—", "—/—"

    # Which suffix to use – R for neg gamma (whipsaw), S for pos gamma (supportive)
    suffix = "R" if net_gamma < 0 else "S" if net_gamma > 0 else ""

    items = sorted(gex_by_strike.items())  # (strike, gex)
    max_abs = max(abs(v) for _, v in items)
    if max_abs == 0:
        return "—/—", "—/—"

    near_threshold = 0.25 * max_abs
    strong_threshold = 0.5 * max_abs

    def _pick(threshold: float) -> Tuple[Optional[float], Optional[float]]:
        below_candidates = [
            (spot - k, k) for k, v in items if k <= spot and abs(v) >= threshold
        ]
        above_candidates = [
            (k - spot, k) for k, v in items if k >= spot and abs(v) >= threshold
        ]
        below = min(below_candidates)[1] if below_candidates else None
        above = min(above_candidates)[1] if above_candidates else None
        return below, above

    near_below, near_above = _pick(near_threshold)
    strong_below, strong_above = _pick(strong_threshold)

    def _fmt_pair(below: Optional[float], above: Optional[float]) -> str:
        def fmt(x: Optional[float]) -> str:
            if x is None:
                return "—"
            rounded = int(round(x / 1000.0) * 1000)  # round to nearest 1k
            return f"{rounded}{suffix}" if suffix else f"{rounded}"

        return f"{fmt(below)}/{fmt(above)}"

    return _fmt_pair(near_below, near_above), _fmt_pair(strong_below, strong_above)


def _classify_environment(net_gamma: float) -> Tuple[str, str, str]:
    abs_g = abs(net_gamma)
    # Thresholds in BTC-gamma units are somewhat arbitrary; we only care about sign & "flat".
    flat_threshold = 1e2  # treat very small exposures as flat

    if abs_g < flat_threshold:
        return "flat", "⚪ Flat γ", "Gamma neutral zone — expect chop / transition"

    if net_gamma < 0:
        return (
            "short",
            "🔴 γ Short",
            "Neg γ – expansion / whipsaw risk",
        )
    else:
        return (
            "long",
            "🟢 γ Long",
            "Pos γ – dealer dampening, mean-revert bias",
        )


# ---------- Formatting helpers for bot & history ----------


def _fmt_price(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{p:,.0f}"


def format_ultra(snapshot: GexSnapshot) -> str:
    """One-liner style print for the fast 'ultra' feed."""
    spot_s = _fmt_price(snapshot.spot)
    flip_s = _fmt_price(snapshot.flip)
    mag_s = _fmt_price(snapshot.magnet)

    delta_s = f"{snapshot.net_delta:,.0f}"
    gamma_m = snapshot.net_gamma / 1_000_000.0
    gamma_s = f"{gamma_m:.2f}M"

    line1 = (
        f"{DERIBIT_CURRENCY} {spot_s} | {snapshot.gamma_env_label} <~{flip_s} | "
        f"🎯 Mag {mag_s} | Δ {delta_s} | Γ {gamma_s}"
    )
    line2 = f"📊 Near {snapshot.near_range} | Strong {snapshot.strong_range}"
    return f"{line1}\n{line2}"


def format_pretty(snapshot: GexSnapshot) -> str:
    """More verbose block for the 'pretty' 15m feed."""
    spot_s = _fmt_price(snapshot.spot)
    flip_s = _fmt_price(snapshot.flip)
    mag_s = _fmt_price(snapshot.magnet)

    delta_s = f"{snapshot.net_delta:,.0f}"
    gamma_m = snapshot.net_gamma / 1_000_000.0
    gamma_s = f"{gamma_m:.2f}M"

    lines = [
        f"📊 {DERIBIT_CURRENCY} GEX Update | Spot {spot_s}",
        "",
        f"• Environment: {snapshot.gamma_env_label} – {snapshot.comment}",
        f"• Magnet: {mag_s}  | Flip: {flip_s}",
        f"• Net Γ: {gamma_s} | Δ: {delta_s}",
        f"• Near: {snapshot.near_range} | Strong: {snapshot.strong_range}",
    ]
    return "\n".join(lines)


def snapshot_to_ultra_row(snapshot: GexSnapshot) -> Dict[str, Any]:
    """Row schema for the 5-minute 'ultra' history log."""
    return {
        "timestamp": snapshot.now_ms,
        "spot": snapshot.spot,
        "gamma_env": snapshot.gamma_env,
        "gamma_env_label": snapshot.gamma_env_label,
        "flip": snapshot.flip,
        "magnet": snapshot.magnet,
        "delta": snapshot.net_delta,
        "net_gamma": snapshot.net_gamma,
        "near_range": snapshot.near_range,
        "strong_range": snapshot.strong_range,
        "comment": snapshot.comment,
    }


def snapshot_to_pretty_row(snapshot: GexSnapshot) -> Dict[str, Any]:
    """Row schema for the 15-minute 'pretty' history log."""
    # Same fields for now; you can diverge later if needed.
    return snapshot_to_ultra_row(snapshot)
