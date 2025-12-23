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
    gamma_tactical: float        # Gamma expiring <= 24h (tactical)
    gamma_structure: float       # Gamma expiring > 24h (structural)
    
    # --- Levels ---
    magnet: Optional[float]
    flip: Optional[float]
    
    # --- Range Descriptions ---
    near_range: str
    strong_range: str
    
    # --- Environment & Labels ---
    gamma_env: str               # "short" / "long" / "flat"
    gamma_env_label: str
    comment: str
    
    # --- High Level Signal (Kept for CSV, removed from Pretty) ---
    expiry_outlook: str
    
    # --- Walls & Trends ---
    top_walls: str
    band_walls: str
    band_bias: str
    band_bias_interpretation: str
    intraday_structure: str      
    delta_trend: str
    delta_trend_long: str
    wide_range_flag: str
    interpretation: str
    bounce_map: str

    # --- Metrics ---
    atm_iv: Optional[float]
    atm_iv_regime: str
    atm_iv_trend: str
    regime_stability_score: float
    regime_stability_label: str
    trade_score: float
    trade_zone: str
    trade_note: str


_last_delta: Optional[float] = None
_delta_history: List[Tuple[int, float]] = []  # (timestamp_ms, net_delta)
_atm_iv_history: List[Tuple[int, float]] = []  # (timestamp_ms, atm_iv)


# ---------- HTTP helpers ----------


def _get(path: str, params: Dict[str, Any]) -> Any:
    """Thin wrapper around requests.get for Deribit public HTTP API."""
    url = f"{DERIBIT_BASE_URL}/api/v2/{path.lstrip('/')}"
    headers = {
        "User-Agent": "GexBot/9.0 (Hard Structure)"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "result" not in data:
            raise RuntimeError(f"Unexpected Deribit response for {path}: {data}")
        return data["result"]
    except Exception as e:
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
        {"currency": DERIBIT_CURRENCY, "kind": "option", "expired": "false"},
    )

    summaries = _get(
        "public/get_book_summary_by_currency",
        {"currency": DERIBIT_CURRENCY, "kind": "option"},
    )

    index_name = f"{DERIBIT_CURRENCY.lower()}_usd"
    index_result = _get("public/get_index_price", {"index_name": index_name})
    
    spot = float(index_result.get("index_price", index_result.get("price", 0.0)))

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

        contract_size = 1.0
        
        customer_gamma_exposure = gamma * (underlying_price ** 2) * open_interest * contract_size
        dealer_gamma_exposure = -customer_gamma_exposure
        dealer_delta_exposure = -1.0 * (delta * open_interest * contract_size)

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

    near_threshold = 0.10 * max_abs 
    strong_threshold = 0.5 * max_abs

    def _pick(threshold: float) -> Tuple[Optional[float], Optional[float]]:
        below_candidates = [(spot - k, k) for k, v in items if k <= spot and abs(v) >= threshold]
        above_candidates = [(k - spot, k) for k, v in items if k >= spot and abs(v) >= threshold]
        below = min(below_candidates)[1] if below_candidates else None
        above = min(above_candidates)[1] if above_candidates else None
        return below, above

    def _fmt_pair(below: Optional[float], above: Optional[float]) -> str:
        def fmt(x: Optional[float]) -> str:
            if x is None:
                return "—"
            rounded = int(round(x / 1000.0) * 1000)
            return f"{rounded}{suffix}" if suffix else f"{rounded}"
        return f"{fmt(below)}/{fmt(above)}"

    return _fmt_pair(*_pick(near_threshold)), _fmt_pair(*_pick(strong_threshold))


def _classify_environment(net_gamma: float) -> Tuple[str, str, str]:
    abs_g = abs(net_gamma)
    TH_FLAT = 1_000_000_000.0
    TH_MILD = 12_000_000_000.0
    TH_HIGH = 30_000_000_000.0

    if abs_g < TH_FLAT:
        return "flat", "⚪ Flat γ", "Gamma neutral."

    if net_gamma < 0:
        if abs_g < TH_MILD:
            label = "🔴 γ Short (Mild)"
            comment = "Mild neg γ."
        elif abs_g < TH_HIGH:
            label = "🔴 γ Short (Elevated)"
            comment = "Elevated neg γ."
        else:
            label = "🔴 γ Short (Extreme)"
            comment = "Extreme neg γ."
        return "short", label, comment
    else:
        if abs_g < TH_MILD:
            label = "🟢 γ Long (Mild)"
            comment = "Mild pos γ."
        elif abs_g < TH_HIGH:
            label = "🟢 γ Long (Elevated)"
            comment = "Elevated pos γ."
        else:
            label = "🟢 γ Long (Extreme)"
            comment = "Extreme pos γ."
        return "long", label, comment


def _top_gamma_walls(gex_by_strike: Dict[float, float], top_n: int = 5) -> str:
    if not gex_by_strike:
        return "—"
    sorted_walls = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    parts: List[str] = []
    for strike, gex in sorted_walls:
        gex_m = gex / 1_000_000
        direction = "R" if gex < 0 else "S"
        # Allows negative M for correct V9 parsing
        parts.append(f"{int(strike):,} ({gex_m:+.2f}M {direction})")
    return " | ".join(parts)


def _compute_band_walls(strong_range: str, gex_by_strike: Dict[float, float]) -> str:
    if not gex_by_strike or strong_range == "—/—":
        return "—"
    try:
        low_str, high_str = strong_range.split("/")
        if "—" in low_str or "—" in high_str:
            return "—"
        low_v = int(low_str[:-1] if low_str.endswith(("R", "S")) else low_str)
        high_v = int(high_str[:-1] if high_str.endswith(("R", "S")) else high_str)
        if low_v > high_v:
            low_v, high_v = high_v, low_v
    except Exception:
        return "—"

    band_items = [(k, v) for k, v in gex_by_strike.items() if low_v <= k <= high_v]
    threshold = 500_000_000.0
    band_items = [(k, v) for k, v in band_items if abs(v) >= threshold]
    if not band_items:
        return "—"
    
    band_items = sorted(band_items, key=lambda x: abs(x[1]), reverse=True)[:4]
    band_items = sorted(band_items, key=lambda x: x[0])

    parts: List[str] = []
    for strike, gex in band_items:
        strike_k = int(round(strike / 1000.0))
        direction = "R" if gex < 0 else "S"
        gex_m = abs(gex) / 1_000_000.0
        parts.append(f"{strike_k}k{direction} ({gex_m:.1f}M)")
    return " | ".join(parts)


def _compute_intraday_structure(spot: float, gex_by_strike: Dict[float, float]) -> str:
    # Logic kept for CSV integrity, but not displayed
    range_pct = 0.015 
    low_bound = spot * (1 - range_pct)
    high_bound = spot * (1 + range_pct)
    
    local_items = [
        (k, v) for k, v in gex_by_strike.items() 
        if low_bound <= k <= high_bound
    ]
    threshold = 50_000_000.0
    significant_items = [(k, v) for k, v in local_items if abs(v) >= threshold]
    significant_items.sort(key=lambda x: x[0])
    if not significant_items:
        return "— (Vacuum)"
    parts = []
    for strike, gex in significant_items:
        k_val = strike / 1000.0
        k_str = f"{int(k_val)}k" if k_val.is_integer() else f"{k_val:.1f}k"
        direction = "R" if gex < 0 else "S"
        gex_m = abs(gex) / 1_000_000.0
        parts.append(f"{k_str}{direction} ({gex_m:.0f}M)")
    return " → ".join(parts)


def _compute_band_bias(strong_range: str, gex_by_strike: Dict[float, float]) -> str:
    # Logic kept for CSV
    if not gex_by_strike or strong_range == "—/—": return "—"
    try:
        low_str, high_str = strong_range.split("/")
        if "—" in low_str or "—" in high_str: return "—"
        low_v = int(low_str[:-1] if low_str.endswith(("R", "S")) else low_str)
        high_v = int(high_str[:-1] if high_str.endswith(("R", "S")) else high_str)
        if low_v > high_v: low_v, high_v = high_v, low_v
    except: return "—"

    mid = (low_v + high_v) / 2.0
    lower_sum = sum(abs(v) for k, v in gex_by_strike.items() if low_v <= k <= mid)
    upper_sum = sum(abs(v) for k, v in gex_by_strike.items() if mid < k <= high_v)
    total = lower_sum + upper_sum
    if total == 0: return "balanced"
    lower_pct = lower_sum / total
    if lower_pct >= 0.60: return f"lower-heavy ({lower_pct*100:.0f}%)"
    elif lower_pct <= 0.40: return f"upper-heavy ({(1-lower_pct)*100:.0f}%)"
    else: return "balanced (≈50/50)"


def _describe_delta_trend(diff: float, ref: float) -> str:
    if ref == 0: ref = 1.0
    rel = abs(diff) / abs(ref)
    if abs(diff) < 1000 and rel < 0.03: return "Stable"
    elif diff > 0: return "Rising"
    else: return "Falling"


def _interpret_band_bias(band_bias: str) -> str:
    # Logic kept for CSV
    if not band_bias or band_bias == "—": return "Neutral"
    if "lower-heavy" in band_bias: return "Support focus"
    if "upper-heavy" in band_bias: return "Resistance focus"
    return "Balanced"


def _compute_bounce_map(spot: float, strong_range: str, gamma_env: str, struct_gamma: float, net_delta: float) -> str:
    # Logic kept for CSV
    return "—"


def _build_interpretation(spot: float, gamma_env: str, near_range: str, top_walls: str, wide_range_flag: str) -> str:
    # Logic kept for CSV
    return "—"


def _compute_atm_iv(options: List[OptionPoint], spot: float, now_ms: int) -> Tuple[Optional[float], str, str]:
    band_low = spot * 0.95
    band_high = spot * 1.05
    numer = 0.0
    denom = 0.0

    for p in options:
        if band_low <= p.strike <= band_high and p.mark_iv > 0:
            numer += p.mark_iv * p.open_interest
            denom += p.open_interest

    if denom <= 0:
        atm_iv = None
    else:
        atm_iv = numer / denom

    if atm_iv is None:
        return None, "—", "—"

    if atm_iv < 40.0: regime = "Low"
    elif atm_iv <= 70.0: regime = "Normal"
    else: regime = "High"

    _atm_iv_history.append((now_ms, atm_iv))
    cutoff_hist = now_ms - 2 * 3600_000
    while _atm_iv_history and _atm_iv_history[0][0] < cutoff_hist:
        _atm_iv_history.pop(0)

    target = now_ms - 3600_000 
    past_candidates = [x for x in _atm_iv_history if x[0] <= target]
    if not past_candidates:
        trend = "Flat"
    else:
        ref_t, ref_iv = max(past_candidates, key=lambda x: x[0])
        diff = atm_iv - ref_iv
        if abs(diff) < 1.0: trend = "Flat"
        elif diff > 0: trend = "Rising"
        else: trend = "Falling"

    return atm_iv, regime, trend


def _compute_regime_stability(now_ms: int, gamma_structure: float) -> Tuple[float, str]:
    abs_struct = abs(gamma_structure)
    base = min(abs_struct / 50_000_000_000.0, 1.0)

    if len(_delta_history) >= 2:
        deltas = [d for (_, d) in _delta_history]
        mean = sum(deltas) / len(deltas)
        var = sum((d - mean) ** 2 for d in deltas) / len(deltas)
        delta_std = var ** 0.5
    else:
        delta_std = 0.0

    delta_vol_norm = min(delta_std / 50_000.0, 1.0)
    stability = 100.0 * (0.5 * base + 0.25 * 1.0 + 0.25 * (1.0 - delta_vol_norm))
    stability = max(0.0, min(100.0, stability))

    if stability >= 75.0: label = "🟢 Stable"
    elif stability >= 50.0: label = "🟡 Mixed"
    elif stability >= 25.0: label = "🔴 Unstable"
    else: label = "🔥 Chaotic"

    return stability, label


def _compute_tradeability(spot: float, strong_range: str, gamma_env: str, regime_stability_label: str, band_bias: str) -> Tuple[float, str, str]:
    # Logic kept for CSV
    return 5.0, "Standard", "Standard"


def compute_gex_snapshot(raw: Dict[str, Any]) -> GexSnapshot:
    global _last_delta, _delta_history

    spot = float(raw["spot"])
    now_ms = int(raw["now_ms"])
    options = _build_option_points(raw)

    if not options:
        raise RuntimeError("No active BTC options")

    net_gamma = sum(p.gamma_exposure for p in options)
    net_delta = sum(p.delta_exposure for p in options)

    ONE_DAY_MS = 24 * 60 * 60 * 1000
    cutoff_ts = now_ms + ONE_DAY_MS

    gamma_tactical = sum(p.gamma_exposure for p in options if p.expiration_timestamp <= cutoff_ts)
    gamma_structure = sum(p.gamma_exposure for p in options if p.expiration_timestamp > cutoff_ts)

    if _last_delta is None:
        delta_trend_short = "First"
    else:
        diff_short = net_delta - _last_delta
        delta_trend_short = _describe_delta_trend(diff_short, _last_delta)
    _last_delta = net_delta

    _delta_history.append((now_ms, net_delta))
    cutoff_hist = now_ms - 2 * 3600_000
    _delta_history = [(t, d) for (t, d) in _delta_history if t >= cutoff_hist]

    target = now_ms - 3600_000
    past_candidates = [(t, d) for (t, d) in _delta_history if t <= target]
    if not past_candidates:
        delta_trend_long = "Init"
    else:
        ref_t, ref_delta = max(past_candidates, key=lambda x: x[0])
        diff_long = net_delta - ref_delta
        delta_trend_long = _describe_delta_trend(diff_long, ref_delta)

    gex_by_strike: Dict[float, float] = {}
    for p in options:
        gex_by_strike[p.strike] = gex_by_strike.get(p.strike, 0.0) + p.gamma_exposure

    magnet = _compute_magnet(gex_by_strike)
    flip = _compute_flip(gex_by_strike)
    near_range, strong_range = _compute_ranges(spot, gex_by_strike, net_gamma)
    top_walls = _top_gamma_walls(gex_by_strike)
    band_walls = _compute_band_walls(strong_range, gex_by_strike)
    band_bias = _compute_band_bias(strong_range, gex_by_strike)
    band_bias_interpretation = _interpret_band_bias(band_bias)
    intraday_structure = _compute_intraday_structure(spot, gex_by_strike)

    wide_range_flag = ""
    try:
        low_str, high_str = near_range.split("/")
        if "—" not in low_str:
            low_v = int(low_str[:-1] if low_str.endswith(("R", "S")) else low_str)
            high_v = int(high_str[:-1] if high_str.endswith(("R", "S")) else high_str)
            if abs(high_v - low_v) >= 4_000:
                wide_range_flag = "Wide Zone"
    except: pass

    gamma_env, gamma_env_label, comment = _classify_environment(gamma_structure)

    atm_iv, atm_iv_regime, atm_iv_trend = _compute_atm_iv(options, spot, now_ms)
    regime_stability_score, regime_stability_label = _compute_regime_stability(now_ms, gamma_structure)
    trade_score, trade_zone, trade_note = _compute_tradeability(spot, strong_range, gamma_env, regime_stability_label, band_bias)

    return GexSnapshot(
        spot=spot,
        now_ms=now_ms,
        options=options,
        net_gamma=net_gamma,
        net_delta=net_delta,
        gamma_tactical=gamma_tactical,
        gamma_structure=gamma_structure,
        expiry_outlook="Standard",
        magnet=magnet,
        flip=flip,
        near_range=near_range,
        strong_range=strong_range,
        gamma_env=gamma_env,
        gamma_env_label=gamma_env_label,
        comment=comment,
        top_walls=top_walls,
        band_walls=band_walls,
        band_bias=band_bias,
        band_bias_interpretation=band_bias_interpretation,
        intraday_structure=intraday_structure,
        delta_trend=delta_trend_short,
        delta_trend_long=delta_trend_long,
        wide_range_flag=wide_range_flag,
        interpretation="—",
        bounce_map="—",
        atm_iv=atm_iv,
        atm_iv_regime=atm_iv_regime,
        atm_iv_trend=atm_iv_trend,
        regime_stability_score=regime_stability_score,
        regime_stability_label=regime_stability_label,
        trade_score=trade_score,
        trade_zone=trade_zone,
        trade_note=trade_note,
    )


# ---------- Formatting helpers ----------


def _fmt_price(p: Optional[float]) -> str:
    if p is None: return "—"
    return f"{p:,.0f}"


def format_ultra(snapshot: GexSnapshot) -> str:
    return format_pretty(snapshot)


def format_pretty(snapshot: GexSnapshot) -> str:
    """
    Professional Grade Output: Hard Structure Only.
    Cleanest version. No Voodoo.
    """
    spot_s = _fmt_price(snapshot.spot)
    flip_s = (
        _fmt_price(snapshot.flip)
        if snapshot.flip and abs(snapshot.flip - snapshot.spot) / snapshot.spot <= 0.25
        else "—"
    )
    mag_s = _fmt_price(snapshot.magnet)
    delta_s = f"{snapshot.net_delta:,.0f}"
    
    # Billions for High Level
    gamma_b = snapshot.net_gamma / 1_000_000_000.0
    gt_b = snapshot.gamma_tactical / 1_000_000_000.0
    gs_b = snapshot.gamma_structure / 1_000_000_000.0

    lines: List[str] = [
        f"📊 {DERIBIT_CURRENCY} GEX Update | Spot {spot_s}",
        "",
        f"• Env: {snapshot.gamma_env_label}",
    ]

    if snapshot.atm_iv is not None:
        lines.append(f"• IV: {snapshot.atm_iv:.0f}% ({snapshot.atm_iv_trend})")
    else:
        lines.append("• IV: —")

    lines.extend([
        "",
        f"• Struct Γ: {gs_b:+.1f}B",
        f"• Tact Γ:   {gt_b:+.1f}B",
        "",
        f"• Magnet: {mag_s}  | Flip: {flip_s}",
        f"• Net Γ: {gamma_b:+.1f}B | Δ: {delta_s}",
        "",
        f"• Near: {snapshot.near_range}",
        f"• Strong: {snapshot.strong_range}",
    ])

    if snapshot.wide_range_flag:
        lines.append(f"• Note: {snapshot.wide_range_flag}")

    lines.append("")
    lines.append(f"• Walls: {snapshot.top_walls}")
    
    if snapshot.band_walls and snapshot.band_walls != "—":
        lines.append(f"• Band: {snapshot.band_walls}")

    lines.append("")
    lines.append(f"• Stability: {snapshot.regime_stability_label} ({snapshot.regime_stability_score:.0f})")

    return "\n".join(lines)


def snapshot_to_ultra_row(snapshot: GexSnapshot) -> Dict[str, Any]:
    return {
        "timestamp": snapshot.now_ms,
        "spot": snapshot.spot,
        "gamma_env": snapshot.gamma_env,
        "gamma_env_label": snapshot.gamma_env_label,
        "expiry_outlook": snapshot.expiry_outlook,
        "gamma_tactical": snapshot.gamma_tactical,
        "gamma_structure": snapshot.gamma_structure,
        "flip": snapshot.flip,
        "magnet": snapshot.magnet,
        "delta": snapshot.net_delta,
        "net_gamma": snapshot.net_gamma,
        "near_range": snapshot.near_range,
        "strong_range": snapshot.strong_range,
        "comment": snapshot.comment,
        "top_walls": snapshot.top_walls,
        "band_walls": snapshot.band_walls,
        "band_bias": snapshot.band_bias,
        "band_bias_interpretation": snapshot.band_bias_interpretation,
        "intraday_structure": snapshot.intraday_structure,
        "delta_trend_short": snapshot.delta_trend,
        "delta_trend_long": snapshot.delta_trend_long,
        "wide_range_flag": snapshot.wide_range_flag,
        "interpretation": snapshot.interpretation,
        "bounce_map": snapshot.bounce_map,
        "atm_iv": snapshot.atm_iv,
        "atm_iv_regime": snapshot.atm_iv_regime,
        "atm_iv_trend": snapshot.atm_iv_trend,
        "regime_stability_score": snapshot.regime_stability_score,
        "regime_stability_label": snapshot.regime_stability_label,
        "trade_score": snapshot.trade_score,
        "trade_zone": snapshot.trade_zone,
        "trade_note": snapshot.trade_note,
    }


def snapshot_to_pretty_row(snapshot: GexSnapshot) -> Dict[str, Any]:
    return snapshot_to_ultra_row(snapshot)