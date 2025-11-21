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
        # Dollar Gamma approx gamma * spot^2 * OI
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
    """
    Classifies the environment based on MAGNITUDE and SIGN.
    Using Billions thresholds for current high-vol regime.
    """
    abs_g = abs(net_gamma)

    # Thresholds in USD gamma (billions).
    TH_FLAT = 1_000_000_000.0      # 1B
    TH_MILD = 12_000_000_000.0     # 12B
    TH_HIGH = 30_000_000_000.0     # 30B

    if abs_g < TH_FLAT:
        return "flat", "⚪ Flat γ", "Gamma neutral — expect chop."

    if net_gamma < 0:
        if abs_g < TH_MILD:
            label = "🔴 γ Short (Mild)"
            comment = "Mild neg γ — some whipsaw risk."
        elif abs_g < TH_HIGH:
            label = "🔴 γ Short (Elevated)"
            comment = "Elevated neg γ — expansion and trap risk."
        else:
            label = "🔴 γ Short (Extreme)"
            comment = "Extreme neg γ — large moves, high whipsaw risk."
        return "short", label, comment
    else:
        if abs_g < TH_MILD:
            label = "🟢 γ Long (Mild)"
            comment = "Mild pos γ — weak mean-revert bias."
        elif abs_g < TH_HIGH:
            label = "🟢 γ Long (Elevated)"
            comment = "Elevated pos γ — strong dampening effect."
        else:
            label = "🟢 γ Long (Extreme)"
            comment = "Extreme pos γ — strong lean to mean-reversion."
        return "long", label, comment


def _top_gamma_walls(gex_by_strike: Dict[float, float], top_n: int = 5) -> str:
    if not gex_by_strike:
        return "—"
    sorted_walls = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    parts: List[str] = []
    for strike, gex in sorted_walls:
        gex_m = gex / 1_000_000
        direction = "R" if gex < 0 else "S"
        parts.append(f"{int(strike):,} ({gex_m:.2f}M {direction})")
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


def _compute_band_bias(strong_range: str, gex_by_strike: Dict[float, float]) -> str:
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

    mid = (low_v + high_v) / 2.0
    lower_sum = sum(abs(v) for k, v in gex_by_strike.items() if low_v <= k <= mid)
    upper_sum = sum(abs(v) for k, v in gex_by_strike.items() if mid < k <= high_v)
    total = lower_sum + upper_sum

    if total == 0:
        return "balanced (no γ)"
    lower_pct = lower_sum / total

    if lower_pct >= 0.60:
        return f"lower-heavy ({lower_pct*100:.0f}% below midpoint)"
    elif lower_pct <= 0.40:
        return f"upper-heavy ({(1-lower_pct)*100:.0f}% above midpoint)"
    else:
        return "balanced (≈50/50)"


def _describe_delta_trend(diff: float, ref: float) -> str:
    if ref == 0:
        ref = 1.0
    rel = abs(diff) / abs(ref)
    if abs(diff) < 1000 and rel < 0.03:
        return "Delta stable — hedging unchanged."
    elif diff > 0:
        return "Delta rising — stronger hedging pressure."
    else:
        return "Delta falling — hedging pressure easing."


def _interpret_band_bias(band_bias: str) -> str:
    if not band_bias or band_bias == "—":
        return "No clear bias — interior gamma evenly distributed."
    txt = band_bias.lower()
    if "lower-heavy" in txt:
        return "Lower-heavy — stronger dip support; upward microstructure lean."
    if "upper-heavy" in txt:
        return "Upper-heavy — drops accelerate; weaker dip support."
    if "balanced" in txt:
        return "Balanced — no directional lean; pure wall-to-wall chop."
    return "Band bias unclear."


def _fmt_k_level(price: float) -> str:
    if price <= 0 or math.isnan(price):
        return "—"
    rounded = round(price / 100.0) * 100
    k = rounded / 1000.0
    return f"{k:.0f}k" if abs(rounded % 1000) < 1e-6 else f"{k:.1f}k"


def _compute_bounce_map(
    spot: float,
    strong_range: str,
    gamma_env: str,
    struct_gamma: float,
    net_delta: float,
) -> str:
    if gamma_env != "short" or not strong_range or strong_range == "—/—":
        return "Map only shown in short-γ with defined strong band."
    
    try:
        low_str, high_str = strong_range.split("/")
        if "—" in low_str or "—" in high_str:
            return "Map unavailable."
        low_v = int(low_str[:-1] if low_str.endswith(("R", "S")) else low_str)
        high_v = int(high_str[:-1] if high_str.endswith(("R", "S")) else high_str)
        if low_v > high_v:
            low_v, high_v = high_v, low_v
    except Exception:
        return "Map unavailable."

    if spot < low_v - 2_000 or spot > high_v + 2_000:
        return "Map focuses on reactions from the lower band; spot currently far from that zone."

    abs_gamma = abs(struct_gamma)
    abs_delta = abs(net_delta)
    extreme_gamma = abs_gamma >= 25_000_000
    strong_delta = abs_delta >= 40_000

    lvl_tag = low_v
    lvl_rebound = low_v * 1.006
    lvl_extend = low_v * 1.014
    lvl_mid = (low_v + high_v) / 2.0

    tag_prob = "very likely"
    if extreme_gamma and strong_delta:
        rebound_prob, extend_prob, mid_prob = "high", "medium", "low"
    else:
        rebound_prob, extend_prob, mid_prob = "medium", "low", "low"

    stage = 0
    if spot > lvl_tag:
        stage = 1
    if spot > lvl_rebound:
        stage = 2
    if spot > lvl_extend:
        stage = 3
    if spot > lvl_mid:
        stage = 4

    if stage == 0:
        tag_desc = f"{_fmt_k_level(lvl_tag)} tag zone — {tag_prob}"
    else:
        tag_desc = f"{_fmt_k_level(lvl_tag)} tag zone — tested"

    if stage <= 1:
        rebound_desc = f"{_fmt_k_level(lvl_rebound)} rebound zone — {rebound_prob}"
    elif stage == 2:
        rebound_desc = f"{_fmt_k_level(lvl_rebound)} rebound zone — engaged"
    else:
        rebound_desc = f"{_fmt_k_level(lvl_rebound)} rebound zone — likely completed"

    if stage <= 2:
        extend_desc = f"{_fmt_k_level(lvl_extend)} extension zone — {extend_prob}"
    elif stage == 3:
        extend_desc = f"{_fmt_k_level(lvl_extend)} extension zone — engaged"
    else:
        extend_desc = f"{_fmt_k_level(lvl_extend)} extension zone — likely completed"

    mid_desc = (
        f"{_fmt_k_level(lvl_mid)} stretch zone — in play"
        if stage >= 4
        else f"{_fmt_k_level(lvl_mid)} stretch zone — {mid_prob}"
    )

    return f"{tag_desc} → {rebound_desc} → {extend_desc} → {mid_desc}"


def _build_interpretation(
    spot: float,
    gamma_env: str,
    near_range: str,
    top_walls: str,
    wide_range_flag: str,
) -> str:
    if not top_walls or top_walls == "—":
        return "No dominant walls — GEX structure light; behaviour driven more by flows than walls."

    primary = top_walls.split(" | ")[0]
    strike_label = "key level"
    direction = ""
    try:
        strike_str, _ = primary.split(" ", 1)
        strike_val = int(strike_str.replace(",", ""))
        strike_label = f"{strike_val // 1000}k"
        if " R)" in primary:
            direction = "R"
        elif " S)" in primary:
            direction = "S"
    except Exception:
        pass

    behaviour = "choppy, transition-type price action."
    if gamma_env == "short":
        behaviour = (
            "wide negative-gamma chop with sharp moves."
            if wide_range_flag
            else "negative-gamma chop with potential sharp squeezes."
        )
    elif gamma_env == "long":
        behaviour = "mean-reversion towards the magnet with dampened swings."

    wall_desc = (
        "major R wall" if direction == "R"
        else "major S wall" if direction == "S"
        else "major wall"
    )
    return f"{wall_desc} at {strike_label} — expect {behaviour}"


def _get_regime_sign(gamma_val: float, threshold: float = GAMMA_SIGNAL_THRESHOLD) -> int:
    """Helper to clamp small gamma values to 0 to avoid signal noise."""
    if abs(gamma_val) < threshold:
        return 0
    return 1 if gamma_val > 0 else -1


def compute_gex_snapshot(raw: Dict[str, Any]) -> GexSnapshot:
    global _last_delta, _delta_history

    spot = float(raw["spot"])
    now_ms = int(raw["now_ms"])
    options = _build_option_points(raw)

    if not options:
        raise RuntimeError("No active BTC options with open interest returned from Deribit")

    net_gamma = sum(p.gamma_exposure for p in options)
    net_delta = sum(p.delta_exposure for p in options)

    ONE_DAY_MS = 24 * 60 * 60 * 1000
    cutoff_ts = now_ms + ONE_DAY_MS

    gamma_tactical = sum(
        p.gamma_exposure for p in options if p.expiration_timestamp <= cutoff_ts
    )
    gamma_structure = sum(
        p.gamma_exposure for p in options if p.expiration_timestamp > cutoff_ts
    )

    if _last_delta is None:
        delta_trend_short = "Δ baseline — first sample."
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
        delta_trend_long = "Δ (1h) baseline — insufficient history."
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

    wide_range_flag = ""
    try:
        low_str, high_str = near_range.split("/")
        if "—" not in low_str and "—" not in high_str:
            low_v = int(low_str[:-1] if low_str.endswith(("R", "S")) else low_str)
            high_v = int(high_str[:-1] if high_str.endswith(("R", "S")) else high_str)
            if abs(high_v - low_v) >= 4_000:
                wide_range_flag = "Wide chop zone — dispersed walls."
    except Exception:
        pass

    gamma_env, gamma_env_label, comment = _classify_environment(gamma_structure)

    s_sign = _get_regime_sign(gamma_structure)
    t_sign = _get_regime_sign(gamma_tactical)

    expiry_outlook = "Balanced / Weak Signal"

    if s_sign == 1 and t_sign == -1:
        expiry_outlook = "⚠️ Volatility within Support (Dip-Buy)"
    elif s_sign == -1 and t_sign == -1:
        expiry_outlook = "🚨 DANGER: Structural + Tactical Instability"
    elif s_sign == -1 and t_sign == 1:
        expiry_outlook = "🛑 Tactical Pinning (Structural Breakout Risk)"
    elif s_sign == 1 and t_sign == 1:
        expiry_outlook = "✅ Full Spectrum Stability"
    elif s_sign == 0 and t_sign != 0:
        expiry_outlook = f"Tactical Flow Dominant ({'Long' if t_sign > 0 else 'Short'})"

    interpretation = _build_interpretation(
        spot=spot,
        gamma_env=gamma_env,
        near_range=near_range,
        top_walls=top_walls,
        wide_range_flag=wide_range_flag,
    )
    bounce_map = _compute_bounce_map(
        spot=spot,
        strong_range=strong_range,
        gamma_env=gamma_env,
        struct_gamma=gamma_structure,
        net_delta=net_delta,
    )

    return GexSnapshot(
        spot=spot,
        now_ms=now_ms,
        options=options,
        net_gamma=net_gamma,
        net_delta=net_delta,
        gamma_tactical=gamma_tactical,
        gamma_structure=gamma_structure,
        expiry_outlook=expiry_outlook,
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
        delta_trend=delta_trend_short,
        delta_trend_long=delta_trend_long,
        wide_range_flag=wide_range_flag,
        interpretation=interpretation,
        bounce_map=bounce_map,
    )


# ---------- Formatting helpers ----------


def _fmt_price(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return f"{p:,.0f}"


def format_ultra(snapshot: GexSnapshot) -> str:
    """
    Ultra-short print for the main feed.
    """
    spot_s = _fmt_price(snapshot.spot)
    mag_s = _fmt_price(snapshot.magnet)
    delta_s = f"{snapshot.net_delta:,.0f}"
    gamma_m = snapshot.net_gamma / 1_000_000.0

    alert = ""
    if "DANGER" in snapshot.expiry_outlook:
        alert = " 🚨"
    elif "Volatility" in snapshot.expiry_outlook:
        alert = " ⚠️"

    line1 = (
        f"{DERIBIT_CURRENCY} {spot_s} | {snapshot.gamma_env_label}{alert} "
        f"| 🎯 Mag {mag_s} | Δ {delta_s} | Γ {gamma_m:+.2f}M"
    )
    line2 = f"📊 Near {snapshot.near_range} | Strong {snapshot.strong_range}"
    return f"{line1}\n{line2}"


def format_pretty(snapshot: GexSnapshot) -> str:
    """More verbose block for the 'pretty' 15m feed."""
    spot_s = _fmt_price(snapshot.spot)
    flip_s = (
        _fmt_price(snapshot.flip)
        if snapshot.flip and abs(snapshot.flip - snapshot.spot) / snapshot.spot <= 0.25
        else "—"
    )
    mag_s = _fmt_price(snapshot.magnet)
    delta_s = f"{snapshot.net_delta:,.0f}"
    gamma_m = snapshot.net_gamma / 1_000_000.0
    
    gt_m = snapshot.gamma_tactical / 1_000_000.0
    gs_m = snapshot.gamma_structure / 1_000_000.0

    lines: List[str] = [
        f"📊 {DERIBIT_CURRENCY} GEX Update | Spot {spot_s}",
        "",
        f"• Environment: {snapshot.gamma_env_label}",
        f"• Signal: {snapshot.expiry_outlook}",
        "",
        f"• Structure Γ: {gs_m:+.1f}M (Book)",
        f"• Tactical Γ:  {gt_m:+.1f}M (≤24h)",
        "",
        f"• Magnet: {mag_s}  | Flip: {flip_s}",
        f"• Net Γ: {gamma_m:.1f}M | Δ: {delta_s}",
        f"• Near: {snapshot.near_range} | Strong: {snapshot.strong_range}",
    ]

    if snapshot.wide_range_flag:
        lines.append(f"• Range note: {snapshot.wide_range_flag}")
    lines.append(f"• Walls: {snapshot.top_walls}")
    
    if snapshot.band_walls and snapshot.band_walls != "—":
        lines.append(f"• Band walls: {snapshot.band_walls}")
    
    if snapshot.band_bias and snapshot.band_bias != "—":
        lines.append(f"• Band bias: {snapshot.band_bias}")
        if (
            snapshot.band_bias_interpretation
            and "balanced" not in snapshot.band_bias.lower()
        ):
            lines.append(f"• Bias note: {snapshot.band_bias_interpretation}")

    if snapshot.bounce_map and snapshot.bounce_map != "—":
        lines.append(f"• Bounce map: {snapshot.bounce_map}")
        
    lines.append(f"• Δ trend (short): {snapshot.delta_trend}")
    lines.append(f"• Δ trend (1h): {snapshot.delta_trend_long}")
    lines.append(f"• Interpretation: {snapshot.interpretation}")

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
        "delta_trend_short": snapshot.delta_trend,
        "delta_trend_long": snapshot.delta_trend_long,
        "wide_range_flag": snapshot.wide_range_flag,
        "interpretation": snapshot.interpretation,
        "bounce_map": snapshot.bounce_map,
    }


def snapshot_to_pretty_row(snapshot: GexSnapshot) -> Dict[str, Any]:
    # We return the same struct; gexbot_v8 adds the 'flow_bias' field manually
    return snapshot_to_ultra_row(snapshot)