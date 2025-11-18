# gex_core.py
from dataclasses import dataclass
from typing import Dict, Any

# === Data model examples ===

@dataclass
class GexSnapshot:
    spot: float
    gamma_sign: str             # "Long" / "Short"
    gamma_flip: float | None    # flip level, if any
    magnet: float | None        # target / magnet level
    delta: float | None
    net_gamma: float | None
    near_range: str | None      # e.g. "95000R/96000R"
    strong_range: str | None    # e.g. "95000R/96000R"

# === Fetch / compute functions (stubbed) ===

def fetch_raw_gex_data() -> Dict[str, Any]:
    """
    TODO: Plug in your actual Deribit / data source logic here.
    Must return whatever you need to compute GEX.
    """
    # Placeholder example:
    return {
        "spot": 95184.0,
        # ...
    }

def compute_gex_snapshot(raw: Dict[str, Any]) -> GexSnapshot:
    """
    TODO: Implement using your actual formulas from v7.4 + patches.
    For now we mock a simple snapshot so the skeleton runs.
    """
    return GexSnapshot(
        spot=raw.get("spot", 0),
        gamma_sign="Short",
        gamma_flip=106500.0,
        magnet=100000.0,
        delta=11099.0,
        net_gamma=-120_000_000.0,
        near_range="95000R/96000R",
        strong_range="95000R/96000R",
    )

# === Formatters ===

def format_ultra(snapshot: GexSnapshot) -> str:
    """
    Ultra 5-min print – dense, machine-like.
    """
    sign_emoji = "🟢" if snapshot.gamma_sign == "Long" else "🔴"
    flip_str = f"<~{snapshot.gamma_flip:,.0f}" if snapshot.gamma_flip else "—"
    mag_str = f"{snapshot.magnet:,.0f}" if snapshot.magnet else "—"
    delta_str = f"{snapshot.delta:,.0f}" if snapshot.delta is not None else "—"
    net_gamma_str = f"{snapshot.net_gamma/1_000_000:.2f}M" if snapshot.net_gamma is not None else "—"

    return (
        f"BTC {snapshot.spot:,.0f} | {sign_emoji} γ {snapshot.gamma_sign} {flip_str} "
        f"| 🎯 Mag {mag_str} | Δ {delta_str} | Γ {net_gamma_str}\n"
        f"📊 Near {snapshot.near_range or '—'} | Strong {snapshot.strong_range or '—'}"
    )

def format_pretty(snapshot: GexSnapshot) -> str:
    """
    Pretty 15-min print – more human commentary.
    """
    env_label = "Neg γ – expansion / whipsaw risk" if snapshot.gamma_sign == "Short" else "Pos γ – mean-revert / damped moves"

    mag_str = f"{snapshot.magnet:,.0f}" if snapshot.magnet else "—"
    delta_str = f"{snapshot.delta:,.0f}" if snapshot.delta is not None else "—"
    net_gamma_str = f"{snapshot.net_gamma/1_000_000:.2f}M" if snapshot.net_gamma is not None else "—"

    lines = [
        f"📊 *BTC GEX Update* | Spot {snapshot.spot:,.0f}",
        "",
        f"• Environment: {env_label}",
        f"• Magnet: `{mag_str}`  | Flip: `{snapshot.gamma_flip:,.0f}`" if snapshot.gamma_flip else f"• Magnet: `{mag_str}`",
        f"• Net Γ: `{net_gamma_str}` | Δ: `{delta_str}`",
        f"• Near: `{snapshot.near_range or '—'}` | Strong: `{snapshot.strong_range or '—'}`",
        "",
        "_History bot is logging this print for later review._",
    ]
    return "\n".join(lines)

def snapshot_to_ultra_row(snapshot: GexSnapshot) -> Dict[str, Any]:
    return {
        "spot": snapshot.spot,
        "gamma_sign": snapshot.gamma_sign,
        "gamma_flip": snapshot.gamma_flip,
        "magnet": snapshot.magnet,
        "delta": snapshot.delta,
        "net_gamma": snapshot.net_gamma,
        "near_range": snapshot.near_range,
        "strong_range": snapshot.strong_range,
    }

def snapshot_to_pretty_row(snapshot: GexSnapshot) -> Dict[str, Any]:
    env_label = "Neg" if snapshot.gamma_sign == "Short" else "Pos"
    return {
        "spot": snapshot.spot,
        "gamma_env_label": env_label,
        "magnet": snapshot.magnet,
        "delta": snapshot.delta,
        "net_gamma": snapshot.net_gamma,
        "comment": env_label,
    }
