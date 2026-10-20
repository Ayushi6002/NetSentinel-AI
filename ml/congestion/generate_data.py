"""
Simulated Network Congestion Data Generator — NetSentinel AI Phase 2
====================================================================
Generates synthetic network telemetry for a monitored 100 Mbps bottleneck link
at 1-minute resolution over a configurable duration (default: 7 days, 10,080 rows).

SCIENTIFIC & ETHICAL DISCLAIMER:
--------------------------------
This dataset is ENTIRELY SIMULATED. It does NOT represent physical network measurements,
campus router captures, or live ISP telemetry. Physical network queue lengths and packet drops
occur inside hardware ASICs and switch buffers that require specialized SNMP/NetFlow or eBPF probes.
This generator uses an idealized bottleneck queuing model to produce reproducible, realistic telemetry
for developing, testing, and demonstrating the Phase 2 predictive modeling pipeline.

MODELING ASSUMPTIONS & DESIGN:
------------------------------
1. Bottleneck Link Capacity: Fixed at 100.0 Mbps.
2. Offered Load: Composed of:
   - Diurnal sinusoidal variation (peaks during afternoon/business hours, troughs at night)
   - Day-of-week profile (lower background on weekends)
   - Autoregressive short-term noise (AR-1 process for temporal correlation)
   - Poisson-distributed transient traffic bursts
   - Multi-hour regime shifts (e.g., scheduled offsite backups, enterprise sync surges)
3. Physical Utilization:
   - When offered_load <= capacity: utilization = offered_load / capacity
   - When offered_load > capacity: utilization is clamped to 1.0 (the physical link cannot
     transmit faster than wire speed). The excess load represents queued or dropped traffic.
4. Queue Delay Approximation:
   - Uses an idealized queuing model: base propagation/processing delay (2.5 ms) plus non-linear
     queuing growth as utilization approaches 1.0, plus buffer backlog for excess load,
     clamped to a finite buffer limit (120.0 ms).
   - Limitation: This is a macro-level approximation, not a discrete packet-level simulator (e.g. ns-3).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SEED = 42
DEFAULT_DAYS = 7
INTERVAL_MINUTES = 1
LINK_CAPACITY_MBPS = 100.0
DEFAULT_OUTPUT_PATH = Path("data/processed/simulated_congestion_1min.csv")


def generate_congestion_dataset(
    days: int = DEFAULT_DAYS,
    seed: Optional[int] = DEFAULT_SEED,
    start_time: str = "2024-11-01 00:00:00",
    link_capacity_mbps: float = LINK_CAPACITY_MBPS,
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute network bottleneck telemetry.

    Parameters
    ----------
    days : int, default 7
        Duration in days (7 days = 10,080 minutes).
    seed : int or None, default 42
        Random seed for reproducibility. If None, uses non-deterministic random state.
    start_time : str, default '2024-11-01 00:00:00'
        Initial timestamp.
    link_capacity_mbps : float, default 100.0
        Capacity of the simulated bottleneck link in Mbps.

    Returns
    -------
    pd.DataFrame with columns:
        [timestamp, offered_load_mbps, utilization, queue_delay_ms, flow_count, udp_share, simulated]
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    total_steps = days * 24 * 60
    timestamps = pd.date_range(start=start_time, periods=total_steps, freq=f"{INTERVAL_MINUTES}min")

    minute_of_day = timestamps.hour * 60 + timestamps.minute
    day_of_week = timestamps.dayofweek  # 0=Monday, ..., 6=Sunday

    # 1. Base Diurnal Cycle (24h period)
    # Peak at ~15:00 (minute 900), trough at ~04:00 (minute 240)
    diurnal_phase = 2 * np.pi * (minute_of_day - 240) / 1440.0
    diurnal_factor = 0.5 * (1.0 - np.cos(diurnal_phase))  # ranges 0.0 to 1.0
    base_load = 22.0 + 36.0 * diurnal_factor  # 22 Mbps at night, 58 Mbps peak

    # 2. Weekend reduction factor (weekend traffic ~75% of weekday)
    is_weekend = day_of_week >= 5
    weekend_factor = np.where(is_weekend, 0.75, 1.0)
    base_load = base_load * weekend_factor

    # 3. Autoregressive AR(1) short-term traffic noise (persistence over several minutes)
    ar_noise = np.zeros(total_steps)
    phi = 0.85
    white_noise = rng.normal(0.0, 3.5, total_steps)
    for t in range(1, total_steps):
        ar_noise[t] = phi * ar_noise[t - 1] + white_noise[t]

    # 4. Regime Shifts & Planned Burst Events across different days
    # Ensures all temporal splits encounter realistic congestion regimes
    regime_shift = np.zeros(total_steps)

    # Shift 1 (Friday, day 4): Afternoon software rollout surge (14:00-17:00, +30 Mbps)
    mask_fri = (day_of_week == 4) & (timestamps.hour >= 14) & (timestamps.hour < 17)
    regime_shift[mask_fri] += 30.0

    # Shift 2 (Monday, day 0): Morning business ramp-up surge (09:00-11:30, +26 Mbps)
    mask_mon = (day_of_week == 0) & (
        ((timestamps.hour == 9) & (timestamps.minute >= 0))
        | (timestamps.hour == 10)
        | ((timestamps.hour == 11) & (timestamps.minute <= 30))
    )
    regime_shift[mask_mon] += 26.0

    # Shift 3 (Wednesday, day 2): Offsite backup window (02:30-05:00, +32 Mbps)
    mask_wed = (day_of_week == 2) & (
        ((timestamps.hour == 2) & (timestamps.minute >= 30))
        | (timestamps.hour == 3)
        | (timestamps.hour == 4)
        | ((timestamps.hour == 5) & (timestamps.minute == 0))
    )
    regime_shift[mask_wed] += 32.0

    # Shift 4 (Thursday, day 3): Enterprise analytics & cloud sync (14:00-16:30, +30 Mbps)
    mask_thu = (day_of_week == 3) & (
        (timestamps.hour == 14)
        | (timestamps.hour == 15)
        | ((timestamps.hour == 16) & (timestamps.minute <= 30))
    )
    regime_shift[mask_thu] += 30.0

    # 5. Sporadic high-load bursts (Poisson arrivals of file downloads / syncs)
    num_bursts = int(days * 12)  # ~12 bursts per day
    burst_starts = rng.integers(0, total_steps - 30, size=num_bursts)
    burst_durations = rng.integers(5, 25, size=num_bursts)  # 5 to 25 minutes
    burst_amplitudes = rng.uniform(15.0, 30.0, size=num_bursts)

    burst_load = np.zeros(total_steps)
    for start, dur, amp in zip(burst_starts, burst_durations, burst_amplitudes):
        end = min(total_steps, start + dur)
        w = np.sin(np.pi * np.linspace(0, 1, end - start)) ** 2
        burst_load[start:end] += amp * w

    # Total Offered Load
    offered_load = base_load + ar_noise + regime_shift + burst_load
    offered_load = np.maximum(offered_load, 5.0)  # Minimum 5 Mbps baseline

    # Physical Link Utilization:
    # Offered load divided by capacity, capped at 1.0 (physical wire saturation)
    raw_utilization = offered_load / link_capacity_mbps
    utilization = np.clip(raw_utilization, 0.0, 1.0)

    # Queue Delay Approximation:
    # Base delay = 2.5 ms
    # As utilization approaches 1.0, buffer builds up non-linearly.
    # Excess offered load beyond capacity incurs queue backlog delay.
    excess_load = np.maximum(0.0, offered_load - link_capacity_mbps)
    nonlinear_factor = np.where(
        utilization < 0.98,
        (utilization / (1.005 - utilization)) ** 1.3,
        (0.98 / (1.005 - 0.98)) ** 1.3 + (utilization - 0.98) * 100.0,
    )
    queue_delay = 2.5 + 4.5 * nonlinear_factor + 2.0 * excess_load
    jitter = rng.normal(0.0, 0.4, total_steps)
    queue_delay = np.clip(queue_delay + jitter, 1.5, 120.0)

    # Active Flow Count (correlated with offered load + independent variation)
    flow_count = 60 + 5.5 * offered_load + rng.normal(0, 15.0, total_steps)
    flow_count = np.clip(flow_count, 20, 1200).astype(int)

    # UDP Share (baseline ~0.18, higher during burst periods up to ~0.55)
    udp_baseline = 0.18 + 0.15 * (burst_load / 35.0) + rng.normal(0, 0.02, total_steps)
    udp_share = np.clip(udp_baseline, 0.05, 0.65)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "offered_load_mbps": np.round(offered_load, 2),
            "utilization": np.round(utilization, 4),
            "queue_delay_ms": np.round(queue_delay, 2),
            "flow_count": flow_count,
            "udp_share": np.round(udp_share, 4),
            "simulated": True,
        }
    )

    return df


def save_dataset(df: pd.DataFrame, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Save generated dataset to CSV, ensuring directory exists."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows of simulated congestion data to %s", len(df), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated network congestion telemetry.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of days (default: 7).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42).")
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path."
    )
    parser.add_argument(
        "--capacity", type=float, default=LINK_CAPACITY_MBPS, help="Bottleneck capacity in Mbps."
    )
    args = parser.parse_args()

    df = generate_congestion_dataset(days=args.days, seed=args.seed, link_capacity_mbps=args.capacity)
    saved = save_dataset(df, args.output)
    print(f"Generated {len(df)} rows. Saved to {saved}")
    print(f"Utilization range: [{df['utilization'].min():.4f}, {df['utilization'].max():.4f}]")
    print(f"Mean utilization: {df['utilization'].mean():.4f}")
    congested_count = (df['utilization'] >= 0.80).sum()
    print(f"Congested rows (utilization >= 0.80): {congested_count} ({congested_count / len(df) * 100:.2f}%)")


if __name__ == "__main__":
    main()
