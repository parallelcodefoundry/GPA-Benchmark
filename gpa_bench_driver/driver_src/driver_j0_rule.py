"""The ONE J0 credit rule for Frontier GPA (GPA-G1 J0 / fix round 5 K2). Standard library only.

gpa_test, the runner verdict, the T6 rescore and analysis/consolidate all decide credit with
:func:`j0_credited` on the numbers :func:`j0_estimate` computes (agent M's M.md T3):

- ``speedup = S_pair`` when the series is unstable, else ``min(S_med, S_pair)``, where
  ``S_med = median(B) / (median(O) + charge)`` and ``S_pair`` is the median of the per-pair ratios
  ``B_k / (O_k + charge)`` (ABBA pairs);
- ``unstable``: an arm's robust spread ``(2nd largest - 2nd smallest) / median`` exceeds
  ``level_tol`` (VRAM placement changed between runs);
- ``lower_bound``: the distribution-free lower confidence bound of the median pair ratio, the
  order statistic ``sorted(ratios)[lower_bound_index(m)]``;
- **credited** iff ``speedup > 1.005`` and (not unstable or ``lower_bound > 1.005``).
"""

from __future__ import annotations

import math
from typing import Any

J0_CREDIT_FLOOR = 1.005
J0_LEVEL_TOL = 0.025
# one-sided miss probability of the lower bound: coverage >= 98% for every m >= 6
J0_LB_ALPHA = 0.02


def median(values: list[float]) -> float:
    """Median of a non-empty list."""
    v = sorted(float(x) for x in values)
    n = len(v)
    if n == 0:
        msg = "median of an empty series"
        raise ValueError(msg)
    mid = n // 2
    return v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2.0


def mad(values: list[float]) -> float:
    """Median absolute deviation (unscaled)."""
    med = median(values)
    return median([abs(float(x) - med) for x in values])


def robust_spread(values: list[float]) -> float:
    """(2nd largest - 2nd smallest) / median; 0 for fewer than 4 values (M.md T3)."""
    if len(values) < 4:
        return 0.0
    v = sorted(float(x) for x in values)
    med = median(v)
    return (v[-2] - v[1]) / med if med > 0 else 0.0


def binom_half_cdf(j: int, m: int) -> float:
    """P(Bin(m, 1/2) <= j)."""
    return sum(math.comb(m, i) for i in range(j + 1)) / 2.0 ** m


def lower_bound_index(m: int, alpha: float = J0_LB_ALPHA) -> int:
    """Index j (0-based) of the order statistic that bounds the median pair ratio from below.

    The largest j with P(Bin(m, 1/2) <= j) <= alpha: the bound sorted(r)[j] lies below the true
    median with probability >= 1 - alpha (distribution-free). With alpha = 0.02 this reproduces
    M.md at m = 6, 10, 12, 20 (0, 1, 2, 4); below m = 6 no j reaches the coverage and 0 is used.
    """
    m = int(m)
    j = 0
    while j + 1 < m and binom_half_cdf(j + 1, m) <= alpha:
        j += 1
    return j


def lower_bound_coverage(m: int, alpha: float = J0_LB_ALPHA) -> float:
    """Coverage 1 - P(Bin(m, 1/2) <= j) of :func:`lower_bound_index` at m pairs."""
    return 1.0 - binom_half_cdf(lower_bound_index(m, alpha), m)


def j0_credited(speedup: float | None, unstable: bool, lower_bound: float | None,
                floor: float = J0_CREDIT_FLOOR) -> bool:
    """THE J0 credit rule: speedup > floor and (not unstable or lower_bound > floor)."""
    if speedup is None or not speedup > floor:
        return False
    if not unstable:
        return True
    return lower_bound is not None and lower_bound > floor


def j0_estimate(
    baseline_scored_ns: list[float],
    optimized_scored_ns: list[float],
    *,
    charge_ns: float = 0.0,
    pairs_b: list | None = None,
    pairs_o: list | None = None,
    level_tol: float = J0_LEVEL_TOL,
    placement_levels_ms: dict | None = None,
) -> dict[str, Any]:
    """M.md T3/T5 from per-run scored times (ns). ``pairs_b``/``pairs_o`` are the runs' ABBA pair
    indices (default: position, i.e. the k-th run of each arm is pair k).

    Returns the ``j0`` dict of ``driver_rocprof.score_frontier``.
    """
    b = [float(x) for x in baseline_scored_ns]
    o = [float(x) for x in optimized_scored_ns]
    if not b or not o:
        msg = "j0_estimate needs at least one run per arm"
        raise ValueError(msg)
    kb = list(pairs_b) if pairs_b is not None else list(range(len(b)))
    ko = list(pairs_o) if pairs_o is not None else list(range(len(o)))
    omap = dict(zip(ko, o))
    bmap = dict(zip(kb, b))
    keys = [k for k in kb if k in omap]
    if not keys:
        msg = "no ABBA pairs in common between the baseline and optimized series"
        raise ValueError(msg)
    charge = float(charge_ns)
    r = [bmap[k] / (omap[k] + charge) for k in keys]
    s_med = median(b) / (median(o) + charge)
    s_pair = median(r)
    spread_b = robust_spread(b)
    spread_o = robust_spread(o)
    unstable = spread_b > level_tol or spread_o > level_tol
    speedup = s_pair if unstable else min(s_med, s_pair)
    j = lower_bound_index(len(r))
    lower = sorted(r)[j]
    level_ms = median(b) / 1e6
    regime = None
    if placement_levels_ms:
        regime = min(placement_levels_ms,
                     key=lambda n: abs(float(placement_levels_ms[n]) - level_ms))
    return {"protocol": "j0", "speedup": speedup, "speedup_median_ratio": s_med,
            "speedup_pair_median": s_pair, "pair_ratios": r, "spread_b": spread_b,
            "spread_o": spread_o, "level_tol": level_tol, "unstable": unstable,
            "lower_bound": lower, "lower_bound_index": j,
            "lower_bound_coverage": 1.0 - binom_half_cdf(j, len(r)),
            "credit_floor": J0_CREDIT_FLOOR,
            "credited": j0_credited(speedup, unstable, lower),
            "n_pairs": len(r), "baseline_level_ms": level_ms, "regime": regime,
            "remeasured": False}


def j0_decision(j0: dict) -> tuple[bool, str]:
    """Re-decide a recorded ``j0`` dict (e.g. a sidecar's) with :func:`j0_credited`.

    When ``pair_ratios`` is present the lower bound is recomputed with :func:`lower_bound_index`
    (records written before fix round 5 used a fixed table).
    """
    speedup = j0.get("speedup")
    unstable = bool(j0.get("unstable"))
    ratios = j0.get("pair_ratios")
    lower = j0.get("lower_bound")
    if ratios:
        lower = sorted(float(x) for x in ratios)[lower_bound_index(len(ratios))]
    ok = j0_credited(speedup, unstable, lower)
    if speedup is None:
        return False, "no J0 speedup"
    if not speedup > J0_CREDIT_FLOOR:
        return False, f"J0 speedup {speedup:.4f} is not above {J0_CREDIT_FLOOR}"
    if unstable and not ok:
        return False, (f"unstable series and its lower bound {lower:.4f} is not above "
                       f"{J0_CREDIT_FLOOR}")
    return True, (f"J0 speedup {speedup:.4f} > {J0_CREDIT_FLOOR}"
                  + (f" (unstable; lower bound {lower:.4f} > {J0_CREDIT_FLOOR})" if unstable else ""))


def other_charge(other_b_ns: list[float], other_o_ns: list[float],
                 baseline_scored_ns: list[float] | None = None,
                 floor: float = J0_CREDIT_FLOOR) -> dict[str, float | None]:
    """T2 noise-tolerant other-kernel charge (fix round 5, capped in fix round 6 L1):
    charge = max(0, med(O) - med(B) - tol),
    tol = min(max(1 us, 3 (MAD_O + MAD_B) / sqrt(n)), 0.5 (floor - 1) med(B_scored)).

    The cap (half the credit margin of the baseline's scored time) bounds what a pure
    target -> other-kernel shift can gain uncharged to a speedup of 1 / (1 - (floor - 1) / 2),
    about 1.0025, so such a shift can never be credited on its own. Without ``baseline_scored_ns``
    the tolerance is not capped (old callers)."""
    n = min(len(other_b_ns), len(other_o_ns))
    med_b = median(other_b_ns)
    med_o = median(other_o_ns)
    mad_b = mad(other_b_ns)
    mad_o = mad(other_o_ns)
    noise_tol = max(1000.0, 3.0 * (mad_o + mad_b) / math.sqrt(max(n, 1)))
    cap = None
    if baseline_scored_ns:
        cap = 0.5 * (floor - 1.0) * median(baseline_scored_ns)
    tol = noise_tol if cap is None else min(noise_tol, cap)
    return {"charge_ns": max(0.0, med_o - med_b - tol), "baseline_median_ns": med_b,
            "optimized_median_ns": med_o, "mad_b_ns": mad_b, "mad_o_ns": mad_o, "tol_ns": tol,
            "tol_noise_ns": noise_tol, "tol_cap_ns": cap}
