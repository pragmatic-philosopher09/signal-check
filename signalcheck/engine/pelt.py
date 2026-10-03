"""Pure-numpy PELT with an L2 (change-in-mean) cost.

Statistical meaning: find the segmentation of ``signal`` that minimises the sum,
over segments, of the within-segment sum of squared deviations from the segment
mean, plus ``penalty`` per segment, subject to every segment having at least
``min_size`` points (Killick, Fearnhead & Eckley 2012). Candidate split points
whose best cost can no longer beat the current optimum by more than ``penalty``
are pruned, which keeps the search exact while avoiding the full O(n^2) scan.

This is a faithful re-implementation of ``ruptures.Pelt(model="l2", jump=1)``
(ruptures 1.1.x): the same candidate order, the same per-segment cost
(``segment.var() * len(segment)``), the same left-to-right cost accumulation, the
same first-minimum tie-break and the same pruning rule, so the breakpoints are
identical (``tests/test_pelt.py`` checks this against ruptures on many seeded
series). It exists because ruptures ships compiled extensions with no
WebAssembly wheel, so it cannot be installed in Pyodide; using this one
implementation in every runtime keeps the browser and the server bit-for-bit
consistent. ruptures remains a dev/test dependency only.
"""

from __future__ import annotations

from math import ceil, floor

import numpy as np


class BadSegmentationParameters(ValueError):
    """No segmentation is possible (fewer than ``min_size`` points)."""


def _l2_cost(signal: np.ndarray, start: int, end: int) -> float:
    """Sum of squared deviations from the mean on ``signal[start:end]``.

    Computed exactly as ruptures' ``CostL2.error`` (variance times length on a
    column array) so floating-point results match.
    """
    return float(signal[start:end].var(axis=0).sum() * (end - start))


def _feasible(n_samples: int, min_size: int, jump: int = 1) -> bool:
    """ruptures' ``sanity_check`` for zero breakpoints."""
    return not (0 * ceil(min_size / jump) * jump + min_size > n_samples)


def pelt(signal: np.ndarray, min_size: int, penalty: float) -> list[int]:
    """Breakpoints (segment end indices, the last one equal to ``len(signal)``).

    Matches ``ruptures.Pelt(model="l2", min_size=min_size, jump=1).fit(signal)
    .predict(pen=penalty)``. ``min_size`` is clamped to at least 1 (the L2 cost's
    own minimum); a series shorter than ``min_size`` raises
    :class:`BadSegmentationParameters`.
    """
    x = np.asarray(signal, dtype=float)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    n = int(x.shape[0])
    min_size = max(int(min_size), 1)
    if not _feasible(n, min_size):
        raise BadSegmentationParameters(f"cannot segment {n} points with min_size={min_size}")

    # best[t]: (total cost of the optimal partition of x[:t], its segment ends).
    best: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    admissible: list[int] = []
    ends = [k for k in range(n) if k >= min_size] + [n]
    for end in ends:
        admissible.append(floor((end - min_size) / 1) * 1)
        candidates: list[tuple[int, float, tuple[int, ...]]] = []
        for t in admissible:
            if t not in best:
                continue
            prev_cost, prev_ends = best[t]
            candidates.append((t, prev_cost + (_l2_cost(x, t, end) + penalty), (*prev_ends, end)))
        # First minimum wins, as with Python's min() in ruptures.
        winner = candidates[0]
        for cand in candidates[1:]:
            if cand[1] < winner[1]:
                winner = cand
        best[end] = (winner[1], winner[2])
        admissible = [t for t, cost, _ in candidates if cost <= winner[1] + penalty]
    return sorted(best[n][1])
