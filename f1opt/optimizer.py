"""Team selection as an exact mixed-integer program (HiGHS via scipy), not a greedy pick.

Rules modelled: exactly ``n_drivers`` drivers and ``n_constructors`` constructors, total price within
the budget cap, and a DRS booster that doubles the points of one chosen driver. The DRS pick is
optimised jointly with the team (it is not simply "the best driver" once budget binds).
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


class InfeasibleTeam(ValueError):
    """No selection satisfies the roster and budget rules."""


@dataclass(frozen=True)
class Asset:
    id: str
    price: float
    points: float  # predicted Fantasy points for the target race


@dataclass
class TeamSelection:
    drivers: List[Asset]
    constructors: List[Asset]
    drs_driver: Optional[str]
    total_price: float
    expected_points: float  # includes the DRS doubling
    budget: float = 0.0


def optimize_team(drivers: List[Asset], constructors: List[Asset], budget: float,
                  n_drivers: int = 5, n_constructors: int = 2, drs: bool = True) -> TeamSelection:
    nd, nc = len(drivers), len(constructors)
    if nd < n_drivers or nc < n_constructors:
        raise InfeasibleTeam("not enough candidates for the roster")

    # Variable layout: x (driver picked) | y (DRS driver) | z (constructor picked)
    n = 2 * nd + nc
    c = np.zeros(n)
    c[:nd] = [-d.points for d in drivers]
    if drs:
        c[nd:2 * nd] = [-d.points for d in drivers]
    c[2 * nd:] = [-k.points for k in constructors]

    rows, lo, hi = [], [], []

    def add(row, lower, upper):
        rows.append(row), lo.append(lower), hi.append(upper)

    row = np.zeros(n); row[:nd] = 1; add(row, n_drivers, n_drivers)
    row = np.zeros(n); row[2 * nd:] = 1; add(row, n_constructors, n_constructors)
    row = np.zeros(n); row[:nd] = [d.price for d in drivers]; row[2 * nd:] = [k.price for k in constructors]
    add(row, -np.inf, budget + 1e-9)
    row = np.zeros(n); row[nd:2 * nd] = 1 if drs else 0; add(row, 1 if drs else 0, 1 if drs else 0)
    for i in range(nd):  # the DRS driver must be on the team
        row = np.zeros(n); row[nd + i] = 1; row[i] = -1; add(row, -np.inf, 0)

    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        raise InfeasibleTeam(f"no team fits the budget of {budget}: {res.message}")

    x = np.round(res.x).astype(int)
    picked = [drivers[i] for i in range(nd) if x[i]]
    picked_c = [constructors[j] for j in range(nc) if x[2 * nd + j]]
    drs_id = next((drivers[i].id for i in range(nd) if x[nd + i]), None)
    total = sum(a.price for a in picked) + sum(a.price for a in picked_c)
    points = sum(a.points for a in picked) + sum(a.points for a in picked_c)
    if drs_id is not None:
        points += next(a.points for a in picked if a.id == drs_id)
    return TeamSelection(picked, picked_c, drs_id, float(total), float(points), budget)


def greedy_team(drivers: List[Asset], constructors: List[Asset], budget: float,
                n_drivers: int = 5, n_constructors: int = 2) -> Optional[TeamSelection]:
    """The old approach (rank by value, take while affordable), kept as a baseline for comparison.

    Ranks by points per $M, fills drivers then constructors, then applies DRS to the best driver.
    Returns None when the greedy walk cannot complete the roster.
    """
    spent, pick_d, pick_c = 0.0, [], []
    for a in sorted(drivers, key=lambda a: a.points / a.price, reverse=True):
        if len(pick_d) < n_drivers and spent + a.price <= budget:
            pick_d.append(a); spent += a.price
    for a in sorted(constructors, key=lambda a: a.points / a.price, reverse=True):
        if len(pick_c) < n_constructors and spent + a.price <= budget:
            pick_c.append(a); spent += a.price
    if len(pick_d) < n_drivers or len(pick_c) < n_constructors:
        return None
    best = max(pick_d, key=lambda a: a.points)
    points = sum(a.points for a in pick_d + pick_c) + best.points
    return TeamSelection(pick_d, pick_c, best.id, spent, points, budget)
