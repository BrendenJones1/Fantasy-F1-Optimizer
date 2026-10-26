from itertools import combinations

import numpy as np
import pytest

from f1opt.optimizer import Asset, InfeasibleTeam, greedy_team, optimize_team


def brute_force(drivers, constructors, budget, nd=5, nc=2):
    best = -np.inf
    for ds in combinations(drivers, nd):
        for cs in combinations(constructors, nc):
            cost = sum(a.price for a in ds) + sum(a.price for a in cs)
            if cost <= budget + 1e-9:
                pts = sum(a.points for a in ds) + sum(a.points for a in cs) + max(a.points for a in ds)
                best = max(best, pts)
    return best


def random_market(rng, n_drivers=9, n_constructors=4):
    drivers = [Asset(f"d{i}", float(rng.uniform(5, 30)), float(rng.normal(8, 10))) for i in range(n_drivers)]
    constructors = [Asset(f"c{i}", float(rng.uniform(5, 30)), float(rng.normal(15, 15))) for i in range(n_constructors)]
    return drivers, constructors


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    drivers, constructors = random_market(rng)
    budget = float(rng.uniform(70, 110))
    try:
        expected = brute_force(drivers, constructors, budget)
    except ValueError:
        return
    if expected == -np.inf:
        with pytest.raises(InfeasibleTeam):
            optimize_team(drivers, constructors, budget)
        return
    team = optimize_team(drivers, constructors, budget)
    assert team.expected_points == pytest.approx(expected)


def test_respects_roster_and_budget_rules():
    rng = np.random.default_rng(3)
    drivers, constructors = random_market(rng, 20, 10)
    team = optimize_team(drivers, constructors, 100.0)
    assert len(team.drivers) == 5 and len(team.constructors) == 2
    assert len({a.id for a in team.drivers}) == 5
    assert team.total_price <= 100.0 + 1e-6
    assert team.drs_driver in {a.id for a in team.drivers}
    # DRS goes to the highest-scoring driver on the chosen team
    assert team.drs_driver == max(team.drivers, key=lambda a: a.points).id


def test_strictly_beats_greedy_on_most_random_markets():
    """Greedy-by-value wastes budget; on seeded random markets the MILP is strictly better most of the time."""
    rng = np.random.default_rng(5)
    wins, compared = 0, 0
    for _ in range(100):
        drivers, constructors = random_market(rng, 20, 10)
        opt = optimize_team(drivers, constructors, 100.0)
        greedy = greedy_team(drivers, constructors, 100.0)
        if greedy is None:  # greedy could not even complete a legal roster
            continue
        compared += 1
        wins += opt.expected_points > greedy.expected_points + 1e-9
    assert compared > 50 and wins / compared > 0.5


def test_infeasible_budget_raises():
    drivers = [Asset(f"d{i}", 20, 10) for i in range(6)]
    constructors = [Asset(f"c{i}", 20, 10) for i in range(3)]
    with pytest.raises(InfeasibleTeam):
        optimize_team(drivers, constructors, 50.0)


def test_never_worse_than_greedy():
    rng = np.random.default_rng(11)
    for _ in range(30):
        drivers, constructors = random_market(rng, 20, 10)
        opt = optimize_team(drivers, constructors, 100.0)
        greedy = greedy_team(drivers, constructors, 100.0)
        if greedy is not None:
            assert opt.expected_points >= greedy.expected_points - 1e-9
