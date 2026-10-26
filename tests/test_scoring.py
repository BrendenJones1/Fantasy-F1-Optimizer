import pytest

from f1opt import scoring as sc


def race(position, grid, text=None, fl=False, field=22):
    return sc.session_points(position, text or str(position), grid, field, sc.RACE_POINTS, fl,
                             sc.RACE_FASTEST_LAP_POINTS)


def test_pole_win_with_fastest_lap_matches_known_2025_round_1():
    # Norris, 2025 Australia: pole (10) + win (25) + fastest lap (10) = 45, no places gained.
    assert sc.qualifying_points(1) + race(1, 1, fl=True) == 45


def test_positions_gained_and_lost_are_uncapped_in_races():
    assert race(1, 16) == 25 + 15
    assert race(15, 1) == 0 - 14


def test_not_classified_and_dsq_take_the_race_penalty_only():
    assert race(20, 3, text="R") == -20
    assert race(5, 5, text="D") == -20


def test_pit_lane_start_counts_as_last_on_grid():
    assert race(10, 0, field=20) == 1 + 10


def test_qualifying_points_and_no_time_penalty():
    assert [sc.qualifying_points(p) for p in (1, 10, 11)] == [10, 1, 0]
    assert sc.qualifying_points(None, has_time=False) == -5


def sprint(position, grid, text=None, fl=False, season=2026):
    return sc.session_points(position, text or str(position), grid, 22, sc.SPRINT_POINTS, fl,
                             sc.SPRINT_FASTEST_LAP_POINTS, not_classified_penalty=sc.sprint_penalty(season),
                             max_positions_lost=sc.SPRINT_MAX_POSITIONS_LOST)


def test_sprint_losses_capped_at_ten_and_penalty_changed_in_2026():
    assert sprint(20, 2) == -10          # lost 18 places, capped
    assert sprint(1, 1, fl=True) == 8 + 5
    assert sprint(10, 3, text="R", season=2026) == -10
    assert sprint(10, 3, text="R", season=2025) == -20
