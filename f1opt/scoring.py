"""Fantasy F1 driver scoring, reconstructed from the published race-weekend rules.

This is an *approximation* of the official game. It covers the components that
can be derived from result tables (finishing/qualifying position, positions
gained/lost, fastest lap, DNF/DSQ). It omits overtakes, Driver of the Day and
teammate-beating bonuses, which are not in the results data, and the official
rules have changed between seasons. The same formula is applied to every
season, so model comparisons are internally consistent even where absolute
values differ from the official game.
"""
from typing import Optional

# Points by classified position
QUALIFYING_POINTS = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

POSITION_GAINED_POINTS = 1      # per place gained (negative when places are lost)
RACE_FASTEST_LAP_POINTS = 10
SPRINT_FASTEST_LAP_POINTS = 5
NOT_CLASSIFIED_PENALTY = -20    # retired short of the classification threshold, DSQ

# Ergast/Jolpica positionText values for drivers without a classified finish
UNCLASSIFIED = {"R", "D", "W", "N", "E", "F"}


def qualifying_points(position: Optional[int]) -> float:
    return QUALIFYING_POINTS.get(position, 0) if position else 0


def session_points(position: int, position_text: str, grid: int, field_size: int,
                   table: dict, fastest_lap: bool, fastest_lap_points: int) -> float:
    """Points for a race or sprint: finish points, places gained/lost, fastest lap, DNF."""
    if position_text in UNCLASSIFIED:
        return NOT_CLASSIFIED_PENALTY
    start = grid if grid > 0 else field_size  # pit-lane starters count as last
    gained = (start - position) * POSITION_GAINED_POINTS
    bonus = fastest_lap_points if fastest_lap else 0
    return table.get(position, 0) + gained + bonus
