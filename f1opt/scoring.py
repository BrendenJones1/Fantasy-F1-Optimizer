"""Fantasy F1 driver scoring, following the official 2026 rules published by the game.

Covered: qualifying position points and the -5 for no time set, race and sprint finishing points,
positions gained/lost (sprint losses capped at -10), fastest lap, and Not Classified/DSQ penalties.
Not available from result tables, hence missing from the target: overtakes (+1 each) and Driver of
the Day (+10). The season totals this produces correlate ~0.99 with the game's own season totals
(see ``scripts/validate_scoring.py``), with a mean shortfall of about 6 points per race that is
consistent with those two missing components.

The 2026 rules are applied to every season, except the sprint Not Classified/DSQ penalty, which was
-20 before 2026. Constructor bonuses (qualifying progress, pit stops) are not part of driver scores.
"""
from typing import Optional

# Points by classified position
QUALIFYING_POINTS = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

POSITION_GAINED_POINTS = 1      # per place gained (negative when places are lost)
RACE_FASTEST_LAP_POINTS = 10
SPRINT_FASTEST_LAP_POINTS = 5
NOT_CLASSIFIED_PENALTY = -20    # race: retired short of the classification threshold, DSQ
SPRINT_NOT_CLASSIFIED_PENALTY_2026 = -10   # reduced from -20 for 2026
SPRINT_MAX_POSITIONS_LOST = 10
NO_QUALIFYING_TIME_PENALTY = -5

# Ergast/Jolpica positionText values for drivers without a classified finish
UNCLASSIFIED = {"R", "D", "W", "N", "E", "F"}


def sprint_penalty(season: int) -> int:
    return SPRINT_NOT_CLASSIFIED_PENALTY_2026 if season >= 2026 else NOT_CLASSIFIED_PENALTY


def qualifying_points(position: Optional[int], has_time: bool = True) -> float:
    if not has_time:
        return NO_QUALIFYING_TIME_PENALTY
    return QUALIFYING_POINTS.get(position, 0) if position else 0


def session_points(position: int, position_text: str, grid: int, field_size: int,
                   table: dict, fastest_lap: bool, fastest_lap_points: int,
                   not_classified_penalty: int = NOT_CLASSIFIED_PENALTY,
                   max_positions_lost: Optional[int] = None) -> float:
    """Points for a race or sprint: finish points, places gained/lost, fastest lap, DNF."""
    if position_text in UNCLASSIFIED:
        return not_classified_penalty
    start = grid if grid > 0 else field_size  # pit-lane starters count as last
    moved = (start - position) * POSITION_GAINED_POINTS
    if max_positions_lost is not None:
        moved = max(moved, -max_positions_lost)
    bonus = fastest_lap_points if fastest_lap else 0
    return table.get(position, 0) + moved + bonus
