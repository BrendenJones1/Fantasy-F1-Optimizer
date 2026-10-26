"""Pre-race features for predicting a driver's Fantasy points at a race weekend.

Leakage policy
--------------
Fantasy team lock is before qualifying, so *nothing from the target weekend*
(grid, qualifying, race) may be used. Every feature is built from weekends
strictly earlier in the calendar via ``shift(1)`` before any rolling/expanding
operation. Only information known in advance is used from the target weekend:
the circuit, whether it is a sprint weekend, the round number and which team
the driver races for. ``tests/test_features.py`` enforces this by scrambling
outcomes and checking that features do not move.
"""
import numpy as np
import pandas as pd

ID_COLUMNS = ["season", "round", "race_index", "date", "driver_id", "constructor_id", "circuit_id"]
TARGET = "fantasy_points"

# Columns that describe what happened at a weekend (not knowable beforehand).
OUTCOME_COLUMNS = ["grid", "finish_position", "position_text", "quali_position",
                   "quali_pts", "sprint_pts", "race_pts", "fantasy_points"]

FEATURE_COLUMNS = [
    "drv_pts_ma3", "drv_pts_ma5", "drv_pts_ma10", "drv_pts_ewm",
    "drv_season_mean", "drv_pos_ma5", "drv_grid_ma5", "drv_dnf_rate10", "drv_n_prior",
    "team_pts_ma3", "team_pts_ma10", "team_grid_ma5",
    "circuit_drv_mean", "circuit_drv_n", "circuit_team_mean",
    "is_sprint_weekend", "round",
]


def _prior(series_group, func):
    """Apply ``func`` to each group's series shifted by one race."""
    return series_group.transform(lambda s: func(s.shift(1)))


def build_features(weekends: pd.DataFrame) -> pd.DataFrame:
    df = weekends.sort_values(["season", "round", "driver_id"]).reset_index(drop=True).copy()
    df["race_index"] = df.groupby(["season", "round"]).ngroup()
    df["dnf"] = df["position_text"].isin(["R", "D", "W", "N", "E", "F"]).astype(float)
    # Qualifying order is the better signal of pace; use the grid where qualifying is missing.
    df["pace_pos"] = df["quali_position"].fillna(df["grid"].where(df["grid"] > 0))

    by_drv = df.groupby("driver_id", sort=False)
    pts = by_drv[TARGET]
    df["drv_pts_ma3"] = _prior(pts, lambda s: s.rolling(3, min_periods=1).mean())
    df["drv_pts_ma5"] = _prior(pts, lambda s: s.rolling(5, min_periods=1).mean())
    df["drv_pts_ma10"] = _prior(pts, lambda s: s.rolling(10, min_periods=1).mean())
    df["drv_pts_ewm"] = _prior(pts, lambda s: s.ewm(alpha=0.3, min_periods=1).mean())
    df["drv_pos_ma5"] = _prior(by_drv["finish_position"], lambda s: s.rolling(5, min_periods=1).mean())
    df["drv_grid_ma5"] = _prior(by_drv["pace_pos"], lambda s: s.rolling(5, min_periods=1).mean())
    df["drv_dnf_rate10"] = _prior(by_drv["dnf"], lambda s: s.rolling(10, min_periods=1).mean())
    df["drv_n_prior"] = by_drv.cumcount()
    df["drv_season_mean"] = (df.groupby(["driver_id", "season"], sort=False)[TARGET]
                             .transform(lambda s: s.shift(1).expanding().mean()))

    # Constructor form: average of both cars per race, then rolled over prior races.
    team_race = (df.groupby(["constructor_id", "race_index"], sort=True)
                 .agg(team_pts=(TARGET, "mean"), team_grid=("pace_pos", "mean"))
                 .reset_index())
    team_by = team_race.groupby("constructor_id", sort=False)
    team_race["team_pts_ma3"] = _prior(team_by["team_pts"], lambda s: s.rolling(3, min_periods=1).mean())
    team_race["team_pts_ma10"] = _prior(team_by["team_pts"], lambda s: s.rolling(10, min_periods=1).mean())
    team_race["team_grid_ma5"] = _prior(team_by["team_grid"], lambda s: s.rolling(5, min_periods=1).mean())
    df = df.merge(team_race[["constructor_id", "race_index", "team_pts_ma3", "team_pts_ma10",
                             "team_grid_ma5"]], on=["constructor_id", "race_index"], how="left")

    # Track history. Rows are race-ordered, so shift(1) only sees earlier visits.
    drv_circuit = df.groupby(["driver_id", "circuit_id"], sort=False)[TARGET]
    df["circuit_drv_mean"] = _prior(drv_circuit, lambda s: s.expanding().mean())
    df["circuit_drv_n"] = drv_circuit.cumcount()
    team_circuit_race = (df.groupby(["constructor_id", "circuit_id", "race_index"], sort=True)[TARGET]
                         .mean().reset_index())
    tc = team_circuit_race.groupby(["constructor_id", "circuit_id"], sort=False)[TARGET]
    team_circuit_race["circuit_team_mean"] = _prior(tc, lambda s: s.expanding().mean())
    df = df.merge(team_circuit_race[["constructor_id", "circuit_id", "race_index", "circuit_team_mean"]],
                  on=["constructor_id", "circuit_id", "race_index"], how="left")

    df = df.sort_values(["race_index", "driver_id"]).reset_index(drop=True)
    columns = list(dict.fromkeys(ID_COLUMNS + OUTCOME_COLUMNS + FEATURE_COLUMNS))
    return df[columns]
