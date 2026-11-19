"""
CSE 163 Final Project: NBA home-court advantage.

Tests for the calculation functions in eda_ranking.py (RQ4). Each test
builds a tiny hand-made DataFrame whose correct result can be worked
out by hand. Plotting functions and functions that only format text
are not tested.
"""

import math

import pandas as pd

from eda_ranking import (final_standings, add_hca, unusual_game_counts,
                         merge_teams, team_hca_table, season_hca_table,
                         hca_by_team, split_altitude, team_residuals,
                         lag1_correlation, eta_squared, sampling_sd,
                         pair_permutation_p)


def is_close(a, b, tol=1e-9):
    """
    Returns True if numbers a and b differ by at most tol.
    """
    return abs(a - b) <= tol


def small_hca_table():
    """
    Returns 4 team-seasons with known HCA values: team A has 0.1 and
    0.3 (mean 0.2), team B has 0.0 and 0.2 (mean 0.1).
    """
    return pd.DataFrame({
        "TEAM_NAME": ["A", "A", "B", "B"],
        "SEASON": [2010, 2011, 2010, 2011],
        "HCA": [0.1, 0.3, 0.0, 0.2],
        "HOME_PCT": [0.6, 0.7, 0.5, 0.6],
        "ROAD_PCT": [0.5, 0.4, 0.5, 0.4],
    })


def test_final_standings():
    """
    Team 1 has 3 dates in 2010, given out of order; only the latest
    (2011-04-13, HOME_W 3) may remain. Team 2 has one 2010 row, team 1
    one 2011 row, and a 2019 row that must be dropped because 2019 is
    not in the season list. Expected: 3 rows.
    """
    df = pd.DataFrame({
        "TEAM_ID": [1, 1, 1, 2, 1, 1],
        "SEASON": [2010, 2010, 2010, 2010, 2011, 2019],
        "STANDINGSDATE": pd.to_datetime([
            "2011-01-01", "2011-04-13", "2010-11-01",
            "2011-04-13", "2012-04-26", "2020-03-11"]),
        "HOME_W": [2, 3, 1, 9, 7, 5],
    })
    result = final_standings(df, [2010, 2011])

    assert len(result) == 3
    assert list(result["SEASON"]) == [2010, 2010, 2011]
    assert list(result["TEAM_ID"]) == [1, 2, 1]
    assert list(result["HOME_W"]) == [3, 9, 7]


def test_add_hca():
    """
    Row 1: home 30-11, road 20-21, so HCA = 30/41 - 20/41 = 10/41 and
    82 games. Row 2 (a 66-game season): home 20-13, road 11-22, so
    HCA = 20/33 - 11/33 = 9/33.
    """
    df = pd.DataFrame({"HOME_W": [30, 20], "HOME_L": [11, 13],
                       "ROAD_W": [20, 11], "ROAD_L": [21, 22]})
    result = add_hca(df)

    assert list(result["GAMES_PLAYED"]) == [82, 66]
    assert is_close(result.loc[0, "HOME_PCT"], 30 / 41)
    assert is_close(result.loc[0, "ROAD_PCT"], 20 / 41)
    assert is_close(result.loc[0, "HCA"], 10 / 41)
    assert is_close(result.loc[1, "HCA"], 9 / 33)
    assert "HCA" not in df.columns


def test_unusual_game_counts():
    """
    82 games is normal except in 2011, where 66 is normal. So only the
    81-game 2010 row and the 82-game 2011 row should be flagged.
    """
    df = pd.DataFrame({"SEASON": [2010, 2010, 2011, 2011],
                       "GAMES_PLAYED": [82, 81, 66, 82]})
    result = unusual_game_counts(df)

    assert list(result.index) == [1, 3]


def test_merge_teams():
    """
    TEAM_IDs 1 and 2 are in teams, 3 is not. The merge must keep all
    3 rows, fill in the two names, and report [3] as unmatched.
    """
    standings = pd.DataFrame({"TEAM_ID": [1, 2, 3],
                              "HCA": [0.1, 0.2, 0.3]})
    teams = pd.DataFrame({"TEAM_ID": [1, 2],
                          "TEAM_NAME": ["Denver Nuggets", "Utah Jazz"]})
    merged, unmatched = merge_teams(standings, teams)

    assert len(merged) == 3
    assert list(merged["TEAM_NAME"][:2]) == ["Denver Nuggets",
                                             "Utah Jazz"]
    assert pd.isna(merged.loc[2, "TEAM_NAME"])
    assert unmatched == [3]


def test_team_and_season_tables():
    """
    Team means: A = (0.1 + 0.3) / 2 = 0.2, B = (0.0 + 0.2) / 2 = 0.1, so
    A comes first. Season means: 2010 = (0.1 + 0.0) / 2 = 0.05 and
    2011 = (0.3 + 0.2) / 2 = 0.25.
    """
    df = small_hca_table()
    teams = team_hca_table(df)
    seasons = season_hca_table(df)

    assert list(teams.index) == ["A", "B"]
    assert list(teams["seasons"]) == [2, 2]
    assert is_close(teams.loc["A", "mean_hca"], 0.2)
    assert is_close(teams.loc["B", "mean_hca"], 0.1)
    assert is_close(seasons.loc[2010, "mean_hca"], 0.05)
    assert is_close(seasons.loc[2011, "mean_hca"], 0.25)
    assert list(seasons["teams"]) == [2, 2]


def test_hca_by_team_and_split():
    """
    hca_by_team must group the HCA values by team. split_altitude with
    altitude team ["A"] must return A's two values in the first group
    and B's two values in the second.
    """
    df = small_hca_table()
    groups = hca_by_team(df)
    altitude, other = split_altitude(df, ["A"])

    assert groups == {"A": [0.1, 0.3], "B": [0.0, 0.2]}
    assert list(altitude) == [0.1, 0.3]
    assert list(other) == [0.0, 0.2]


def test_team_residuals():
    """
    Residual = HCA - team mean: A gives 0.1 - 0.2 and 0.3 - 0.2, B gives
    0.0 - 0.1 and 0.2 - 0.1, i.e. -0.1, 0.1, -0.1, 0.1.
    """
    result = team_residuals(small_hca_table())
    expected = [-0.1, 0.1, -0.1, 0.1]

    for actual, want in zip(result, expected):
        assert is_close(actual, want)


def test_lag1_correlation():
    """
    One team, seasons given out of order, HCA by season 0, 0, 3, 3.
    Mean 1.5, residuals -1.5, -1.5, 1.5, 1.5. Pairs (this season, next
    season): (-1.5, -1.5), (-1.5, 1.5), (1.5, 1.5). Centering x and y
    gives (-1, -2), (-1, 1), (2, 1): sum of products 3, sums of squares
    6 and 6, so r = 3 / 6 = 0.5.
    """
    df = pd.DataFrame({"TEAM_NAME": ["A"] * 4,
                       "SEASON": [2012, 2010, 2013, 2011],
                       "HCA": [3.0, 0.0, 3.0, 0.0]})

    assert is_close(lag1_correlation(df), 0.5)


def test_eta_squared():
    """
    Groups [1, 3] and [5, 7]. Grand mean 4. Total sum of squares
    9 + 1 + 1 + 9 = 20. Group means 2 and 6, so between-group sum of
    squares 2 * 4 + 2 * 4 = 16. eta^2 = 16 / 20 = 0.8.
    """
    assert is_close(eta_squared({"a": [1, 3], "b": [5, 7]}), 0.8)


def test_sampling_sd():
    """
    Row 1: home 1-1 and road 1-1, so each rate is 0.5 out of 2 games and
    the HCA variance is 0.25 / 2 + 0.25 / 2 = 0.25. Row 2: home 2-0 and
    road 0-2, rates 1 and 0, variance 0. Average variance 0.125, so the
    SD is sqrt(0.125).
    """
    df = add_hca(pd.DataFrame({"HOME_W": [1, 2], "HOME_L": [1, 0],
                               "ROAD_W": [1, 0], "ROAD_L": [1, 2]}))

    assert is_close(sampling_sd(df), math.sqrt(0.125))


def test_pair_permutation_p():
    """
    Team means A 0.4, B 0.3, C 0.2, D 0.1 give 6 pairs with averages
    AB 0.35, AC 0.30, AD 0.25, BC 0.25, BD 0.20, CD 0.15.
    Chosen A + B (0.35): only AB is >= 0.35, so p = 1/6.
    Chosen B + C (0.25): AB, AC, AD, BC are >= 0.25, so p = 4/6.
    """
    means = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})

    assert is_close(pair_permutation_p(means, ["A", "B"]), 1 / 6)
    assert is_close(pair_permutation_p(means, ["B", "C"]), 4 / 6)


def run_all_tests():
    """
    Runs every test function in this file and prints a summary.
    """
    test_final_standings()
    test_add_hca()
    test_unusual_game_counts()
    test_merge_teams()
    test_team_and_season_tables()
    test_hca_by_team_and_split()
    test_team_residuals()
    test_lag1_correlation()
    test_eta_squared()
    test_sampling_sd()
    test_pair_permutation_p()
    print("All eda_ranking tests passed!")


if __name__ == "__main__":
    run_all_tests()
