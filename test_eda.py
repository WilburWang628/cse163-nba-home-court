"""
CSE 163 Final Project: NBA home-court advantage.

Tests for the calculation functions in eda_games.py. Each test builds a
tiny hand-made DataFrame whose correct result can be worked out by hand.
Plotting functions and functions that only format text are not tested.
"""

import math

import pandas as pd

from eda_games import (missing_table, numeric_summary, win_rate_table,
                       period_win_rates, point_diff_table)


def is_close(a, b, tol=1e-6):
    """
    Returns True if numbers a and b differ by at most tol.
    """
    return abs(a - b) <= tol


def test_missing_table():
    """
    4 rows: column a has no missing values, b has 1 (25%) and c has 2
    (50%). Only b and c should be listed, with those counts and
    percents.
    """
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [1, math.nan, 3, 4],
        "c": [math.nan, math.nan, 3, 4],
    })
    result = missing_table(df)

    assert list(result.index) == ["b", "c"]
    assert list(result["missing"]) == [1, 2]
    assert list(result["percent"]) == [25.0, 50.0]


def test_numeric_summary():
    """
    x = 1, 2, 3, 4, 5 has mean 3, sample std sqrt(2.5), min 1, Q1 2,
    median 3, Q3 4 and max 5. y = 10, 10, 10, 10, 10 has std 0 and every
    other statistic equal to 10.
    """
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5],
                       "y": [10, 10, 10, 10, 10]})
    result = numeric_summary(df, ["x", "y"])

    assert list(result.columns) == ["mean", "std", "min", "Q1", "median",
                                    "Q3", "max"]
    assert list(result.index) == ["x", "y"]
    x = result.loc["x"]
    assert list(x.drop("std")) == [3.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert is_close(x["std"], math.sqrt(2.5))
    assert result.loc["y", "std"] == 0.0
    assert result.loc["y", "median"] == 10.0


def test_win_rate_table():
    """
    Team group A: home team wins 3 of 4 games, so the win rate is 0.75
    and the 95% CI is 0.75 +/- 1.96 * sqrt(0.75 * 0.25 / 4). Group B:
    1 game, home team loses, so the rate is 0 and the CI has width 0.
    """
    df = pd.DataFrame({
        "group": ["A", "A", "A", "A", "B"],
        "HOME_TEAM_WINS": [1, 1, 0, 1, 0],
    })
    result = win_rate_table(df, ["group"])

    assert list(result["group"]) == ["A", "B"]
    assert list(result["games"]) == [4, 1]
    assert list(result["home_wins"]) == [3, 0]
    assert list(result["win_rate"]) == [0.75, 0.0]

    margin = 1.96 * math.sqrt(0.75 * 0.25 / 4)
    assert is_close(result.loc[0, "ci_low"], 0.75 - margin)
    assert is_close(result.loc[0, "ci_high"], 0.75 + margin)
    assert result.loc[1, "ci_low"] == 0.0
    assert result.loc[1, "ci_high"] == 0.0


def test_period_win_rates():
    """
    6 games: the "exclude" game must be dropped, and the remaining
    groups must come out in the order normal, covid_2021, bubble, with
    regular season (2) before playoffs (4) inside normal.
    - normal, 2: wins 1, 0 -> 0.5
    - normal, 4: win 1 -> 1.0
    - covid_2021, 2: win 1 -> 1.0
    - bubble, 2: loss 0 -> 0.0
    """
    df = pd.DataFrame({
        "PERIOD": ["bubble", "exclude", "normal", "covid_2021",
                   "normal", "normal"],
        "GAME_TYPE": [2, 2, 4, 2, 2, 2],
        "HOME_TEAM_WINS": [0, 1, 1, 1, 1, 0],
    })
    result = period_win_rates(df)

    assert list(result["PERIOD"]) == ["normal", "normal", "covid_2021",
                                      "bubble"]
    assert list(result["GAME_TYPE"]) == [2, 4, 2, 2]
    assert list(result["games"]) == [2, 1, 1, 1]
    assert list(result["win_rate"]) == [0.5, 1.0, 1.0, 0.0]


def test_point_diff_table():
    """
    6 games, regular season and playoffs pooled:
    - normal: 10 and -2 -> mean 4, std sqrt(72) (deviations +6, -6),
      median 4
    - covid_2021: 3 -> mean 3, std undefined (one value), median 3
    - bubble: 1 and 3 -> mean 2, std sqrt(2), median 2
    - exclude: 99 -> must be ignored
    """
    df = pd.DataFrame({
        "PERIOD": ["normal", "normal", "covid_2021", "bubble",
                   "bubble", "exclude"],
        "GAME_TYPE": [2, 4, 2, 2, 4, 2],
        "POINT_DIFF": [10, -2, 3, 1, 3, 99],
    })
    result = point_diff_table(df)

    assert list(result.index) == ["normal", "covid_2021", "bubble"]
    assert list(result["count"]) == [2, 1, 2]
    assert list(result["mean"]) == [4.0, 3.0, 2.0]
    assert list(result["median"]) == [4.0, 3.0, 2.0]
    assert is_close(result.loc["normal", "std"], math.sqrt(72))
    assert math.isnan(result.loc["covid_2021", "std"])
    assert is_close(result.loc["bubble", "std"], math.sqrt(2))


def run_all_tests():
    """
    Runs every test function in this file and prints a summary.
    """
    test_missing_table()
    test_numeric_summary()
    test_win_rate_table()
    test_period_win_rates()
    test_point_diff_table()
    print("All eda_games tests passed!")


if __name__ == "__main__":
    run_all_tests()
