"""
CSE 163 Final Project: NBA home-court advantage.

Tests for the calculation functions in validity_checks.py. Each test
uses tiny hand-made data whose correct result can be worked out by hand
(or, for the permutation test, by listing every possible split).
Plotting functions and functions that only format text are not tested.
"""

import itertools
import math

import numpy as np
import pandas as pd

from validity_checks import (group_values, verdict, binomial_tests,
                             lag1_autocorrelation, independence_table,
                             fit_trend, quadratic_term_p_value, rq1b_checks,
                             proportion_tests, shape_table, welch_tests,
                             difference_in_means, permutation_tests)


def is_close(a, b, tol=1e-6):
    """
    Returns True if numbers a and b differ by at most tol.
    """
    return abs(a - b) <= tol


# Four games per group. HOME_TEAM_WINS matches the sign of POINT_DIFF.
# - normal: diffs 2, 4, 6, -4 -> 3 wins (0.75), mean 2
# - covid_2021: diffs 5, -3, 1, -3 -> 2 wins (0.50), mean 0
# - bubble: diffs 3, -1, -1, -1 -> 1 win (0.25), mean 0
GROUP_DIFFS = {
    "normal": [2, 4, 6, -4],
    "covid_2021": [5, -3, 1, -3],
    "bubble": [3, -1, -1, -1],
}


def make_period_games():
    """
    Returns a DataFrame of games built from GROUP_DIFFS: the same four
    games for each period, once as regular season (GAME_TYPE 2) and once
    as playoffs (GAME_TYPE 4), so both game types have identical,
    hand-checkable results. HOME_TEAM_WINS is 1 when POINT_DIFF > 0.
    """
    rows = []
    for game_type in [2, 4]:
        for period, diffs in GROUP_DIFFS.items():
            for diff in diffs:
                rows.append({"PERIOD": period, "GAME_TYPE": game_type,
                             "POINT_DIFF": diff,
                             "HOME_TEAM_WINS": int(diff > 0)})
    return pd.DataFrame(rows)


def exact_permutation_p(x, y):
    """
    Independent check for the permutation test: lists every way to
    split the pooled values of x and y into groups of the original
    sizes and returns the share of splits whose |mean difference| is at
    least the observed |mean difference|.
    """
    pooled = list(x) + list(y)
    observed = abs(np.mean(x) - np.mean(y))
    splits = list(itertools.combinations(range(len(pooled)), len(x)))
    extreme = 0
    for chosen in splits:
        first = [pooled[i] for i in chosen]
        second = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        if abs(np.mean(first) - np.mean(second)) >= observed - 1e-9:
            extreme += 1
    return extreme / len(splits)


def test_group_values():
    """
    Picks the bubble playoff point differentials out of
    make_period_games, which must be exactly 3, -1, -1, -1.
    """
    df = make_period_games()

    bubble_diffs = group_values(df, "bubble", 4, "POINT_DIFF")
    normal_wins = group_values(df, "normal", 2, "HOME_TEAM_WINS")

    assert list(bubble_diffs) == [3, -1, -1, -1]
    assert list(normal_wins) == [1, 1, 1, 0]


def test_verdict():
    """
    verdict must turn True into "OK" and False into "NOT MET".
    """
    assert verdict(True) == "OK"
    assert verdict(False) == "NOT MET"


def test_binomial_tests():
    """
    Two seasons:
    - 2010: 4 of 4 home wins. Under H0 (p = 0.5) the only results at
      least this extreme are 4 wins or 0 wins, so p = 2 * (1/2)^4
      = 0.125. Bonferroni (2 tests): 0.25.
    - 2011: 3 of 6 home wins, exactly the expected count, so p = 1.
      Bonferroni: 2, capped at 1.
    """
    season_table = pd.DataFrame({
        "SEASON": [2010, 2011], "games": [4, 6], "home_wins": [4, 3],
        "win_rate": [1.0, 0.5],
    })
    result = binomial_tests(season_table)

    assert list(result["expected_wins_H0"]) == [2.0, 3.0]
    assert is_close(result.loc[0, "p_value"], 0.125)
    assert is_close(result.loc[1, "p_value"], 1.0)
    assert is_close(result.loc[0, "p_bonferroni"], 0.25)
    assert result.loc[1, "p_bonferroni"] == 1.0


def test_lag1_autocorrelation():
    """
    - 1, 0, 1, 0, 1, 0 always flips, so each value is the opposite of
      the next: correlation -1.
    - 1, 1, 0, 0 pairs (1, 1), (1, 0), (0, 0): by hand, covariance
      1/3 and both variances 2/3, so correlation 0.5.
    """
    assert is_close(lag1_autocorrelation(np.array([1, 0, 1, 0, 1, 0])),
                    -1.0)
    assert is_close(lag1_autocorrelation(np.array([1, 1, 0, 0])), 0.5)


def test_independence_table():
    """
    5 games of one season given out of date order. Sorted by date the
    outcomes are 1, 1, 0, 0, 1, whose lag-1 pairs (1, 1), (1, 0),
    (0, 0), (0, 1) have correlation 0. In the given (unsorted) order
    the correlation would be -1/3, so this also checks the sorting.
    The bound is 1.96 / sqrt(5), and 0 is inside it.
    """
    df = pd.DataFrame({
        "SEASON": [2010] * 5,
        "GAME_DATE_EST": pd.to_datetime(["2010-11-03", "2010-11-01",
                                         "2010-11-05", "2010-11-02",
                                         "2010-11-04"]),
        "GAME_ID": [3, 1, 5, 2, 4],
        "HOME_TEAM_WINS": [0, 1, 1, 1, 0],
    })
    result = independence_table(df)

    assert list(result["SEASON"]) == [2010]
    assert is_close(result.loc[0, "lag1_r"], 0.0)
    assert is_close(result.loc[0, "bound_95"], 1.96 / math.sqrt(5))
    assert bool(result.loc[0, "within_bound"])


def test_fit_trend():
    """
    Win rates 0.60, 0.59, 0.58, 0.57 over seasons 2000-2003 fall on a
    straight line with slope -0.01 per season, so OLS must return that
    slope and an intercept of 0.60 + 0.01 * 2000 = 20.6.
    """
    season_table = pd.DataFrame({"SEASON": [2000, 2001, 2002, 2003],
                                 "win_rate": [0.60, 0.59, 0.58, 0.57]})
    result = fit_trend(season_table)

    assert is_close(result.params["SEASON"], -0.01)
    assert is_close(result.params["const"], 20.6, tol=1e-4)


def test_quadratic_term_p_value():
    """
    Seasons 2000-2004 (centered: -2, -1, 0, 1, 2).
    - Straight line plus noise -1, 2, 0, -2, 1 (times 0.01). That noise
      has no linear or squared part, so the fitted squared coefficient
      is exactly 0 and its p-value is 1.
    - Win rate = 0.5 + 0.01 * centered^2 plus the same tiny noise
      (times 0.001): clearly curved, so the p-value is small.
    """
    seasons = [2000, 2001, 2002, 2003, 2004]
    centered = np.array([-2, -1, 0, 1, 2])
    noise = np.array([-1, 2, 0, -2, 1])
    straight = pd.DataFrame({"SEASON": seasons,
                             "win_rate": 0.6 - 0.01 * centered
                             + 0.01 * noise})
    curved = pd.DataFrame({"SEASON": seasons,
                           "win_rate": 0.5 + 0.01 * centered ** 2
                           + 0.001 * noise})

    assert is_close(quadratic_term_p_value(straight), 1.0)
    assert quadratic_term_p_value(curved) < 0.01


def test_rq1b_checks():
    """
    Seasons 2000-2003 follow slope -0.01 (with noise 1, -1, -1, 1 times
    0.001, which has no linear part), and 2004 is an outlier at 0.40.
    The outlier at the end of the range must have the largest Cook's
    distance; dropping it must give back the slope of exactly -0.01.
    The cutoff is 4 / 5 = 0.8.
    """
    season_table = pd.DataFrame({
        "SEASON": [2000, 2001, 2002, 2003, 2004],
        "win_rate": [0.601, 0.589, 0.579, 0.571, 0.40],
    })
    checks = rq1b_checks(season_table)

    assert checks["max_cooks_season"] == 2004
    assert is_close(checks["cooks_cutoff"], 0.8)
    assert is_close(checks["slope_without"], -0.01)
    assert checks["slope"] < -0.01
    assert 0 <= checks["shapiro_p"] <= 1


def test_proportion_tests():
    """
    Uses make_period_games. Regular season, bubble vs normal: 1/4 vs
    3/4 wins. Pooled rate 4/8 = 0.5, standard error
    sqrt(0.5 * 0.5 * (1/4 + 1/4)) = sqrt(0.125), so
    z = (0.25 - 0.75) / sqrt(0.125) = -sqrt(2) and two-sided
    p = 0.1573. covid vs normal: 2/4 vs 3/4, pooled 5/8, so
    z = -0.25 / sqrt(0.625 * 0.375 * 0.5) = -0.7303 and p = 0.4652.
    With only 4 games per group, the smallest of n*p and n*(1-p) is 1,
    so the n*p >= 10 check must fail.
    """
    result = proportion_tests(make_period_games())
    regular = result[result["GAME_TYPE"] == 2].set_index("period")

    assert list(result["period"]) == ["covid_2021", "bubble"] * 2
    assert list(regular["rate_period"]) == [0.5, 0.25]
    assert list(regular["rate_normal"]) == [0.75, 0.75]
    assert is_close(regular.loc["bubble", "z"], -math.sqrt(2))
    assert is_close(regular.loc["bubble", "p_value"], 0.157299, 1e-5)
    assert is_close(regular.loc["covid_2021", "z"], -0.730297, 1e-5)
    assert is_close(regular.loc["covid_2021", "p_value"], 0.465209, 1e-5)
    assert list(result["min_np"]) == [1, 1, 1, 1]
    assert not result["np_check"].any()


def test_shape_table():
    """
    Uses make_period_games. normal diffs 2, 4, 6, -4: mean 2,
    deviations 0, 2, 4, -6, so sample sd sqrt(56 / 3), skewness
    (-36) / 14^1.5 and excess kurtosis 392 / 14^2 - 3 = -1.
    bubble diffs 3, -1, -1, -1: mean 0, sd 2, skewness 6 / 3^1.5 and
    excess kurtosis 21 / 3^2 - 3 = -2/3.
    """
    result = shape_table(make_period_games())
    regular = result[result["GAME_TYPE"] == 2].set_index("period")

    assert list(result["n"]) == [4] * 6
    normal = regular.loc["normal"]
    assert normal["mean"] == 2.0
    assert is_close(normal["sd"], math.sqrt(56 / 3))
    assert is_close(normal["skewness"], -36 / 14 ** 1.5)
    assert is_close(normal["excess_kurtosis"], -1.0)
    bubble = regular.loc["bubble"]
    assert bubble["mean"] == 0.0
    assert is_close(bubble["sd"], 2.0)
    assert is_close(bubble["skewness"], 6 / 3 ** 1.5)
    assert is_close(bubble["excess_kurtosis"], -2 / 3)
    assert 0 <= bubble["shapiro_p"] <= 1


def test_welch_tests():
    """
    Uses make_period_games. bubble vs normal: means 0 and 2, variances
    4 and 56/3, so t = (0 - 2) / sqrt(4/4 + (56/3)/4) = -0.8402 and the
    sd ratio is 2 / sqrt(56/3). covid vs normal: covid variance 44/3,
    so t = -2 / sqrt((44/3)/4 + (56/3)/4) = -2 / sqrt(100/12) = -0.6928.
    """
    result = welch_tests(make_period_games())
    regular = result[result["GAME_TYPE"] == 2].set_index("period")

    bubble = regular.loc["bubble"]
    assert bubble["mean_diff"] == -2.0
    assert is_close(bubble["welch_t"], -2 / math.sqrt(1 + 56 / 12))
    assert is_close(bubble["sd_ratio"], 2 / math.sqrt(56 / 3))
    covid = regular.loc["covid_2021"]
    assert is_close(covid["welch_t"], -2 / math.sqrt(100 / 12))
    assert 0 <= bubble["levene_p"] <= 1
    assert 0 <= bubble["welch_p"] <= 1


def test_difference_in_means():
    """
    mean(1, 2, 3) - mean(1, 1) = 2 - 1 = 1. For 2-D input each row is
    its own sample: rows (1, 3) and (5, 7) minus rows (0, 0) and (2, 2)
    give 2 - 0 = 2 and 6 - 2 = 4.
    """
    assert difference_in_means(np.array([1, 2, 3]), np.array([1, 1]),
                               axis=0) == 1.0
    x = np.array([[1, 3], [5, 7]])
    y = np.array([[0, 0], [2, 2]])
    assert list(difference_in_means(x, y, axis=1)) == [2.0, 4.0]


def test_permutation_tests():
    """
    Uses make_period_games. With 4 + 4 games there are only C(8, 4) = 70
    ways to split the pooled values, fewer than the number of shuffles,
    so the test is exact and must match listing all 70 splits.
    For HOME_TEAM_WINS, bubble vs normal (1/4 vs 3/4), the splits with
    a win-rate gap of at least 0.5 put 0, 1, 3 or 4 of the 4 wins in
    the first group: 1 + 16 + 16 + 1 = 34 splits, so p = 34/70.
    """
    df = make_period_games()
    result = permutation_tests(df)
    regular = result[result["GAME_TYPE"] == 2].set_index(["period",
                                                          "outcome"])

    wins = regular.loc[("bubble", "HOME_TEAM_WINS")]
    assert is_close(wins["observed_diff"], -0.5)
    assert is_close(wins["perm_p"], 34 / 70)

    diffs = regular.loc[("bubble", "POINT_DIFF")]
    expected = exact_permutation_p(GROUP_DIFFS["bubble"],
                                   GROUP_DIFFS["normal"])
    assert is_close(diffs["observed_diff"], -2.0)
    assert is_close(diffs["perm_p"], expected)
    assert len(result) == 8


def run_all_tests():
    """
    Runs every test function in this file and prints a summary.
    """
    test_group_values()
    test_verdict()
    test_binomial_tests()
    test_lag1_autocorrelation()
    test_independence_table()
    test_fit_trend()
    test_quadratic_term_p_value()
    test_rq1b_checks()
    test_proportion_tests()
    test_shape_table()
    test_welch_tests()
    test_difference_in_means()
    test_permutation_tests()
    print("All validity_checks tests passed!")


if __name__ == "__main__":
    run_all_tests()
