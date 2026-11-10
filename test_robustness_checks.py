"""
CSE 163 Final Project: NBA home-court advantage.

Tests for the calculation functions in robustness_checks.py. Each test
uses tiny hand-made data whose correct result can be worked out by hand.
The plotting function and functions that only format text or combine
already-tested functions (series_vs_game_table, main) are not tested.
"""

import math

import numpy as np
import pandas as pd

from robustness_checks import (logistic_trend, linearity_lr_test,
                               logistic_summary, add_series_id,
                               cluster_permutation_test,
                               series_variance_ratio, adjust_p_values,
                               multiple_comparisons_table,
                               proportion_power, min_detectable_rate_gap,
                               mean_power, min_detectable_mean_gap,
                               games_needed, power_table)
from test_validity_checks import make_period_games

# A two-sided normal test whose expected z equals the critical value
# (1.96, rounded) has power P(Z > 0) + P(Z < -3.92) = 0.5 + 0.00004.
# The exact critical value is 1.95996, so results match to about 1e-4.
POWER_AT_CRITICAL = 0.5 + 0.0000443
POWER_TOL = 1e-4


def is_close(a, b, tol=1e-6):
    """
    Returns True if numbers a and b differ by at most tol.
    """
    return abs(a - b) <= tol


def make_two_season_games():
    """
    Returns 6 games: SEASON 2000 with 1 home win in 2 games (rate 0.5)
    and SEASON 2001 with 3 home wins in 4 games (rate 0.75).
    """
    return pd.DataFrame({
        "SEASON": [2000, 2000, 2001, 2001, 2001, 2001],
        "HOME_TEAM_WINS": [1, 0, 1, 1, 0, 1],
    })


def make_series_games():
    """
    Returns 8 games in 4 series. Series s1 (bubble): home team wins
    both games; s2 (bubble): wins 1 of 2; s3 (normal): loses all 3;
    s4 (normal): wins its 1 game. Bubble games: 3 wins in 4; normal
    games: 1 win in 4.
    """
    return pd.DataFrame({
        "GAME_ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "SERIES_ID": ["s1", "s1", "s2", "s2", "s3", "s3", "s3", "s4"],
        "PERIOD": ["bubble"] * 4 + ["normal"] * 4,
        "HOME_TEAM_WINS": [1, 1, 1, 0, 0, 0, 0, 1],
    })


def test_logistic_trend():
    """
    With two seasons the logistic model fits both rates exactly:
    intercept = logit(0.5) = 0 and slope = logit(0.75) - logit(0.5)
    = log(3) per season.
    """
    result = logistic_trend(make_two_season_games())

    assert is_close(result.params["Intercept"], 0.0, 1e-5)
    assert is_close(result.params["SEASON_FROM_START"], math.log(3), 1e-5)


def test_logistic_summary():
    """
    Same fit as test_logistic_trend: the odds ratio per decade is
    exp(10 * log 3) = 3^10, and the predicted rates in the first and
    last season are the observed 0.5 and 0.75.
    """
    summary = logistic_summary(logistic_trend(make_two_season_games()))

    assert is_close(summary["slope"], math.log(3), 1e-5)
    assert is_close(summary["odds_ratio_decade"], 3 ** 10, 1e-1)
    assert is_close(summary["pred_first"], 0.5, 1e-5)
    assert is_close(summary["pred_last"], 0.75, 1e-5)
    assert 0 < summary["p_value"] < 1


def test_linearity_lr_test():
    """
    Rates 0.25, 0.5, 0.75 over three seasons have log-odds -log 3, 0,
    log 3, which lie exactly on a straight line. The straight-line
    model then fits as well as the one-effect-per-season model, so the
    LR statistic is 0 on 3 - 2 = 1 degree of freedom and p = 1.
    """
    games = pd.DataFrame({
        "SEASON": [2000] * 4 + [2001] * 4 + [2002] * 4,
        "HOME_TEAM_WINS": [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    })
    lr, df, p = linearity_lr_test(games)

    assert is_close(lr, 0.0, 1e-5)
    assert df == 1
    assert is_close(p, 1.0, 1e-3)


def test_add_series_id():
    """
    Games 1 and 2 are the same two teams in the same season with home
    and away swapped, so they share a SERIES_ID. Game 3 is the same
    pair in another season and game 4 a different pair, so both get
    their own IDs. The smaller team ID always comes first.
    """
    games = pd.DataFrame({"SEASON": [2019, 2019, 2020, 2019],
                          "HOME_TEAM_ID": [1, 2, 1, 3],
                          "VISITOR_TEAM_ID": [2, 1, 2, 1]})
    result = add_series_id(games)

    assert list(result["SERIES_ID"]) == ["2019_1_2", "2019_1_2",
                                         "2020_1_2", "2019_1_3"]
    assert "SERIES_ID" not in games.columns


def test_cluster_permutation_test():
    """
    Uses make_series_games. Observed: 3/4 - 1/4 = 0.5.
    Series level: the 6 ways to choose 2 of the 4 series as "bubble"
    give differences 0.5, -4/15, 0.8, -0.8, 4/15, -0.5. Four of the six
    are at least 0.5 in size, so p = 4/6.
    Game level: the 70 ways to choose 4 of the 8 games give
    p = 34/70 (see test_validity_checks.test_permutation_tests). The
    p-values differ because series keep their games together.
    """
    games = make_series_games()
    series = cluster_permutation_test(games, "bubble", "HOME_TEAM_WINS",
                                      "SERIES_ID")
    single = cluster_permutation_test(games, "bubble", "HOME_TEAM_WINS",
                                      "GAME_ID")

    assert is_close(series["observed_diff"], 0.5)
    assert is_close(series["p_value"], 4 / 6)
    assert series["units_period"] == 2
    assert series["units_base"] == 2
    assert is_close(single["observed_diff"], 0.5)
    assert is_close(single["p_value"], 34 / 70)


def test_series_variance_ratio():
    """
    Overall rate 0.5 in both cases, 2 games per series, so independent
    games would give variance 0.5 * 0.5 / 2 = 0.125.
    - Series results (1, 1) and (0, 0): series means 1 and 0, variance
      0.25, ratio 2 (games in a series move together).
    - Series results (1, 0) and (0, 1): both means 0.5, variance 0,
      ratio 0.
    """
    together = pd.DataFrame({"SERIES_ID": ["a", "a", "b", "b"],
                             "HOME_TEAM_WINS": [1, 1, 0, 0]})
    split = pd.DataFrame({"SERIES_ID": ["a", "a", "b", "b"],
                          "HOME_TEAM_WINS": [1, 0, 0, 1]})

    assert is_close(series_variance_ratio(together), 2.0)
    assert is_close(series_variance_ratio(split), 0.0)


def test_adjust_p_values():
    """
    p = 0.01, 0.04, 0.03, 0.5. Bonferroni multiplies by 4 (capped at
    1): 0.04, 0.16, 0.12, 1. Holm sorts them (0.01, 0.03, 0.04, 0.5),
    multiplies by 4, 3, 2, 1 to get 0.04, 0.09, 0.08, 0.5, then makes
    them non-decreasing (0.08 -> 0.09) and puts them back in order:
    0.04, 0.09, 0.09, 0.5.
    """
    holm, bonferroni = adjust_p_values([0.01, 0.04, 0.03, 0.5])

    for got, want in zip(holm, [0.04, 0.09, 0.09, 0.5]):
        assert is_close(got, want)
    for got, want in zip(bonferroni, [0.04, 0.16, 0.12, 1.0]):
        assert is_close(got, want)


def test_multiple_comparisons_table():
    """
    Two families of two tests each. Win rate p = 0.01, 0.04 -> Holm
    0.02, 0.04 (both significant). POINT_DIFF p = 0.03, 0.2 -> Holm
    0.06, 0.2 (neither significant).
    """
    prop_table = pd.DataFrame({"GAME_TYPE": [2, 4],
                               "period": ["bubble", "bubble"],
                               "p_value": [0.01, 0.04]})
    welch_table = pd.DataFrame({"GAME_TYPE": [2, 4],
                                "period": ["bubble", "bubble"],
                                "welch_p": [0.03, 0.2]})
    result = multiple_comparisons_table(prop_table, welch_table)

    assert list(result["p_raw"]) == [0.01, 0.04, 0.03, 0.2]
    for got, want in zip(result["p_holm"], [0.02, 0.04, 0.06, 0.2]):
        assert is_close(got, want)
    assert list(result["significant_holm"]) == [True, True, False, False]


def test_proportion_power():
    """
    - Equal rates: no effect, so power equals alpha = 0.05.
    - Rates 0.5 and sin^2(pi/4 + 0.1) differ by exactly h = 0.2 on
      Cohen's h scale. With 192.08 games per group,
      h * sqrt(n/2) = 0.2 * 9.8 = 1.96, exactly the critical value, so
      power = 0.5 + P(Z < -3.92).
    """
    other = math.sin(math.pi / 4 + 0.1) ** 2

    assert is_close(proportion_power(0.5, 0.5, 100, 100), 0.05)
    assert is_close(proportion_power(other, 0.5, 192.08, 192.08),
                    POWER_AT_CRITICAL, POWER_TOL)


def test_min_detectable_rate_gap():
    """
    The test must have exactly 80% power at the returned gap (round
    trip through proportion_power), and more games must mean a smaller
    detectable gap.
    """
    gap = min_detectable_rate_gap(0.6, 200, 2000)

    assert is_close(proportion_power(0.6 - gap, 0.6, 200, 2000), 0.8, 1e-4)
    assert min_detectable_rate_gap(0.6, 800, 2000) < gap


def test_mean_power():
    """
    - Mean difference 0: power equals alpha = 0.05.
    - Difference 2 with sd 10 is d = 0.2; with 192.08 games per group,
      d * sqrt(n/2) = 1.96, so power = 0.5 + P(Z < -3.92), as in
      test_proportion_power.
    """
    assert is_close(mean_power(0.0, 10, 100, 100), 0.05)
    assert is_close(mean_power(2.0, 10, 192.08, 192.08), POWER_AT_CRITICAL,
                    POWER_TOL)


def test_min_detectable_mean_gap():
    """
    With 200 games per group and sd 10, 80% power needs
    d * sqrt(200/2) = 1.96 + 0.8416 (the other tail is negligible), so
    d = 0.28016 and the gap is 10 * d = 2.8016 points.
    """
    assert is_close(min_detectable_mean_gap(10, 200, 200), 2.8016, 1e-3)


def test_games_needed():
    """
    - Power that grows as n / 100 reaches 0.8 at exactly 80 games.
    - Power stuck at 0.5 never reaches 0.8, so the answer is infinity.
    """
    assert games_needed(lambda n1, n2: n1 / 100, 1000) == 80
    assert games_needed(lambda n1, n2: 0.5, 1000) == float("inf")


def test_power_table():
    """
    Uses make_period_games from test_validity_checks (4 games per
    group). Regular season, bubble vs normal: win rate 0.25 - 0.75 =
    -0.5 and POINT_DIFF mean 0 - 2 = -2; covid vs normal: 0.5 - 0.75 =
    -0.25 and 0 - 2 = -2. Every power must be between 0.05 and 1.
    """
    result = power_table(make_period_games())
    regular = result[result["GAME_TYPE"] == 2].set_index(["period",
                                                          "outcome"])

    assert len(result) == 8
    assert list(result["n_period"]) == [4] * 8
    assert is_close(regular.loc[("bubble", "win rate"), "observed_diff"],
                    -0.5)
    assert is_close(regular.loc[("bubble", "POINT_DIFF"),
                                "observed_diff"], -2.0)
    assert is_close(regular.loc[("covid_2021", "win rate"),
                                "observed_diff"], -0.25)
    assert result["power"].between(0.05, 1).all()
    assert np.all(result["min_detectable"] > 0)


def run_all_tests():
    """
    Runs every test function in this file and prints a summary.
    """
    test_logistic_trend()
    test_logistic_summary()
    test_linearity_lr_test()
    test_add_series_id()
    test_cluster_permutation_test()
    test_series_variance_ratio()
    test_adjust_p_values()
    test_multiple_comparisons_table()
    test_proportion_power()
    test_min_detectable_rate_gap()
    test_mean_power()
    test_min_detectable_mean_gap()
    test_games_needed()
    test_power_table()
    print("All robustness_checks tests passed!")


if __name__ == "__main__":
    run_all_tests()
