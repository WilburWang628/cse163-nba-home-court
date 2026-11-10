"""
CSE 163 Final Project: NBA home-court advantage.

Robustness checks that follow up on the problems found by
validity_checks.py:

1. RQ1 B: game-level logistic regression of HOME_TEAM_WINS on SEASON,
   instead of OLS on 16 season averages (uses every game, weights each
   season by its number of games), with a linearity test and a
   sensitivity check without the influential 2021 season.
2. Playoff games are not independent (the same two teams play 4-7 games
   in a series). Series-level permutation tests shuffle whole series
   between periods, and are compared with game-level permutation tests.
3. Multiple comparisons: Holm and Bonferroni adjustments for the four
   RQ2 comparisons of each outcome.
4. Power: how likely each RQ2 comparison was to detect the observed
   difference, the smallest difference it could detect with 80% power,
   and how many games would be needed.

Writes results/robustness_checks.txt and figures/fig6.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from clean import load_games, clean_games
from eda_games import (set_plot_style, win_rate_table, PERIOD_COLORS,
                       GAME_TYPE_NAMES, MUTED_COLOR, TEXT_COLOR)
from validity_checks import (group_values, proportion_tests, welch_tests,
                             BASE_PERIOD, COMPARISON_PERIODS, ALPHA,
                             N_PERMUTATIONS, RANDOM_SEED, REGULAR_SEASON)

RAW_PATH = "data/games.csv"
FIGURE_PATH = "figures/fig6_rq1_logistic_trend.png"
RESULTS_PATH = "results/robustness_checks.txt"

PLAYOFFS = 4
TARGET_POWER = 0.8
# Largest period size tried when searching for the games needed.
MAX_GAMES = 1e6
INFLUENTIAL_SEASON = 2021
OUTCOMES = ["HOME_TEAM_WINS", "POINT_DIFF"]


# ------------------------------------------------- 1. logistic regression


def logistic_trend(games):
    """
    Fits a logistic regression of HOME_TEAM_WINS on SEASON (counted from
    the first season in games) using one row per game. Returns the
    statsmodels result; the SEASON_FROM_START coefficient is the change
    in log-odds of a home win per season.
    """
    df = games[["HOME_TEAM_WINS", "SEASON"]].copy()
    df["SEASON_FROM_START"] = df["SEASON"] - df["SEASON"].min()
    return smf.logit("HOME_TEAM_WINS ~ SEASON_FROM_START",
                     data=df).fit(disp=0)


def linearity_lr_test(games):
    """
    Compares the logistic model with a straight-line SEASON effect to a
    model with a separate effect for every season (SEASON as a
    category), using a likelihood-ratio test. Returns (LR statistic,
    degrees of freedom, p-value). A small p-value means the season
    effect is not a straight line on the log-odds scale.
    """
    linear = logistic_trend(games)
    by_season = smf.logit("HOME_TEAM_WINS ~ C(SEASON)",
                          data=games).fit(disp=0)
    lr = max(2 * (by_season.llf - linear.llf), 0.0)
    df = games["SEASON"].nunique() - 2
    return lr, df, stats.chi2.sf(lr, df)


def logistic_summary(result):
    """
    Returns a dict with the per-season log-odds slope, its p-value, the
    odds ratio per decade, and the model's predicted home win rate in
    the first and last season of the data.
    """
    slope = result.params["SEASON_FROM_START"]
    seasons = result.model.data.frame["SEASON_FROM_START"]
    first = pd.DataFrame({"SEASON_FROM_START": [seasons.min()]})
    last = pd.DataFrame({"SEASON_FROM_START": [seasons.max()]})
    return {
        "slope": slope,
        "p_value": result.pvalues["SEASON_FROM_START"],
        "odds_ratio_decade": np.exp(10 * slope),
        "pred_first": float(result.predict(first).iloc[0]),
        "pred_last": float(result.predict(last).iloc[0]),
    }


def logistic_section(full, without, lr_test, n_games):
    """
    Returns text reporting the logistic regression with and without the
    influential season and the linearity test.
    """
    lr, df, lr_p = lr_test
    lines = [
        "1. RQ1 B - logistic regression (one row per game)",
        f"Model: logit P(home win) ~ SEASON, normal regular-season games "
        f"(n = {n_games}).",
        "Why: OLS on 16 season averages throws away the game-level data "
        "and gives the short 2011 season (990 games) the same weight as "
        "the others. Logistic regression uses every game and models a "
        "0/1 outcome directly.",
    ]
    for label, s in [("All normal seasons", full),
                     (f"Without SEASON {INFLUENTIAL_SEASON}", without)]:
        lines.append(
            f"{label}: slope = {s['slope']:.5f} log-odds per season, "
            f"p = {s['p_value']:.4g}, odds ratio per decade = "
            f"{s['odds_ratio_decade']:.3f}; predicted home win rate "
            f"{s['pred_first']:.3f} (first season) -> "
            f"{s['pred_last']:.3f} (last season)")
    lines.append(
        "Linearity check (likelihood-ratio test, straight line vs one "
        f"effect per season): LR = {lr:.3f}, df = {df}, p = {lr_p:.4g}. "
        + ("No evidence against a straight line."
           if lr_p > ALPHA else
           "Seasons vary around the line more than chance allows."))
    lines.append("Assumption: games are independent (same caveat as "
                 "validity_checks.txt).")
    return "\n".join(lines)


def plot_logistic_trend(games, full_result, without_result, path):
    """
    Saves a figure to path showing each normal regular season's observed
    home win rate (dots) and the fitted logistic trend with and without
    the influential season (lines).
    """
    observed = win_rate_table(games, ["SEASON"])
    first = games["SEASON"].min()
    seasons = np.arange(first, games["SEASON"].max() + 1)
    grid = pd.DataFrame({"SEASON_FROM_START": seasons - first})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(observed["SEASON"], observed["win_rate"],
               color=PERIOD_COLORS[BASE_PERIOD], s=40, zorder=3,
               label="Observed home win rate")
    ax.plot(seasons, full_result.predict(grid), color=TEXT_COLOR,
            linewidth=2, label="Logistic fit, all normal seasons")
    ax.plot(seasons, without_result.predict(grid), color=MUTED_COLOR,
            linewidth=2, linestyle="--",
            label=f"Logistic fit, without {INFLUENTIAL_SEASON}")
    ax.axhline(0.5, color=MUTED_COLOR, linewidth=1, linestyle=":")

    ax.set_title("RQ1: trend in home win rate, logistic regression")
    ax.set_xlabel("Season (start year)")
    ax.set_ylabel("Home team win rate")
    ax.set_ylim(0.45, 0.65)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------ 2. series-level permutation


def add_series_id(games):
    """
    Returns a copy of games with SERIES_ID, which is the same for every
    game between the same two teams in the same season. In the playoffs
    this identifies a series (two teams meet at most once per playoffs).
    """
    df = games.copy()
    low = df[["HOME_TEAM_ID", "VISITOR_TEAM_ID"]].min(axis=1)
    high = df[["HOME_TEAM_ID", "VISITOR_TEAM_ID"]].max(axis=1)
    df["SERIES_ID"] = (df["SEASON"].astype(str) + "_" + low.astype(str)
                       + "_" + high.astype(str))
    return df


def cluster_permutation_test(games, period, column, unit,
                             n_resamples=N_PERMUTATIONS,
                             seed=RANDOM_SEED):
    """
    Permutation test of mean column in period vs BASE_PERIOD where whole
    units (e.g. SERIES_ID, or GAME_ID for a game-level test) are shuffled
    between the two periods, keeping all games of a unit together. The
    statistic is the per-game mean of column in period minus that in
    BASE_PERIOD. Returns a dict with the observed difference, the
    two-sided p-value and the number of units in each period. The test
    is exact when there are no more possible splits than n_resamples.
    """
    df = games[games["PERIOD"].isin([period, BASE_PERIOD])]
    units = df.groupby(unit).agg(total=(column, "sum"),
                                 games=(column, "size"),
                                 period=("PERIOD", "first"))
    totals = units["total"].to_numpy(dtype=float)
    sizes = units["games"].to_numpy(dtype=float)
    in_period = np.flatnonzero(units["period"] == period)
    in_base = np.flatnonzero(units["period"] == BASE_PERIOD)

    def statistic(x, y, axis):
        """
        Per-game mean of the units indexed by x minus that of y.
        """
        return (totals[x].sum(axis=axis) / sizes[x].sum(axis=axis)
                - totals[y].sum(axis=axis) / sizes[y].sum(axis=axis))

    result = stats.permutation_test(
        (in_period, in_base), statistic, permutation_type="independent",
        vectorized=True, n_resamples=n_resamples,
        alternative="two-sided", random_state=seed)
    return {"observed_diff": result.statistic, "p_value": result.pvalue,
            "units_period": len(in_period), "units_base": len(in_base)}


def series_variance_ratio(games, unit="SERIES_ID",
                          column="HOME_TEAM_WINS"):
    """
    Measures how much games in the same unit move together. Returns the
    game-weighted variance of each unit's mean of column around the
    overall mean, divided by the variance expected if every game were
    independent (p * (1 - p) / average games per unit). About 1 means
    games in a unit behave like independent games; above 1 means they
    are positively correlated, which makes game-level p-values too small.
    """
    overall = games[column].mean()
    units = games.groupby(unit)[column].agg(["mean", "size"])
    observed = np.average((units["mean"] - overall) ** 2,
                          weights=units["size"])
    expected = overall * (1 - overall) / units["size"].mean()
    return observed / expected


def series_vs_game_table(playoffs):
    """
    For each comparison period and outcome, runs the playoff permutation
    test twice: shuffling single games and shuffling whole series.
    Returns a DataFrame comparing the two p-values.
    """
    rows = []
    for period in COMPARISON_PERIODS:
        for column in OUTCOMES:
            game = cluster_permutation_test(playoffs, period, column,
                                            "GAME_ID")
            series = cluster_permutation_test(playoffs, period, column,
                                              "SERIES_ID")
            rows.append({"period": period, "outcome": column,
                         "observed_diff": game["observed_diff"],
                         "games": game["units_period"],
                         "series": series["units_period"],
                         "p_game_level": game["p_value"],
                         "p_series_level": series["p_value"]})
    return pd.DataFrame(rows)


def series_section(table, variance_ratio):
    """
    Returns text explaining the series-level permutation test and
    reporting its results next to the game-level test, plus the
    within-series variance ratio of HOME_TEAM_WINS.
    """
    return "\n".join([
        "2. Playoff dependence - series-level permutation tests",
        "Problem: playoff games come in series of 4-7 games between the "
        "same two teams, so games in a series are not independent. The "
        "game-level tests treat 83 bubble playoff games as 83 independent "
        "observations, but they come from only 15 series.",
        "Fix: shuffle whole series between periods instead of single "
        "games. Exchangeability then only needs different series to be "
        f"independent. {N_PERMUTATIONS} shuffles, seed {RANDOM_SEED}; "
        "the statistic is the per-game mean (win rate or POINT_DIFF) in "
        "the period minus the normal period.",
        table.to_string(index=False, float_format="{:.4g}".format),
        "If p_series_level is much larger than p_game_level, the "
        "game-level result was overstated by treating games in a series "
        "as independent.",
        "",
        "Why the two p-values are close: the variance of series-level "
        "home win rates is "
        f"{variance_ratio:.2f} times what independent games would give "
        "(1 = independent). Home court alternates between the two teams "
        "in a series, so the stronger team's edge shows up as home wins "
        "in some games and home losses in others, and HOME_TEAM_WINS is "
        "close to uncorrelated within a series.",
    ])


# --------------------------------------------- 3. multiple comparisons


def adjust_p_values(p_values):
    """
    Takes a list of p-values from one family of tests and returns two
    lists: the Holm-adjusted and the Bonferroni-adjusted p-values (both
    capped at 1), in the original order.
    """
    holm = multipletests(p_values, method="holm")[1]
    bonferroni = multipletests(p_values, method="bonferroni")[1]
    return list(holm), list(bonferroni)


def multiple_comparisons_table(prop_table, welch_table):
    """
    Takes the z-test table (from proportion_tests) and the Welch table
    (from welch_tests) and returns one DataFrame with the raw, Holm and
    Bonferroni p-values. Each outcome (win rate, POINT_DIFF) is its own
    family of four tests.
    """
    rows = []
    families = [("win rate (z-test)", prop_table, "p_value"),
                ("POINT_DIFF (Welch)", welch_table, "welch_p")]
    for name, table, column in families:
        holm, bonferroni = adjust_p_values(list(table[column]))
        for i, row in enumerate(table.itertuples()):
            rows.append({"family": name, "GAME_TYPE": row.GAME_TYPE,
                         "period": row.period,
                         "p_raw": getattr(row, column),
                         "p_holm": holm[i], "p_bonferroni": bonferroni[i]})
    table = pd.DataFrame(rows)
    table["significant_holm"] = table["p_holm"] < ALPHA
    return table


def multiple_comparisons_section(table):
    """
    Returns text reporting the adjusted p-values.
    """
    return "\n".join([
        "3. Multiple comparisons",
        "RQ2 makes 4 comparisons per outcome (covid_2021 and bubble vs "
        "normal, regular season and playoffs). Holm's method controls "
        "the chance of any false positive in the family at 5%, like "
        "Bonferroni, but is less conservative.",
        table.to_string(index=False, float_format="{:.4g}".format),
    ])


# ---------------------------------------------------------------- 4. power


def proportion_power(rate_period, rate_base, n_period, n_base):
    """
    Returns the power of a two-sided two-proportion z-test at level
    ALPHA to detect the difference between rate_period and rate_base
    with the given group sizes (Cohen's h effect size).
    """
    h = proportion_effectsize(rate_period, rate_base)
    return NormalIndPower().power(effect_size=abs(h), nobs1=n_period,
                                  alpha=ALPHA, ratio=n_base / n_period)


def min_detectable_rate_gap(rate_base, n_period, n_base,
                            power=TARGET_POWER):
    """
    Returns the smallest drop below rate_base in a period's win rate that
    a two-sided two-proportion z-test would detect with the given power
    at level ALPHA, for the given group sizes.
    """
    h = NormalIndPower().solve_power(nobs1=n_period, alpha=ALPHA,
                                     power=power, ratio=n_base / n_period)
    lower = np.sin((2 * np.arcsin(np.sqrt(rate_base)) - h) / 2) ** 2
    return rate_base - lower


def mean_power(mean_diff, sd, n_period, n_base):
    """
    Returns the power of a two-sided two-sample test of means at level
    ALPHA to detect a difference of mean_diff, when both groups have
    standard deviation sd (Cohen's d = mean_diff / sd). Uses the normal
    approximation: every comparison here has over 1,000 degrees of
    freedom, where the t and normal distributions agree.
    """
    return NormalIndPower().power(effect_size=abs(mean_diff) / sd,
                                  nobs1=n_period, alpha=ALPHA,
                                  ratio=n_base / n_period)


def min_detectable_mean_gap(sd, n_period, n_base, power=TARGET_POWER):
    """
    Returns the smallest difference in means (in the units of sd) that a
    two-sided two-sample test of means would detect with the given power
    at level ALPHA, for the given group sizes and standard deviation
    (normal approximation, see mean_power).
    """
    d = NormalIndPower().solve_power(nobs1=n_period, alpha=ALPHA,
                                     power=power, ratio=n_base / n_period)
    return d * sd


def games_needed(power_function, n_base, power=TARGET_POWER):
    """
    Returns the smallest number of games in the period group for which
    power_function(n_period, n_base) reaches power, keeping the base
    group at n_base games. Because the base group stays fixed, power has
    a ceiling, so some differences cannot be detected however many
    period games are added; in that case (checked at MAX_GAMES) this
    returns infinity.
    """
    def gap(n):
        return power_function(n, n_base) - power

    upper = MAX_GAMES
    if not gap(upper) >= 0:
        return float("inf")
    return int(np.ceil(optimize.brentq(gap, 2, upper)))


def power_table(clean):
    """
    For each GAME_TYPE, comparison period and outcome, returns a
    DataFrame with the group sizes, the observed difference from the
    normal period, the power to detect it, the minimum detectable
    difference at TARGET_POWER, and the number of period games needed
    to detect the observed difference with TARGET_POWER.
    """
    rows = []
    for game_type in GAME_TYPE_NAMES:
        for period in COMPARISON_PERIODS:
            wins = group_values(clean, period, game_type, "HOME_TEAM_WINS")
            base_wins = group_values(clean, BASE_PERIOD, game_type,
                                     "HOME_TEAM_WINS")
            diffs = group_values(clean, period, game_type, "POINT_DIFF")
            base_diffs = group_values(clean, BASE_PERIOD, game_type,
                                      "POINT_DIFF")
            n, n_base = len(wins), len(base_wins)
            rate, base_rate = wins.mean(), base_wins.mean()
            sd = np.sqrt((diffs.var(ddof=1) + base_diffs.var(ddof=1)) / 2)
            mean_gap = diffs.mean() - base_diffs.mean()

            def rate_power(n1, n2):
                return proportion_power(rate, base_rate, n1, n2)

            def diff_power(n1, n2):
                return mean_power(mean_gap, sd, n1, n2)

            rows.append({
                "GAME_TYPE": game_type, "period": period,
                "outcome": "win rate", "n_period": n,
                "observed_diff": rate - base_rate,
                "power": rate_power(n, n_base),
                "min_detectable": min_detectable_rate_gap(base_rate, n,
                                                          n_base),
                "games_needed": games_needed(rate_power, n_base)})
            rows.append({
                "GAME_TYPE": game_type, "period": period,
                "outcome": "POINT_DIFF", "n_period": n,
                "observed_diff": mean_gap,
                "power": diff_power(n, n_base),
                "min_detectable": min_detectable_mean_gap(sd, n, n_base),
                "games_needed": games_needed(diff_power, n_base)})
    return pd.DataFrame(rows)


def power_section(table):
    """
    Returns text explaining and reporting the power analysis.
    """
    return "\n".join([
        "4. Power analysis (alpha = 0.05, two-sided, normal period as "
        "the comparison group)",
        "power: chance the test would detect a true difference as large "
        "as the observed one. min_detectable: smallest true difference "
        f"(win-rate points or POINT_DIFF points) detected with "
        f"{TARGET_POWER:.0%} power. games_needed: period games needed to "
        f"detect the observed difference with {TARGET_POWER:.0%} power.",
        "Power computed from the observed difference is only a rough "
        "guide; min_detectable is the more useful number because it "
        "does not depend on this sample's result.",
        table.to_string(index=False, float_format="{:.4g}".format),
        "A non-significant result with low power means 'no evidence of "
        "a difference', not 'no difference'.",
    ])


def main():
    """
    Loads and cleans games.csv, runs the four robustness checks, saves
    the logistic trend figure, then prints the results and writes them
    to RESULTS_PATH.
    """
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    set_plot_style()

    clean = clean_games(load_games(RAW_PATH))
    normal_regular = clean[(clean["PERIOD"] == BASE_PERIOD)
                           & (clean["GAME_TYPE"] == REGULAR_SEASON)]
    without = normal_regular[normal_regular["SEASON"]
                             != INFLUENTIAL_SEASON]
    full_fit = logistic_trend(normal_regular)
    without_fit = logistic_trend(without)
    plot_logistic_trend(normal_regular, full_fit, without_fit,
                        FIGURE_PATH)

    playoffs = add_series_id(clean[clean["GAME_TYPE"] == PLAYOFFS])

    divider = "\n\n" + "=" * 72 + "\n\n"
    report = divider.join([
        logistic_section(logistic_summary(full_fit),
                         logistic_summary(without_fit),
                         linearity_lr_test(normal_regular),
                         len(normal_regular)),
        series_section(series_vs_game_table(playoffs),
                       series_variance_ratio(playoffs)),
        multiple_comparisons_section(multiple_comparisons_table(
            proportion_tests(clean), welch_tests(clean))),
        power_section(power_table(clean)),
    ]) + "\n"
    print(report)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
