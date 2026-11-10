"""
CSE 163 Final Project: NBA home-court advantage.

Result Validity checks for RQ1 and RQ2. For each null hypothesis this
script runs the planned test and checks the assumptions behind it:

RQ1
- H0 A: home win rate = 50% in every normal regular season
  (scipy binomtest; checks sample size and independence)
- H0 B: no trend in the home win rate across seasons (statsmodels OLS;
  checks residual normality, autocorrelation, linearity, influence)

RQ2 (regular season and playoffs separately)
- H0 C: bubble / covid_2021 home win rate = normal home win rate
  (two-proportion z-test; checks n*p and n*(1-p) >= 10)
- H0 D: bubble / covid_2021 mean POINT_DIFF = normal mean POINT_DIFF
  (Welch's t-test; checks normality / CLT and compares variances)
- Permutation tests for C and D, which need almost no distributional
  assumptions

Figures go to figures/ and all numbers to results/validity_checks.txt.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.stattools import durbin_watson

from clean import load_games, clean_games
from eda_games import (set_plot_style, win_rate_table, PERIOD_COLORS,
                       PERIOD_NAMES, GAME_TYPE_NAMES, MUTED_COLOR)

RAW_PATH = "data/games.csv"
FIGURES_DIR = "figures"
RESULTS_PATH = "results/validity_checks.txt"

ALPHA = 0.05
# Rule of thumb for the normal approximation to a proportion.
MIN_SUCCESSES = 10
# Above this size Shapiro-Wilk flags even tiny departures from
# normality, so it is not reported for large groups.
SHAPIRO_MAX_N = 5000
# Rule of thumb for a large Cook's distance.
COOKS_CUTOFF_SCALE = 4
N_PERMUTATIONS = 10000
PERMUTATION_BATCH = 500
RANDOM_SEED = 163

BASE_PERIOD = "normal"
COMPARISON_PERIODS = ["covid_2021", "bubble"]
ALL_PERIODS = [BASE_PERIOD] + COMPARISON_PERIODS
REGULAR_SEASON = 2


def group_values(df, period, game_type, column):
    """
    Returns the values of column for games in the given PERIOD and
    GAME_TYPE as a numpy array.
    """
    mask = (df["PERIOD"] == period) & (df["GAME_TYPE"] == game_type)
    return df.loc[mask, column].to_numpy()


def verdict(ok):
    """
    Returns "OK" if ok is True and "NOT MET" otherwise, for printing
    assumption checks.
    """
    return "OK" if ok else "NOT MET"


# ---------------------------------------------------------------- RQ1 A


def binomial_tests(season_table):
    """
    Takes the home win rate table for normal regular seasons (from
    win_rate_table) and returns a copy with, for each season, the
    expected home wins and losses under H0 (n * 0.5), the two-sided
    binomial test p-value against 50%, and the Bonferroni-adjusted
    p-value (p times the number of seasons, capped at 1).
    """
    table = season_table[["SEASON", "games", "home_wins",
                          "win_rate"]].copy()
    table["expected_wins_H0"] = table["games"] * 0.5
    table["p_value"] = [
        stats.binomtest(int(k), int(n), 0.5).pvalue
        for k, n in zip(table["home_wins"], table["games"])
    ]
    table["p_bonferroni"] = (table["p_value"] * len(table)).clip(upper=1)
    return table


def lag1_autocorrelation(values):
    """
    Returns the correlation between each value and the next one in the
    1-D array values (lag-1 autocorrelation).
    """
    return np.corrcoef(values[:-1], values[1:])[0, 1]


def independence_table(normal_regular):
    """
    For each normal regular season, orders games by date and GAME_ID and
    returns a DataFrame with the lag-1 autocorrelation of HOME_TEAM_WINS
    and the approximate 95% bound for no autocorrelation
    (1.96 / sqrt(n)).
    """
    rows = []
    for season, games in normal_regular.groupby("SEASON"):
        ordered = games.sort_values(["GAME_DATE_EST", "GAME_ID"])
        outcomes = ordered["HOME_TEAM_WINS"].to_numpy()
        rows.append({"SEASON": season,
                     "lag1_r": lag1_autocorrelation(outcomes),
                     "bound_95": 1.96 / np.sqrt(len(outcomes))})
    table = pd.DataFrame(rows)
    table["within_bound"] = table["lag1_r"].abs() <= table["bound_95"]
    return table


def rq1a_section(binom_table, indep_table):
    """
    Returns text reporting H0 A: sample sizes, the n*p check, the
    independence check and the binomial test p-values.
    """
    fmt = "{:.4g}".format
    min_expected = binom_table["expected_wins_H0"].min()
    n_within = indep_table["within_bound"].sum()
    return "\n".join([
        "RQ1 - H0 A: home win rate = 50% in each normal regular season",
        "Test: scipy.stats.binomtest (two-sided), one test per season.",
        "",
        "Sample size per season and test results:",
        binom_table.to_string(index=False, float_format=fmt),
        "",
        "Assumption 1 - large enough sample: binomtest is exact, so it "
        "does not need a large sample. Smallest expected count under H0 "
        f"is {min_expected:.0f} (>= {MIN_SUCCESSES}): "
        f"{verdict(min_expected >= MIN_SUCCESSES)}.",
        "",
        "Assumption 2 - independent games. Lag-1 autocorrelation of "
        "HOME_TEAM_WINS in date order:",
        indep_table.to_string(index=False, float_format=fmt),
        f"{n_within} of {len(indep_table)} seasons are inside the 95% "
        "bound (about 1 in 20 would fall outside by chance).",
        "Limitation: this only checks consecutive games across the "
        "league. Games of the same team (strength, schedule, injuries) "
        "are not fully independent, so p-values may be slightly too "
        "small.",
        "",
        f"Multiple testing: {len(binom_table)} seasons are tested, so "
        "Bonferroni-adjusted p-values are also shown.",
    ])


# ---------------------------------------------------------------- RQ1 B


def fit_trend(season_table):
    """
    Fits and returns the statsmodels OLS result of win_rate on SEASON
    for the given season table.
    """
    x = sm.add_constant(season_table["SEASON"].astype(float))
    return sm.OLS(season_table["win_rate"], x).fit()


def quadratic_term_p_value(season_table):
    """
    Fits win_rate on centered SEASON and centered SEASON squared and
    returns the p-value of the squared term. A small p-value means the
    relationship is curved, not linear.
    """
    centered = season_table["SEASON"] - season_table["SEASON"].mean()
    x = sm.add_constant(pd.DataFrame({"season": centered,
                                      "season_sq": centered ** 2}))
    return sm.OLS(season_table["win_rate"], x).fit().pvalues["season_sq"]


def rq1b_checks(season_table):
    """
    Fits the trend model and returns a dict with the slope, its p-value,
    the Shapiro-Wilk p-value and Durbin-Watson statistic of the
    residuals, the quadratic-term p-value, the largest Cook's distance
    and its season, and the slope / p-value without that season.
    """
    result = fit_trend(season_table)
    cooks = result.get_influence().cooks_distance[0]
    worst = int(np.argmax(cooks))
    worst_season = season_table["SEASON"].iloc[worst]
    without = fit_trend(season_table[season_table["SEASON"]
                                     != worst_season])
    return {
        "result": result,
        "slope": result.params["SEASON"],
        "slope_p": result.pvalues["SEASON"],
        "r_squared": result.rsquared,
        "shapiro_p": stats.shapiro(result.resid).pvalue,
        "durbin_watson": durbin_watson(result.resid),
        "quadratic_p": quadratic_term_p_value(season_table),
        "max_cooks": cooks[worst],
        "cooks_cutoff": COOKS_CUTOFF_SCALE / len(season_table),
        "max_cooks_season": worst_season,
        "slope_without": without.params["SEASON"],
        "slope_p_without": without.pvalues["SEASON"],
    }


def rq1b_section(checks, n_seasons):
    """
    Returns text reporting H0 B: the OLS slope and each assumption
    check with a verdict.
    """
    dw = checks["durbin_watson"]
    return "\n".join([
        "RQ1 - H0 B: no trend in home win rate across normal seasons "
        "(slope = 0)",
        f"Test: statsmodels OLS, win_rate ~ SEASON, n = {n_seasons} "
        "seasons.",
        f"Slope = {checks['slope']:.5f} per season "
        f"({checks['slope'] * 10 * 100:.2f} percentage points per "
        f"decade), p = {checks['slope_p']:.4g}, "
        f"R^2 = {checks['r_squared']:.3f}",
        "",
        "Assumption 1 - residuals roughly normal (Q-Q plot: "
        "figures/fig4_rq1_trend_diagnostics.png). Shapiro-Wilk p = "
        f"{checks['shapiro_p']:.4g}: "
        f"{verdict(checks['shapiro_p'] > ALPHA)}.",
        "Assumption 2 - no autocorrelation in residuals. Durbin-Watson "
        f"= {dw:.3f} (about 2 = none; below 1.5 or above 2.5 is a "
        f"warning): {verdict(1.5 <= dw <= 2.5)}.",
        "Note: seasons 2019 and 2020 are missing, so 2018 and 2021 are "
        "treated as neighbors.",
        "Assumption 3 - linear relationship (residual plot in the same "
        "figure). p-value of an added SEASON^2 term = "
        f"{checks['quadratic_p']:.4g}: "
        f"{verdict(checks['quadratic_p'] > ALPHA)}.",
        "Influence check: largest Cook's distance = "
        f"{checks['max_cooks']:.3f} (SEASON "
        f"{checks['max_cooks_season']}); rule-of-thumb cutoff 4/n = "
        f"{checks['cooks_cutoff']:.3f}: "
        f"{verdict(checks['max_cooks'] <= checks['cooks_cutoff'])}.",
        f"Without SEASON {checks['max_cooks_season']}: slope = "
        f"{checks['slope_without']:.5f}, "
        f"p = {checks['slope_p_without']:.4g}.",
    ])


def plot_trend_diagnostics(season_table, result, path):
    """
    Saves a two-panel figure to path: residuals of the trend model
    against SEASON (to judge linearity and spot outliers) and a normal
    Q-Q plot of the residuals.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    color = PERIOD_COLORS[BASE_PERIOD]

    ax1.scatter(season_table["SEASON"], result.resid, color=color, s=36,
                zorder=3)
    ax1.axhline(0, color=MUTED_COLOR, linestyle="--", linewidth=1)
    ax1.set_title("Residuals vs season")
    ax1.set_xlabel("Season (start year)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("Residual (win rate)")

    (theory, ordered), (slope, intercept, _) = stats.probplot(
        result.resid, dist="norm")
    ax2.scatter(theory, ordered, color=color, s=36, zorder=3)
    ax2.plot(theory, slope * theory + intercept, color=MUTED_COLOR,
             linestyle="--", linewidth=1)
    ax2.set_title("Normal Q-Q plot of residuals")
    ax2.set_xlabel("Theoretical normal quantiles")
    ax2.set_ylabel("Ordered residuals")

    fig.suptitle("RQ1 trend model diagnostics (OLS: win rate ~ season)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- RQ2 C


def proportion_tests(clean):
    """
    For each GAME_TYPE and each comparison period, compares its home win
    rate with the normal period using a two-sided two-proportion z-test.
    Returns a DataFrame with both groups' sizes and rates, the smallest
    of n*p and n*(1-p) across the two groups, the z statistic and the
    p-value.
    """
    rows = []
    for game_type in GAME_TYPE_NAMES:
        base = group_values(clean, BASE_PERIOD, game_type,
                            "HOME_TEAM_WINS")
        for period in COMPARISON_PERIODS:
            other = group_values(clean, period, game_type,
                                 "HOME_TEAM_WINS")
            counts = [other.sum(), base.sum()]
            sizes = [len(other), len(base)]
            z, p = proportions_ztest(counts, sizes)
            min_count = min(counts[0], sizes[0] - counts[0],
                            counts[1], sizes[1] - counts[1])
            rows.append({"GAME_TYPE": game_type, "period": period,
                         "n_period": sizes[0],
                         "rate_period": counts[0] / sizes[0],
                         "n_normal": sizes[1],
                         "rate_normal": counts[1] / sizes[1],
                         "min_np": min_count, "z": z, "p_value": p})
    table = pd.DataFrame(rows)
    table["np_check"] = table["min_np"] >= MIN_SUCCESSES
    return table


def rq2c_section(table):
    """
    Returns text reporting H0 C: the n*p check and z-test results.
    """
    fmt = "{:.4g}".format
    return "\n".join([
        "RQ2 - H0 C: home win rate in bubble / covid_2021 = normal",
        "Test: statsmodels proportions_ztest (two-sided), regular "
        "season (GAME_TYPE 2) and playoffs (GAME_TYPE 4) separately.",
        table.to_string(index=False, float_format=fmt),
        "",
        "Assumption - n*p and n*(1-p) >= 10 in both groups (min_np is "
        "the smallest of the four counts): "
        f"{verdict(table['np_check'].all())}.",
        "Assumption - independent games: same caveat as RQ1 A.",
    ])


# ---------------------------------------------------------------- RQ2 D


def shape_table(clean):
    """
    Returns a DataFrame with, for each PERIOD and GAME_TYPE, the number
    of games and the mean, standard deviation, skewness, excess kurtosis
    and Shapiro-Wilk p-value (only for groups up to SHAPIRO_MAX_N games)
    of POINT_DIFF.
    """
    rows = []
    for game_type in GAME_TYPE_NAMES:
        for period in ALL_PERIODS:
            values = group_values(clean, period, game_type, "POINT_DIFF")
            shapiro_p = (stats.shapiro(values).pvalue
                         if len(values) <= SHAPIRO_MAX_N else np.nan)
            rows.append({"GAME_TYPE": game_type, "period": period,
                         "n": len(values), "mean": values.mean(),
                         "sd": values.std(ddof=1),
                         "skewness": stats.skew(values),
                         "excess_kurtosis": stats.kurtosis(values),
                         "shapiro_p": shapiro_p})
    return pd.DataFrame(rows)


def welch_tests(clean):
    """
    For each GAME_TYPE and comparison period, compares mean POINT_DIFF
    with the normal period. Returns a DataFrame with the difference in
    means, the ratio of standard deviations, the Levene (median-centered)
    p-value for equal variances, and Welch's t statistic and p-value.
    """
    rows = []
    for game_type in GAME_TYPE_NAMES:
        base = group_values(clean, BASE_PERIOD, game_type, "POINT_DIFF")
        for period in COMPARISON_PERIODS:
            other = group_values(clean, period, game_type, "POINT_DIFF")
            welch = stats.ttest_ind(other, base, equal_var=False)
            rows.append({
                "GAME_TYPE": game_type, "period": period,
                "mean_diff": other.mean() - base.mean(),
                "sd_ratio": other.std(ddof=1) / base.std(ddof=1),
                "levene_p": stats.levene(other, base,
                                         center="median").pvalue,
                "welch_t": welch.statistic,
                "welch_p": welch.pvalue,
            })
    return pd.DataFrame(rows)


def rq2d_section(shapes, welch):
    """
    Returns text reporting H0 D: the shape of each group's POINT_DIFF
    distribution, the variance comparison and Welch's t-test results.
    """
    fmt = "{:.4g}".format
    max_skew = shapes["skewness"].abs().max()
    max_kurt = shapes["excess_kurtosis"].abs().max()
    min_n = shapes["n"].min()
    return "\n".join([
        "RQ2 - H0 D: mean POINT_DIFF in bubble / covid_2021 = normal",
        "Test: Welch's t-test (scipy ttest_ind, equal_var=False), "
        "regular season and playoffs separately.",
        "",
        "Assumption 1 - POINT_DIFF roughly normal, or n large enough for "
        "the CLT (Q-Q plots: figures/fig5_rq2_point_diff_qq.png).",
        shapes.to_string(index=False, float_format=fmt),
        f"Largest |skewness| = {max_skew:.3f}, largest |excess "
        f"kurtosis| = {max_kurt:.3f} (|skew| < 1 and |kurtosis| < 1 "
        "is close to normal). Smallest group n = "
        f"{min_n}. Shapiro-Wilk is skipped for groups over "
        f"{SHAPIRO_MAX_N} games because it rejects tiny departures.",
        f"Verdict: {verdict(max_skew < 1 and max_kurt < 1)} "
        "(near-normal shape, and every group has n >= 30 for the CLT).",
        "",
        "Assumption 2 - equal variances: NOT required by Welch's test. "
        "Levene's test is shown to justify using Welch instead of the "
        "pooled t-test:",
        welch.to_string(index=False, float_format=fmt),
        "Group sizes are very unequal (normal is over 100 times larger "
        "than bubble), so the pooled t-test would be unreliable if the "
        "variances differed even a little; Welch's test avoids this.",
        "Assumption 3 - independent games: same caveat as RQ1 A.",
    ])


def plot_point_diff_qq(clean, path):
    """
    Saves a grid of normal Q-Q plots of POINT_DIFF to path, one row per
    GAME_TYPE and one column per PERIOD (normal, covid_2021, bubble).
    """
    fig, axes = plt.subplots(len(GAME_TYPE_NAMES), len(ALL_PERIODS),
                             figsize=(11, 6.8))
    for row, game_type in enumerate(GAME_TYPE_NAMES):
        for col, period in enumerate(ALL_PERIODS):
            ax = axes[row, col]
            values = group_values(clean, period, game_type, "POINT_DIFF")
            (theory, ordered), (slope, intercept, _) = stats.probplot(
                values, dist="norm")
            ax.scatter(theory, ordered, color=PERIOD_COLORS[period],
                       s=8, alpha=0.6, linewidths=0)
            ax.plot(theory, slope * theory + intercept, color=MUTED_COLOR,
                    linestyle="--", linewidth=1)
            ax.set_title(f"{PERIOD_NAMES[period]}\n"
                         f"{GAME_TYPE_NAMES[game_type]} "
                         f"(n={len(values)})", fontsize=10)
            ax.set_xlabel("Theoretical normal quantiles")
            ax.set_ylabel("POINT_DIFF quantiles")
    fig.suptitle("RQ2: normal Q-Q plots of point differential by period",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------- permutation


def difference_in_means(x, y, axis):
    """
    Returns mean(x) - mean(y) along axis. Used as the permutation test
    statistic (for 0/1 outcomes this is the difference in win rates).
    """
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def permutation_tests(clean):
    """
    For each GAME_TYPE, comparison period and outcome (HOME_TEAM_WINS
    and POINT_DIFF), runs a two-sided permutation test of the
    difference in means against the normal period. Returns a DataFrame
    with the observed difference and the permutation p-value.
    """
    rows = []
    for game_type in GAME_TYPE_NAMES:
        for period in COMPARISON_PERIODS:
            for column in ["HOME_TEAM_WINS", "POINT_DIFF"]:
                other = group_values(clean, period, game_type, column)
                base = group_values(clean, BASE_PERIOD, game_type, column)
                result = stats.permutation_test(
                    (other, base), difference_in_means,
                    permutation_type="independent", vectorized=True,
                    n_resamples=N_PERMUTATIONS, batch=PERMUTATION_BATCH,
                    alternative="two-sided", random_state=RANDOM_SEED)
                rows.append({"GAME_TYPE": game_type, "period": period,
                             "outcome": column,
                             "observed_diff": result.statistic,
                             "perm_p": result.pvalue})
    return pd.DataFrame(rows)


def permutation_section(table):
    """
    Returns text explaining why the permutation test is used and
    reporting its results.
    """
    return "\n".join([
        "RQ2 - Permutation tests (for H0 C and H0 D)",
        f"Method: pool the two groups, shuffle the PERIOD labels "
        f"{N_PERMUTATIONS} times, and count how often the shuffled "
        "difference in means is at least as extreme as the observed "
        f"one (two-sided, seed {RANDOM_SEED}).",
        "Why: it does not assume normality, a large sample, or equal "
        "variances. Its only assumption is exchangeability: under H0 "
        "the period label does not matter, which requires independent "
        "games. This makes it a good check for the small bubble groups "
        "(88 and 83 games), where the z-test and t-test rely on "
        "large-sample approximations.",
        table.to_string(index=False, float_format="{:.4g}".format),
        f"Smallest possible p-value with {N_PERMUTATIONS} shuffles is "
        f"about {1 / (N_PERMUTATIONS + 1):.4g}.",
    ])


def main():
    """
    Loads and cleans games.csv, runs every RQ1 and RQ2 assumption check
    and test, saves the diagnostic figures, then prints the results and
    writes them to RESULTS_PATH.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    set_plot_style()

    clean = clean_games(load_games(RAW_PATH))
    normal_regular = clean[(clean["PERIOD"] == BASE_PERIOD)
                           & (clean["GAME_TYPE"] == REGULAR_SEASON)]
    season_table = win_rate_table(normal_regular, ["SEASON"])

    binom_table = binomial_tests(season_table)
    indep_table = independence_table(normal_regular)
    trend = rq1b_checks(season_table)
    prop_table = proportion_tests(clean)
    shapes = shape_table(clean)
    welch = welch_tests(clean)
    perm_table = permutation_tests(clean)

    plot_trend_diagnostics(
        season_table, trend["result"],
        os.path.join(FIGURES_DIR, "fig4_rq1_trend_diagnostics.png"))
    plot_point_diff_qq(
        clean, os.path.join(FIGURES_DIR, "fig5_rq2_point_diff_qq.png"))

    divider = "\n\n" + "=" * 72 + "\n\n"
    report = divider.join([
        rq1a_section(binom_table, indep_table),
        rq1b_section(trend, len(season_table)),
        rq2c_section(prop_table),
        rq2d_section(shapes, welch),
        permutation_section(perm_table),
    ]) + "\n"
    print(report)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
