"""
CSE 163 Final Project: NBA home-court advantage.

RQ4: Does home-court advantage (HCA) differ between teams, and is it
larger for the two high-altitude teams, the Denver Nuggets and the
Utah Jazz?

Exploratory data analysis of ranking.csv and teams.csv. For every team
and season it keeps the final standings row, computes

    HCA = home win rate - road win rate,

and prints and saves to results/eda_ranking.txt:
- the size of the raw and cleaned data and what one row represents
- missing values in the raw ranking.csv and teams.csv
- checks on games played and on the merge with teams_clean.csv
- summaries of the research variables
- the planned tests (one-way ANOVA, Welch's t-test) and checks of
  their assumptions, with Kruskal-Wallis / Mann-Whitney as backups
It also saves four figures (fig10-fig13) to figures/.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy import stats

from clean import NORMAL_SEASONS, load_csv
from eda_games import (missing_table, numeric_summary, set_plot_style,
                       MUTED_COLOR, TEXT_COLOR)

RAW_RANKING_PATH = "data/ranking.csv"
RAW_TEAMS_PATH = "data/teams.csv"
CLEAN_RANKING_PATH = "data/ranking_clean.csv"
CLEAN_TEAMS_PATH = "data/teams_clean.csv"
FIGURES_DIR = "figures"
RESULTS_PATH = "results/eda_ranking.txt"

# High-altitude teams: Denver (about 1,600 m) and Salt Lake City
# (about 1,300 m).
ALTITUDE_TEAMS = ["Denver Nuggets", "Utah Jazz"]

# Games per team in a full season, and in the 2011-12 lockout season.
FULL_SEASON_GAMES = 82
LOCKOUT_SEASON = 2011
LOCKOUT_SEASON_GAMES = 66

NUMERIC_COLUMNS = ["HOME_PCT", "ROAD_PCT", "HCA", "SEASON"]
ALPHA = 0.05

ALTITUDE_COLOR = "#eb6834"
OTHER_COLOR = "#2a78d6"


def size_section(raw_ranking, clean_ranking, raw_teams):
    """
    Returns text describing how many rows and columns raw ranking.csv,
    ranking_clean.csv and teams.csv have and what one row represents.
    """
    days = raw_ranking["STANDINGSDATE"].nunique()
    return "\n".join([
        "1. DATA SIZE",
        f"Raw ranking.csv: {raw_ranking.shape[0]} rows x "
        f"{raw_ranking.shape[1]} columns",
        f"ranking_clean.csv: {clean_ranking.shape[0]} rows x "
        f"{clean_ranking.shape[1]} columns",
        f"Raw teams.csv: {raw_teams.shape[0]} rows x "
        f"{raw_teams.shape[1]} columns",
        "Each row of ranking.csv is one team's standings on one day "
        "(wins, losses, home record and road record so far that "
        f"season). The file covers {days} different dates, which is "
        "why it has so many rows: about 30 teams x hundreds of days "
        "per season.",
        "ranking_clean.csv keeps only regular-season standings from "
        "SEASON 2003 on, drops repeated team/season/date rows and adds "
        "SEASON, HOME_W, HOME_L, ROAD_W and ROAD_L.",
        "Each row of teams.csv is one current NBA franchise (30 teams).",
    ])


def missing_section(raw_ranking, raw_teams):
    """
    Returns text listing the number and percent of missing values in
    each column of raw ranking.csv and teams.csv and whether they
    matter for RQ4.
    """
    fmt = "{:.2f}".format
    ranking_table = missing_table(raw_ranking)
    teams_table = missing_table(raw_teams)
    return "\n".join([
        "2. MISSING VALUES",
        f"ranking.csv ({len(raw_ranking)} rows), columns with missing "
        "values:",
        ranking_table.to_string(float_format=fmt),
        f"The other {raw_ranking.shape[1] - len(ranking_table)} columns "
        "have no missing values.",
        "RETURNTOPLAY only exists for the 2019-20 season restart (was "
        "the team invited to the bubble). RQ4 does not use it, and "
        "HOME_RECORD / ROAD_RECORD, the columns we need, are complete.",
        "",
        f"teams.csv ({len(raw_teams)} rows), columns with missing "
        "values:",
        teams_table.to_string(float_format=fmt),
        "ARENACAPACITY is not used in RQ4. TEAM_ID, CITY and NICKNAME "
        "(used to build TEAM_NAME) are complete, so missing values do "
        "not affect this analysis.",
    ])


def final_standings(ranking, seasons):
    """
    Takes a cleaned ranking DataFrame and a list of seasons and returns
    one row per TEAM_ID and SEASON: the row with the latest
    STANDINGSDATE, i.e. the team's final regular-season record. Only
    seasons in the list are kept. The input DataFrame is not modified.
    """
    df = ranking[ranking["SEASON"].isin(seasons)]
    df = df.sort_values("STANDINGSDATE")
    df = df.groupby(["TEAM_ID", "SEASON"]).tail(1)
    return df.sort_values(["SEASON", "TEAM_ID"]).reset_index(drop=True)


def add_hca(df):
    """
    Returns a copy of df with four new columns computed from HOME_W,
    HOME_L, ROAD_W and ROAD_L:
    - GAMES_PLAYED: all four added together
    - HOME_PCT: HOME_W / (HOME_W + HOME_L)
    - ROAD_PCT: ROAD_W / (ROAD_W + ROAD_L)
    - HCA: HOME_PCT - ROAD_PCT
    """
    df = df.copy()
    home_games = df["HOME_W"] + df["HOME_L"]
    road_games = df["ROAD_W"] + df["ROAD_L"]
    df["GAMES_PLAYED"] = home_games + road_games
    df["HOME_PCT"] = df["HOME_W"] / home_games
    df["ROAD_PCT"] = df["ROAD_W"] / road_games
    df["HCA"] = df["HOME_PCT"] - df["ROAD_PCT"]
    return df


def unusual_game_counts(df):
    """
    Returns the rows of df (which must have SEASON and GAMES_PLAYED)
    whose GAMES_PLAYED is not the expected number of games: 66 in the
    2011 lockout season, 82 in every other season.
    """
    expected = df["SEASON"].map(
        lambda s: LOCKOUT_SEASON_GAMES if s == LOCKOUT_SEASON
        else FULL_SEASON_GAMES)
    return df[df["GAMES_PLAYED"] != expected]


def merge_teams(standings, teams):
    """
    Left-merges standings with teams on TEAM_ID to add TEAM_NAME.
    Returns a tuple (merged DataFrame, list of TEAM_IDs in standings
    that have no match in teams).
    """
    merged = standings.merge(teams[["TEAM_ID", "TEAM_NAME"]],
                             on="TEAM_ID", how="left")
    unmatched = merged.loc[merged["TEAM_NAME"].isna(), "TEAM_ID"]
    return merged, sorted(unmatched.unique())


def checks_section(standings, unusual, unmatched):
    """
    Returns text describing the final-standings table, the games-played
    check and the teams merge check.
    """
    played = standings.groupby("SEASON")["GAMES_PLAYED"].agg(["min",
                                                              "max"])
    wins_match = ((standings["HOME_W"] + standings["ROAD_W"])
                  == standings["W"]).all()
    if len(unusual) == 0:
        unusual_text = "none"
    else:
        unusual_text = unusual[["SEASON", "TEAM_NAME", "GAMES_PLAYED",
                                "HOME_W", "HOME_L", "ROAD_W",
                                "ROAD_L"]].to_string(index=False)
    return "\n".join([
        "3. FINAL STANDINGS AND CHECKS",
        f"Seasons kept: {NORMAL_SEASONS[0]}-{NORMAL_SEASONS[-2]} and "
        f"{NORMAL_SEASONS[-1]} (the 'normal' PERIOD in games_clean.csv)."
        f" {standings['SEASON'].nunique()} seasons, "
        f"{len(standings)} team-seasons.",
        "Each row is now one team in one season, taken from the last "
        "STANDINGSDATE of that season (its final record).",
        "Games played (HOME_W + HOME_L + ROAD_W + ROAD_L) by season:",
        played.to_string(),
        "Team-seasons that do not have 82 games (66 in 2011):",
        unusual_text,
        "The 2012 rows with 81 games are Boston and Indiana: their "
        "April 16, 2013 game was cancelled after the Boston Marathon "
        "bombing and never made up, so 81 is correct.",
        f"HOME_W + ROAD_W equals W in every row: {wins_match}.",
        f"TEAM_IDs with no match in teams_clean.csv: "
        f"{unmatched if unmatched else 'none'}.",
        "Note: TEAM_NAME is the franchise's current name, so e.g. the "
        "Seattle SuperSonics seasons (2004-2007) are listed under "
        "Oklahoma City Thunder and New Jersey under Brooklyn Nets.",
    ])


def variables_section(standings):
    """
    Returns text with the seven-number summary of HOME_PCT, ROAD_PCT,
    HCA and SEASON and the number of seasons for each TEAM_NAME.
    """
    team_counts = standings["TEAM_NAME"].value_counts().sort_index()
    season_counts = standings["SEASON"].value_counts().sort_index()
    lines = [
        f"4. RESEARCH VARIABLES ({len(standings)} team-seasons)",
        "Numeric variables (HOME_PCT = home win rate, ROAD_PCT = road "
        "win rate, HCA = HOME_PCT - ROAD_PCT):",
        numeric_summary(standings, NUMERIC_COLUMNS).to_string(
            float_format="{:.3f}".format),
        "",
        "SEASON (categorical): team-seasons per season:",
        ", ".join(f"{s}: {n}" for s, n in season_counts.items()),
        "",
        "TEAM_NAME (categorical): seasons per team:",
    ]
    for team, count in team_counts.items():
        lines.append(f"  {team}: {count}")
    return "\n".join(lines)


def team_hca_table(standings):
    """
    Returns a DataFrame with one row per TEAM_NAME: the number of
    seasons, mean HCA, standard deviation of HCA, mean home win rate
    and mean road win rate, sorted from largest to smallest mean HCA.
    """
    table = standings.groupby("TEAM_NAME").agg(
        seasons=("HCA", "count"), mean_hca=("HCA", "mean"),
        sd_hca=("HCA", "std"), home_pct=("HOME_PCT", "mean"),
        road_pct=("ROAD_PCT", "mean"))
    return table.sort_values("mean_hca", ascending=False)


def season_hca_table(standings):
    """
    Returns a DataFrame with one row per SEASON: the number of teams
    and the mean HCA, home win rate and road win rate across teams.
    """
    return standings.groupby("SEASON").agg(
        teams=("HCA", "count"), mean_hca=("HCA", "mean"),
        home_pct=("HOME_PCT", "mean"), road_pct=("ROAD_PCT", "mean"))


def hca_by_team(standings):
    """
    Returns a dict that maps each TEAM_NAME to a list of its HCA values
    (one per season), with teams in alphabetical order.
    """
    groups = {}
    for team, values in standings.groupby("TEAM_NAME")["HCA"]:
        groups[team] = list(values)
    return groups


def split_altitude(standings, altitude_teams):
    """
    Returns a tuple (altitude HCA values, other HCA values) as two
    Series: rows whose TEAM_NAME is in altitude_teams and all other
    rows.
    """
    is_altitude = standings["TEAM_NAME"].isin(altitude_teams)
    return (standings.loc[is_altitude, "HCA"],
            standings.loc[~is_altitude, "HCA"])


def team_residuals(standings):
    """
    Returns a Series of ANOVA residuals: each row's HCA minus the mean
    HCA of its team.
    """
    team_mean = standings.groupby("TEAM_NAME")["HCA"].transform("mean")
    return standings["HCA"] - team_mean


def residuals_sd(standings):
    """
    Returns the standard deviation of the ANOVA residuals (see
    team_residuals), i.e. the typical season-to-season spread of a
    team's HCA around its own mean.
    """
    return team_residuals(standings).std()


def lag1_correlation(standings):
    """
    Returns the correlation between a team's residual (see
    team_residuals) in one season and its residual in the next season
    in the data, pooled over all teams. A value near 0 means seasons of
    the same team behave like independent draws.
    """
    df = standings.sort_values(["TEAM_NAME", "SEASON"]).copy()
    df["RESID"] = team_residuals(df)
    df["NEXT_RESID"] = df.groupby("TEAM_NAME")["RESID"].shift(-1)
    df = df.dropna(subset=["NEXT_RESID"])
    return df["RESID"].corr(df["NEXT_RESID"])


def sampling_sd(standings):
    """
    Returns the SD of HCA we would expect from game-to-game chance
    alone. For one team-season, HOME_PCT is a proportion of
    home_games games, so its variance is HOME_PCT * (1 - HOME_PCT) /
    home_games, and the same for ROAD_PCT; HCA's variance is the sum.
    The result is the square root of the average of that variance over
    all rows.
    """
    home_games = standings["HOME_W"] + standings["HOME_L"]
    road_games = standings["ROAD_W"] + standings["ROAD_L"]
    home = standings["HOME_PCT"]
    road = standings["ROAD_PCT"]
    variance = (home * (1 - home) / home_games
                + road * (1 - road) / road_games)
    return variance.mean() ** 0.5


def pair_permutation_p(team_means, chosen):
    """
    Exact team-level permutation test. team_means is a Series mapping
    team name -> mean HCA, chosen is a list of two team names. Returns
    the share of all possible pairs of teams whose average mean HCA is
    at least as large as the chosen pair's average (one-sided p-value).
    """
    observed = team_means[chosen].mean()
    names = list(team_means.index)
    count = 0
    total = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair_mean = (team_means[names[i]] + team_means[names[j]]) / 2
            total += 1
            if pair_mean >= observed - 1e-12:
                count += 1
    return count / total


def eta_squared(groups):
    """
    Takes a dict of group name -> list of values and returns eta
    squared: the share of the total variation explained by the group
    means (between-group sum of squares / total sum of squares).
    """
    values = [v for group in groups.values() for v in group]
    grand_mean = sum(values) / len(values)
    total = sum((v - grand_mean) ** 2 for v in values)
    between = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2
                  for group in groups.values())
    return between / total


def verdict(p_value):
    """
    Returns "reject H0" if p_value < ALPHA, otherwise
    "fail to reject H0".
    """
    return "reject H0" if p_value < ALPHA else "fail to reject H0"


def holds(condition):
    """
    Returns "MET" if condition is True, otherwise "NOT MET".
    """
    return "MET" if condition else "NOT MET"


def anova_section(standings):
    """
    Runs the test for null hypothesis A (all 30 teams have the same mean
    HCA): one-way ANOVA, plus Levene's test, Shapiro-Wilk tests and a
    lag-1 correlation to check its assumptions, and Kruskal-Wallis as
    the rank-based backup. Returns the report text.
    """
    groups = hca_by_team(standings)
    values = list(groups.values())
    f_stat, f_p = stats.f_oneway(*values)
    lev_stat, lev_p = stats.levene(*values, center="median")
    team_shapiro = {team: stats.shapiro(v).pvalue
                    for team, v in groups.items()}
    n_small = sum(p < ALPHA for p in team_shapiro.values())
    residuals = team_residuals(standings)
    resid_w, resid_p = stats.shapiro(residuals)
    lag1 = lag1_correlation(standings)
    kw_stat, kw_p = stats.kruskal(*values)
    eta2 = eta_squared(groups)
    sds = standings.groupby("TEAM_NAME")["HCA"].std()
    noise_sd = sampling_sd(standings)
    n_seasons = len(standings) // len(groups)

    return "\n".join([
        "6. RESULT VALIDITY - NULL HYPOTHESIS A",
        "H0: all 30 teams have the same mean HCA. "
        "H1: at least one team's mean HCA is different.",
        "Planned test: one-way ANOVA (scipy.stats.f_oneway), 30 groups "
        "of 16 seasons each.",
        f"  F = {f_stat:.3f}, p = {f_p:.4f} -> {verdict(f_p)} at "
        f"alpha = {ALPHA}",
        f"  Effect size eta^2 = {eta2:.3f} (share of HCA variation "
        "explained by which team it is)",
        "  Context: with about 41 home and 41 road games a season, "
        "chance alone gives one season's HCA an SD of about "
        f"{noise_sd:.3f}. The observed SD of HCA around each team's "
        f"own mean is {residuals_sd(standings):.3f}, so most of the "
        "season-to-season spread is game-to-game luck, and team "
        "differences are small next to it.",
        "",
        "Assumption 1 - equal variances across teams (Levene's test, "
        "median-centered):",
        f"  W = {lev_stat:.3f}, p = {lev_p:.4f}. Team SDs of HCA range "
        f"from {sds.min():.3f} to {sds.max():.3f}.",
        f"  -> {holds(lev_p >= ALPHA)}",
        "Assumption 2 - roughly normal values in each group:",
        f"  Shapiro-Wilk per team: {n_small} of {len(groups)} teams have "
        f"p < {ALPHA} (about {len(groups) * ALPHA:.1f} would be expected "
        "by chance even if all were normal). With only 16 values per "
        "team these tests have little power, so we also test the "
        "pooled residuals.",
        f"  Shapiro-Wilk on all {len(residuals)} residuals (HCA minus "
        f"team mean): W = {resid_w:.4f}, p = {resid_p:.4f}. "
        "Q-Q plot: figures/fig13_hca_residual_qq.png",
        f"  -> {holds(resid_p >= ALPHA and n_small <= 3)}",
        "Assumption 3 - independent observations:",
        "  Different teams are different groups, but the same team's "
        "seasons may be linked (same coach, players, arena).",
        f"  Lag-1 correlation of a team's residual with its next "
        f"season's residual: r = {lag1:.3f}. Subtracting the team mean "
        f"by itself pushes this correlation to about "
        f"-1/({n_seasons} - 1) = {-1 / (n_seasons - 1):.3f} even for "
        "fully independent seasons, so the observed value is what "
        "independence predicts.",
        f"  -> {holds(abs(lag1 + 1 / (n_seasons - 1)) < 0.1)}",
        "  Caveat: all teams share league-wide changes in a season (e.g. "
        "2021 has the lowest league HCA). That affects every team the "
        "same way, so it adds noise but does not favor any team.",
        "",
        "Backup if the assumptions fail: Kruskal-Wallis test "
        "(scipy.stats.kruskal), which compares ranks and does not need "
        "normality.",
        f"  H = {kw_stat:.3f}, p = {kw_p:.4f} -> {verdict(kw_p)}",
    ])


def welch_section(standings):
    """
    Runs the test for null hypothesis B (Denver and Utah have the same
    mean HCA as the other teams): Welch's t-test on team-seasons, plus
    a team-level Welch's t-test and a Mann-Whitney U test as checks.
    Returns the report text.
    """
    altitude, other = split_altitude(standings, ALTITUDE_TEAMS)
    t_stat, t_p = stats.ttest_ind(altitude, other, equal_var=False)
    one_sided = stats.ttest_ind(altitude, other, equal_var=False,
                                alternative="greater").pvalue
    student_p = stats.ttest_ind(altitude, other).pvalue
    var_p = stats.levene(altitude, other, center="median").pvalue
    sw_alt = stats.shapiro(altitude).pvalue
    mw_p = stats.mannwhitneyu(altitude, other,
                              alternative="two-sided").pvalue

    team_means = standings.groupby("TEAM_NAME")["HCA"].mean()
    perm_p = pair_permutation_p(team_means, ALTITUDE_TEAMS)
    n_teams = len(team_means)
    n_pairs = n_teams * (n_teams - 1) // 2
    rank = team_means.rank(ascending=False)

    return "\n".join([
        "7. RESULT VALIDITY - NULL HYPOTHESIS B",
        "H0: the mean HCA of Denver and Utah equals the mean HCA of the "
        "other 28 teams. H1: the means differ.",
        f"Altitude group: n = {len(altitude)} team-seasons, mean HCA = "
        f"{altitude.mean():.3f}, SD = {altitude.std():.3f}",
        f"Other teams:    n = {len(other)} team-seasons, mean HCA = "
        f"{other.mean():.3f}, SD = {other.std():.3f}",
        f"Difference in means: {altitude.mean() - other.mean():+.3f}",
        "Ranks of the team means (1 = largest HCA of 30): "
        + ", ".join(f"{t}: {int(rank[t])}" for t in ALTITUDE_TEAMS),
        "",
        "Planned test: Welch's t-test "
        "(scipy.stats.ttest_ind, equal_var=False).",
        f"  t = {t_stat:.3f}, two-sided p = {t_p:.6f} -> {verdict(t_p)}",
        f"  One-sided p (altitude HCA greater) = {one_sided:.6f}",
        f"  For comparison, Student's t-test (equal variances): "
        f"p = {student_p:.6f}",
        "Why Welch: the groups are very unequal in size "
        f"({len(altitude)} vs {len(other)}). Student's t-test assumes "
        "equal variances, and when the small group has a different "
        "variance its p-value can be badly wrong. Welch's test does not "
        "assume equal variances and loses almost nothing when they are "
        "equal.",
        "",
        "Assumption 1 - normality: the t-test compares means, and with "
        f"n = {len(altitude)} and {len(other)} the sample means are "
        "close to normal (central limit theorem).",
        f"  Shapiro-Wilk on the altitude group: p = {sw_alt:.4f} -> "
        f"{holds(sw_alt >= ALPHA)}",
        "Assumption 2 - equal variances: NOT needed for Welch. "
        f"(Levene's test between the two groups: p = {var_p:.4f}.)",
        "Assumption 3 - independence: the 32 altitude team-seasons come "
        "from only 2 franchises, so they are not 32 independent draws "
        "if a team's HCA is partly a stable team trait. Team-level "
        "check that treats each team as one observation: an exact "
        f"permutation test over all {n_pairs} pairs of teams. If "
        "altitude did not matter, Denver + Utah would be just a random "
        "pair.",
        f"  Share of pairs with an average mean HCA >= Denver + Utah's: "
        f"p = {perm_p:.4f} (one-sided) -> {verdict(perm_p)}",
        "  (A t-test on the 30 team means is not used: with only 2 "
        "teams in one group, their variance estimate is unreliable.)",
        "",
        "Backup if the assumptions fail: Mann-Whitney U test "
        "(two-group version of Kruskal-Wallis).",
        f"  p = {mw_p:.6f} -> {verdict(mw_p)}",
    ])


def tables_section(team_table, season_table):
    """
    Returns text with the two tables behind the figures: mean HCA by
    team and mean HCA by season.
    """
    fmt = "{:.3f}".format
    return "\n".join([
        "5. HCA TABLES",
        "Figure 10: mean HCA by team (largest first)",
        team_table.to_string(float_format=fmt),
        "",
        "Figure 12: league mean HCA by season",
        season_table.to_string(float_format=fmt),
    ])


def color_handles():
    """
    Returns legend entries (colored squares) for the two team groups:
    high-altitude teams and all other teams.
    """
    return [Patch(color=ALTITUDE_COLOR, label="High altitude (DEN, UTA)"),
            Patch(color=OTHER_COLOR, label="Other teams")]


def plot_team_hca(team_table, path):
    """
    Saves a horizontal bar chart of each team's mean HCA to path,
    largest at the top, with Denver and Utah in a different color and a
    dashed line at the league mean.
    """
    table = team_table.iloc[::-1]
    colors = [ALTITUDE_COLOR if t in ALTITUDE_TEAMS else OTHER_COLOR
              for t in table.index]
    league_mean = team_table["mean_hca"].mean()

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(table.index, table["mean_hca"], color=colors)
    ax.axvline(league_mean, color=MUTED_COLOR, linestyle="--",
               linewidth=1)
    handles = color_handles() + [
        Line2D([], [], color=MUTED_COLOR, linestyle="--",
               label=f"League mean ({league_mean:.3f})")]

    ax.set_title("Mean home-court advantage by team, 2004-2018 and 2021")
    ax.set_xlabel("Mean HCA (home win rate - road win rate)")
    ax.set_ylabel("Team")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.01)
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncols=3)
    fig.savefig(path)
    plt.close(fig)


def plot_team_boxplot(standings, team_table, path):
    """
    Saves box plots of the season HCA values for each team to path,
    ordered by mean HCA (largest at the top), with Denver and Utah in a
    different color.
    """
    order = list(team_table.index[::-1])
    data = [standings.loc[standings["TEAM_NAME"] == t, "HCA"]
            for t in order]

    fig, ax = plt.subplots(figsize=(8, 9))
    boxes = ax.boxplot(data, vert=False, patch_artist=True,
                       widths=0.6, medianprops={"color": TEXT_COLOR},
                       flierprops={"markersize": 3})
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)
    for box, team in zip(boxes["boxes"], order):
        color = ALTITUDE_COLOR if team in ALTITUDE_TEAMS else OTHER_COLOR
        box.set_facecolor(color)
        box.set_alpha(0.75)
    ax.axvline(0, color=MUTED_COLOR, linestyle="--", linewidth=1)

    ax.set_title("HCA per season by team (one value per season)")
    ax.set_xlabel("HCA (home win rate - road win rate)")
    ax.set_ylabel("Team")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=color_handles(), loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncols=2)
    fig.savefig(path)
    plt.close(fig)


def plot_season_hca(season_table, path):
    """
    Saves a line chart of the league mean HCA in each season to path.
    Seasons with no data (2019 and 2020) show as a gap.
    """
    seasons = range(season_table.index.min(),
                    season_table.index.max() + 1)
    hca = season_table["mean_hca"].reindex(seasons)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hca.index, hca.values, color=OTHER_COLOR, linewidth=2,
            marker="o", markersize=5, label="League mean HCA")
    ax.axhline(0, color=MUTED_COLOR, linestyle="--", linewidth=1,
               label="0 (no home-court advantage)")
    ax.text(2019.5, 0.02, "2019-20 and\n2020-21\nnot included",
            ha="center", fontsize=8, color=MUTED_COLOR)

    ax.set_title("League mean home-court advantage by season")
    ax.set_xlabel("Season (start year)")
    ax.set_ylabel("Mean HCA (home - road win rate)")
    ax.set_xticks(list(seasons)[::2])
    ax.set_ylim(-0.02, 0.25)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def plot_residual_qq(standings, path):
    """
    Saves a normal Q-Q plot of the ANOVA residuals (HCA minus team
    mean) to path. Points close to the line mean the residuals are
    roughly normal.
    """
    residuals = team_residuals(standings)
    (theory, ordered), (slope, intercept, _) = stats.probplot(residuals)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(theory, ordered, s=10, color=OTHER_COLOR, alpha=0.7,
               label="Residuals")
    ax.plot(theory, slope * theory + intercept, color=ALTITUDE_COLOR,
            linewidth=1.5, label="Normal reference line")
    ax.set_title("Q-Q plot of HCA residuals (HCA - team mean)")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Residual HCA")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def main():
    """
    Runs the RQ4 EDA: loads the raw and cleaned ranking and teams data,
    builds final standings with HCA, runs the checks and tests, saves
    four figures to FIGURES_DIR, then prints the report and writes it
    to RESULTS_PATH.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    set_plot_style()

    raw_ranking = load_csv(RAW_RANKING_PATH)
    raw_teams = load_csv(RAW_TEAMS_PATH)
    ranking = load_csv(CLEAN_RANKING_PATH)
    ranking["STANDINGSDATE"] = pd.to_datetime(ranking["STANDINGSDATE"])
    teams = load_csv(CLEAN_TEAMS_PATH)

    standings = add_hca(final_standings(ranking, NORMAL_SEASONS))
    standings, unmatched = merge_teams(standings, teams)
    unusual = unusual_game_counts(standings)
    team_table = team_hca_table(standings)
    season_table = season_hca_table(standings)

    figure_paths = [os.path.join(FIGURES_DIR, name) for name in
                    ["fig10_mean_hca_by_team.png",
                     "fig11_hca_boxplot_by_team.png",
                     "fig12_league_hca_by_season.png",
                     "fig13_hca_residual_qq.png"]]
    plot_team_hca(team_table, figure_paths[0])
    plot_team_boxplot(standings, team_table, figure_paths[1])
    plot_season_hca(season_table, figure_paths[2])
    plot_residual_qq(standings, figure_paths[3])

    sections = [
        size_section(raw_ranking, ranking, raw_teams),
        missing_section(raw_ranking, raw_teams),
        checks_section(standings, unusual, unmatched),
        variables_section(standings),
        tables_section(team_table, season_table),
        anova_section(standings),
        welch_section(standings),
        "8. FIGURES\n" + "\n".join(figure_paths),
    ]
    report = "\n\n".join(sections) + "\n"
    print(report)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
