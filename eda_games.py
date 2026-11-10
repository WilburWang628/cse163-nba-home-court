"""
CSE 163 Final Project: NBA home-court advantage.

Exploratory data analysis of games.csv. Prints and saves to
results/eda_games.txt:
- the size of the raw and cleaned data
- missing values in the raw data and how cleaning handles them
- summaries of the variables used in the research questions
- home win rates by season and by PERIOD x GAME_TYPE
and saves three figures to figures/.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator, PercentFormatter

from clean import load_games, clean_games

RAW_PATH = "data/games.csv"
FIGURES_DIR = "figures"
RESULTS_PATH = "results/eda_games.txt"

NUMERIC_COLUMNS = ["POINT_DIFF", "PTS_home", "PTS_away",
                   "FG_PCT_home", "FG_PCT_away", "FT_PCT_home",
                   "FT_PCT_away", "REB_home", "REB_away", "SEASON"]
CATEGORICAL_COLUMNS = ["GAME_TYPE", "PERIOD", "HOME_TEAM_WINS"]

PERIOD_ORDER = ["normal", "covid_2021", "bubble"]
PERIOD_NAMES = {"normal": "Normal (2004-19, 2021-22)",
                "covid_2021": "COVID 2020-21",
                "bubble": "2020 bubble"}
GAME_TYPE_NAMES = {2: "Regular season", 4: "Playoffs"}

# z value for a two-sided 95% confidence interval.
Z_95 = 1.96

# Colors: categorical slots 1-3 for series, neutral inks for text and
# reference lines.
PERIOD_COLORS = {"normal": "#2a78d6", "covid_2021": "#eb6834",
                 "bubble": "#1baf7a"}
GAME_TYPE_COLORS = {2: "#2a78d6", 4: "#eb6834"}
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def set_plot_style():
    """
    Sets shared matplotlib settings so every figure has the same fonts,
    light grid lines and no top / right borders.
    """
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": MUTED_COLOR,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED_COLOR,
        "ytick.color": MUTED_COLOR,
        "legend.frameon": False,
    })


def size_section(raw, clean):
    """
    Returns text describing how many rows and columns the raw and
    cleaned games data have and what one row represents.
    """
    return "\n".join([
        "1. DATA SIZE",
        f"Raw games.csv: {raw.shape[0]} rows x {raw.shape[1]} columns",
        f"Cleaned games: {clean.shape[0]} rows x {clean.shape[1]} columns",
        "Each row is one NBA game, with the home team's and the away "
        "team's box-score totals side by side.",
        "Cleaning removed rows with missing scores, repeated GAME_IDs, "
        "preseason games and play-in games, and added GAME_TYPE, "
        "POINT_DIFF, BUBBLE and PERIOD.",
        f"Cleaned data covers {clean['GAME_DATE_EST'].min().date()} to "
        f"{clean['GAME_DATE_EST'].max().date()} "
        f"(SEASON {clean['SEASON'].min()} to {clean['SEASON'].max()}).",
    ])


def missing_table(df):
    """
    Returns a DataFrame with the number and percent of missing values in
    each column of df, keeping only columns that have any missing values.
    """
    counts = df.isna().sum()
    table = pd.DataFrame({"missing": counts,
                          "percent": counts / len(df) * 100})
    return table[table["missing"] > 0]


def missing_section(raw, clean):
    """
    Returns text listing missing values in the raw data, describing
    which games they belong to, and confirming the cleaned data has no
    missing values.
    """
    table = missing_table(raw)
    missing_rows = raw[raw["PTS_home"].isna()]
    types = missing_rows["GAME_ID"].astype(str).str[0].value_counts()
    statuses = missing_rows["GAME_STATUS_TEXT"].value_counts()
    clean_missing = clean.isna().sum().sum()

    lines = [
        "2. MISSING VALUES (raw games.csv)",
        f"Total rows: {len(raw)}. Columns with missing values:",
        table.to_string(float_format="{:.2f}".format),
        f"All other {raw.shape[1] - len(table)} columns have no missing "
        "values.",
        f"The missing values all come from the same {len(missing_rows)} "
        "rows: every box-score column (points, shooting percentages, "
        "assists, rebounds) is empty for both teams.",
        f"These games were played from {missing_rows['GAME_DATE_EST'].min()}"
        f" to {missing_rows['GAME_DATE_EST'].max()}. GAME_ID first digit "
        f"counts: {types.to_dict()} (1 = preseason).",
        f"GAME_STATUS_TEXT for these rows: {statuses.to_dict()}, so they "
        "are finished games whose box scores were not recorded, not "
        "unfinished games.",
        "Cleaning drops these rows (dropna on PTS_home). They are all "
        "preseason games, which the analysis excludes anyway, so no "
        "regular-season or playoff game is lost.",
        f"Missing values in the cleaned data: {clean_missing}.",
    ]
    return "\n".join(lines)


def numeric_summary(df, columns):
    """
    Returns a DataFrame with the seven-number summary (mean, std, min,
    Q1, median, Q3, max) of each column in columns, one row per column.
    """
    summary = df[columns].describe().T
    summary = summary.rename(columns={"25%": "Q1", "50%": "median",
                                      "75%": "Q3"})
    return summary[["mean", "std", "min", "Q1", "median", "Q3", "max"]]


def variables_section(clean):
    """
    Returns text with the seven-number summary of the numeric research
    variables and the value counts of the categorical ones, computed on
    the cleaned data.
    """
    lines = [
        "3. RESEARCH VARIABLES (cleaned data, all "
        f"{len(clean)} games)",
        "Numeric variables:",
        numeric_summary(clean, NUMERIC_COLUMNS).to_string(
            float_format="{:.3f}".format),
        "",
        "Categorical variables:",
    ]
    for column in CATEGORICAL_COLUMNS:
        counts = clean[column].value_counts().sort_index()
        lines.append(f"{column}:")
        for value, count in counts.items():
            share = count / len(clean) * 100
            lines.append(f"  {value}: {count} ({share:.1f}%)")
    lines.append("GAME_TYPE: 2 = regular season, 4 = playoffs. "
                 "HOME_TEAM_WINS: 1 = home team won, 0 = home team lost.")
    return "\n".join(lines)


def win_rate_table(df, group_columns):
    """
    Groups df by group_columns and returns a DataFrame with the number
    of games, home wins, home win rate and its 95% confidence interval
    (normal approximation: rate +/- 1.96 * sqrt(rate * (1 - rate) / n))
    for each group.
    """
    table = df.groupby(group_columns)["HOME_TEAM_WINS"].agg(
        games="count", home_wins="sum").reset_index()
    rate = table["home_wins"] / table["games"]
    margin = Z_95 * (rate * (1 - rate) / table["games"]) ** 0.5
    table["win_rate"] = rate
    table["ci_low"] = rate - margin
    table["ci_high"] = rate + margin
    return table


def period_win_rates(clean):
    """
    Returns the home win rate table for the normal, covid_2021 and
    bubble periods, split by GAME_TYPE, in PERIOD_ORDER.
    """
    df = clean[clean["PERIOD"].isin(PERIOD_ORDER)]
    table = win_rate_table(df, ["PERIOD", "GAME_TYPE"])
    table["PERIOD"] = pd.Categorical(table["PERIOD"], PERIOD_ORDER)
    return table.sort_values(["PERIOD", "GAME_TYPE"]).reset_index(drop=True)


def point_diff_table(clean):
    """
    Returns a DataFrame with the number of games and the mean, standard
    deviation and median of POINT_DIFF for each of the three periods
    (regular season and playoffs together).
    """
    df = clean[clean["PERIOD"].isin(PERIOD_ORDER)]
    table = df.groupby("PERIOD")["POINT_DIFF"].agg(
        ["count", "mean", "std", "median"])
    return table.reindex(PERIOD_ORDER)


def win_rate_section(season_table, period_table, diff_table):
    """
    Returns text with the three tables behind the figures: home win
    rate by season, home win rate by PERIOD x GAME_TYPE, and POINT_DIFF
    by PERIOD.
    """
    fmt = "{:.3f}".format
    return "\n".join([
        "4. HOME-COURT ADVANTAGE TABLES",
        "Figure 1: home win rate by season (normal regular-season games)",
        season_table.to_string(index=False, float_format=fmt),
        "",
        "Figure 2: home win rate by PERIOD and GAME_TYPE "
        "(95% CI, normal approximation)",
        period_table.to_string(index=False, float_format=fmt),
        "",
        "Figure 3: POINT_DIFF (home - away) by PERIOD "
        "(regular season and playoffs)",
        diff_table.to_string(float_format=fmt),
    ])


def plot_season_win_rate(season_table, path):
    """
    Saves a line chart of the home win rate in each normal regular
    season to path, with a dashed reference line at 50% (no home-court
    advantage). Seasons with no normal games show as a gap in the line.
    """
    seasons = range(season_table["SEASON"].min(),
                    season_table["SEASON"].max() + 1)
    rates = season_table.set_index("SEASON")["win_rate"].reindex(seasons)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rates.index, rates.values, color=PERIOD_COLORS["normal"],
            linewidth=2, marker="o", markersize=5,
            label="Home win rate")
    ax.axhline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1,
               label="50% (no home-court advantage)")
    ax.axvspan(2018.5, 2020.5, color=GRID_COLOR, alpha=0.6, linewidth=0)
    ax.text(2019.5, 0.515, "2019-20 and\n2020-21\nnot 'normal'",
            ha="center", va="bottom", fontsize=8, color=MUTED_COLOR)

    ax.set_title("Home win rate by season, normal regular-season games")
    ax.set_xlabel("Season (start year)")
    ax.set_ylabel("Home team win rate")
    ax.set_xticks(list(seasons)[::2])
    ax.set_ylim(0.45, 0.65)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_period_win_rate(period_table, path):
    """
    Saves a grouped bar chart of the home win rate in the normal,
    covid_2021 and bubble periods to path, with regular-season and
    playoff bars side by side and 95% confidence interval error bars.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    positions = range(len(PERIOD_ORDER))
    for i, game_type in enumerate(GAME_TYPE_NAMES):
        rows = period_table[period_table["GAME_TYPE"] == game_type]
        rows = rows.set_index("PERIOD").reindex(PERIOD_ORDER)
        x = [p + (i - 0.5) * width for p in positions]
        errors = [rows["win_rate"] - rows["ci_low"],
                  rows["ci_high"] - rows["win_rate"]]
        ax.bar(x, rows["win_rate"], width=width - 0.03,
               color=GAME_TYPE_COLORS[game_type],
               label=GAME_TYPE_NAMES[game_type])
        ax.errorbar(x, rows["win_rate"], yerr=errors, fmt="none",
                    ecolor=TEXT_COLOR, elinewidth=1.2, capsize=4)
    ax.axhline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1,
               label="50% (no home-court advantage)")

    ax.set_title("Home win rate by period (error bars: 95% CI)")
    ax.set_xlabel("Period")
    ax.set_ylabel("Home team win rate")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([PERIOD_NAMES[p] for p in PERIOD_ORDER])
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper right", ncols=3, bbox_to_anchor=(1, 1.0))
    fig.savefig(path)
    plt.close(fig)


def plot_point_diff(clean, diff_table, path):
    """
    Saves overlapping kernel density curves of POINT_DIFF for the
    normal, covid_2021 and bubble periods to path. A dashed line marks
    0 (tie) and each legend entry shows the period's mean.
    """
    df = clean[clean["PERIOD"].isin(PERIOD_ORDER)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for period in PERIOD_ORDER:
        values = df.loc[df["PERIOD"] == period, "POINT_DIFF"]
        n = int(diff_table.loc[period, "count"])
        mean = diff_table.loc[period, "mean"]
        sns.kdeplot(values, ax=ax, color=PERIOD_COLORS[period],
                    linewidth=2,
                    label=f"{PERIOD_NAMES[period]} "
                          f"(n={n}, mean={mean:+.2f})")
    ax.axvline(0, color=MUTED_COLOR, linestyle="--", linewidth=1)

    ax.set_title("Distribution of home point differential by period")
    ax.set_xlabel("Point differential (home points - away points)")
    ax.set_ylabel("Density")
    ax.set_xlim(-50, 50)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              title="Period")
    fig.savefig(path)
    plt.close(fig)


def main():
    """
    Runs the EDA: loads and cleans games.csv, builds the summary text
    and tables, saves the three figures to FIGURES_DIR, then prints the
    results and writes them to RESULTS_PATH.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    set_plot_style()

    raw = load_games(RAW_PATH)
    clean = clean_games(raw)

    normal_regular = clean[(clean["PERIOD"] == "normal")
                           & (clean["GAME_TYPE"] == 2)]
    season_table = win_rate_table(normal_regular, ["SEASON"])
    period_table = period_win_rates(clean)
    diff_table = point_diff_table(clean)

    figure_paths = [os.path.join(FIGURES_DIR, name) for name in
                    ["fig1_home_win_rate_by_season.png",
                     "fig2_home_win_rate_by_period.png",
                     "fig3_point_diff_by_period.png"]]
    plot_season_win_rate(season_table, figure_paths[0])
    plot_period_win_rate(period_table, figure_paths[1])
    plot_point_diff(clean, diff_table, figure_paths[2])

    sections = [
        size_section(raw, clean),
        missing_section(raw, clean),
        variables_section(clean),
        win_rate_section(season_table, period_table, diff_table),
        "5. FIGURES\n" + "\n".join(figure_paths),
    ]
    report = "\n\n".join(sections) + "\n"
    print(report)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
