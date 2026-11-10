"""
CSE 163 Final Project: NBA home-court advantage.

Data cleaning for the Kaggle dataset nathanlauga/nba-games. Reads
games.csv, games_details.csv, ranking.csv and teams.csv from the data
folder and writes a cleaned version of each:
- games_clean.csv: one row per regular-season or playoff game, plus
  GAME_TYPE, POINT_DIFF, BUBBLE and PERIOD
- games_details_clean.csv: one row per player who played in those games,
  with minutes as a number and HOME / BUBBLE flags
- ranking_clean.csv: daily regular-season standings with numeric
  home / road wins and losses
- teams_clean.csv: one row per team with name and arena information
"""

import os

import pandas as pd

DATA_DIR = "data"

# First digit of GAME_ID: 1 = preseason, 2 = regular season,
# 4 = playoffs, 5 = play-in.
REGULAR_SEASON = 2
PLAYOFFS = 4
KEEP_TYPES = [REGULAR_SEASON, PLAYOFFS]

# The 2020 "bubble" in Orlando (no fans), restart through the Finals.
BUBBLE_START = "2020-07-30"
BUBBLE_END = "2020-10-11"

# games.csv starts with the 2003-04 season.
FIRST_SEASON = 2003

# Seasons used for the PERIOD label (see add_period). SEASON 2020 is the
# 2020-21 season, played with few or no fans.
NORMAL_SEASONS = list(range(2004, 2019)) + [2021]
COVID_SEASON = 2020

TEAM_COLUMNS = ["TEAM_ID", "ABBREVIATION", "CITY", "NICKNAME",
                "ARENA", "ARENACAPACITY"]
GAME_COLUMNS_FOR_DETAILS = ["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID",
                            "SEASON", "GAME_TYPE", "BUBBLE"]


def load_games(path):
    """
    Reads the games CSV file at the given path and returns it as a
    pandas DataFrame.
    """
    return pd.read_csv(path)


def load_csv(path):
    """
    Reads any CSV file at the given path and returns it as a pandas
    DataFrame. low_memory=False makes pandas read each column in one
    pass, which avoids mixed-type warnings on the large details file.
    """
    return pd.read_csv(path, low_memory=False)


def clean_games(df):
    """
    Takes the raw games DataFrame and returns a cleaned copy:
    - drops rows where PTS_home is missing
    - drops repeated GAME_IDs, keeping the first occurrence
    - converts GAME_DATE_EST to a datetime column
    - adds GAME_TYPE (first digit of GAME_ID), POINT_DIFF
      (PTS_home - PTS_away) and BUBBLE (True if the game date is
      between BUBBLE_START and BUBBLE_END, inclusive)
    - keeps only regular-season and playoff games
    - adds PERIOD (see add_period)
    The input DataFrame is not modified.
    """
    df = df.dropna(subset=["PTS_home"])
    df = df.drop_duplicates(subset="GAME_ID", keep="first")
    df = df.copy()

    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
    df["GAME_TYPE"] = df["GAME_ID"].astype(str).str[0].astype(int)
    df["POINT_DIFF"] = df["PTS_home"] - df["PTS_away"]
    df["BUBBLE"] = df["GAME_DATE_EST"].between(BUBBLE_START, BUBBLE_END)

    df = df[df["GAME_TYPE"].isin(KEEP_TYPES)]
    df = add_period(df)
    return df.reset_index(drop=True)


def add_period(df):
    """
    Takes a games DataFrame with SEASON and BUBBLE columns and returns a
    copy with a PERIOD column:
    - "bubble": BUBBLE is True (2020 Orlando bubble)
    - "covid_2021": SEASON is COVID_SEASON (the 2020-21 season)
    - "normal": SEASON is in NORMAL_SEASONS (2004-2018 and 2021)
    - "exclude": everything else, i.e. SEASON 2003, SEASON 2022
      (incomplete) and the non-bubble games of SEASON 2019
    The input DataFrame is not modified.
    """
    df = df.copy()
    df["PERIOD"] = "exclude"
    df.loc[df["SEASON"].isin(NORMAL_SEASONS), "PERIOD"] = "normal"
    df.loc[df["SEASON"] == COVID_SEASON, "PERIOD"] = "covid_2021"
    df.loc[df["BUBBLE"], "PERIOD"] = "bubble"
    return df


def clean_teams(df):
    """
    Takes the raw teams DataFrame and returns a cleaned copy that keeps
    only the columns in TEAM_COLUMNS and adds TEAM_NAME (CITY plus
    NICKNAME, e.g. "Los Angeles Lakers"). Missing ARENACAPACITY values
    are left as missing. The input DataFrame is not modified.
    """
    df = df[TEAM_COLUMNS].copy()
    df["TEAM_NAME"] = df["CITY"] + " " + df["NICKNAME"]
    return df.reset_index(drop=True)


def split_record(records):
    """
    Takes a Series of win-loss strings such as "13-2" and returns two
    integer Series: the wins and the losses.
    """
    parts = records.str.split("-", expand=True)
    return parts[0].astype(int), parts[1].astype(int)


def clean_ranking(df):
    """
    Takes the raw ranking DataFrame and returns a cleaned copy:
    - keeps only regular-season standings (SEASON_ID starts with 2)
    - adds SEASON (last four digits of SEASON_ID, matching the SEASON
      column in games.csv) and keeps seasons from FIRST_SEASON on
    - drops repeated rows for the same team, season and date
    - converts STANDINGSDATE to a datetime column
    - adds HOME_W, HOME_L, ROAD_W and ROAD_L as integers parsed from
      HOME_RECORD and ROAD_RECORD
    The input DataFrame is not modified.
    """
    df = df[df["SEASON_ID"] // 10000 == REGULAR_SEASON].copy()
    df["SEASON"] = df["SEASON_ID"] % 10000
    df = df[df["SEASON"] >= FIRST_SEASON]
    df = df.drop_duplicates(subset=["TEAM_ID", "SEASON_ID",
                                    "STANDINGSDATE"])
    df = df.copy()

    df["STANDINGSDATE"] = pd.to_datetime(df["STANDINGSDATE"])
    df["HOME_W"], df["HOME_L"] = split_record(df["HOME_RECORD"])
    df["ROAD_W"], df["ROAD_L"] = split_record(df["ROAD_RECORD"])
    return df.reset_index(drop=True)


def parse_minutes(value):
    """
    Converts one MIN value from games_details.csv into minutes played as
    a float. Handles the formats found in the data: "24:56",
    "24.000000:56" and "24". Returns NaN for missing or negative values.
    """
    if pd.isna(value):
        return float("nan")
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":")
        result = float(minutes) + float(seconds) / 60
    else:
        result = float(text)
    if result < 0:
        return float("nan")
    return result


def clean_details(details, games):
    """
    Takes the raw games_details DataFrame and an already cleaned games
    DataFrame (from clean_games) and returns a cleaned copy of details:
    - drops players who did not play (PTS is missing)
    - drops repeated rows for the same GAME_ID and PLAYER_ID
    - converts MIN to minutes as a float (see parse_minutes)
    - keeps only games that appear in the cleaned games, adding their
      SEASON, GAME_TYPE and BUBBLE columns
    - drops rows whose team is neither the home nor the visiting team
    - adds HOME (True if the player's team was the home team)
    Neither input DataFrame is modified.
    """
    df = details.dropna(subset=["PTS"])
    df = df.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], keep="first")
    df = df.copy()
    df["MIN"] = df["MIN"].apply(parse_minutes)

    df = df.merge(games[GAME_COLUMNS_FOR_DETAILS], on="GAME_ID")
    in_game = ((df["TEAM_ID"] == df["HOME_TEAM_ID"])
               | (df["TEAM_ID"] == df["VISITOR_TEAM_ID"]))
    df = df[in_game].copy()
    df["HOME"] = df["TEAM_ID"] == df["HOME_TEAM_ID"]

    df = df.drop(columns=["HOME_TEAM_ID", "VISITOR_TEAM_ID"])
    return df.reset_index(drop=True)


def main(data_dir=DATA_DIR):
    """
    Loads the four raw CSV files from data_dir, cleans each one, prints
    row counts before and after cleaning (plus the number of bubble
    games), and saves the cleaned files back into data_dir with a
    "_clean" suffix.
    """
    raw_games = load_games(os.path.join(data_dir, "games.csv"))
    raw_details = load_csv(os.path.join(data_dir, "games_details.csv"))
    raw_ranking = load_csv(os.path.join(data_dir, "ranking.csv"))
    raw_teams = load_csv(os.path.join(data_dir, "teams.csv"))

    games = clean_games(raw_games)
    details = clean_details(raw_details, games)
    ranking = clean_ranking(raw_ranking)
    teams = clean_teams(raw_teams)

    print(f"Rows before cleaning: {len(raw_games)}")
    print(f"Rows after cleaning: {len(games)}")
    print(f"Bubble games: {games['BUBBLE'].sum()}")
    print("Games by PERIOD and GAME_TYPE (2 = regular season, "
          "4 = playoffs):")
    print(pd.crosstab(games["PERIOD"], games["GAME_TYPE"], margins=True))
    print(f"games_details rows: {len(raw_details)} -> {len(details)}")
    print(f"ranking rows: {len(raw_ranking)} -> {len(ranking)}")
    print(f"teams rows: {len(raw_teams)} -> {len(teams)}")

    outputs = {"games_clean.csv": games,
               "games_details_clean.csv": details,
               "ranking_clean.csv": ranking,
               "teams_clean.csv": teams}
    for name, df in outputs.items():
        df.to_csv(os.path.join(data_dir, name), index=False)
    print(f"Saved cleaned files to {data_dir}/")


if __name__ == "__main__":
    main()
