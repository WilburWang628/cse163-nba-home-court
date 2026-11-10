"""
CSE 163 Final Project: NBA home-court advantage.

Tests for clean.py. Each test builds a tiny hand-made DataFrame whose
correct result can be worked out by hand.
"""

import contextlib
import io
import math
import os
import tempfile

import pandas as pd

from clean import (load_games, load_csv, clean_games, add_period,
                   clean_teams, split_record, clean_ranking, parse_minutes,
                   clean_details, main)


def make_raw_games():
    """
    Returns a 6-row raw games DataFrame for testing clean_games:
    - row 0: normal regular-season game, should be kept
    - row 1: repeats row 0's GAME_ID, should be dropped
    - row 2: regular-season game on the first bubble day, kept
    - row 3: playoff game on the last bubble day, kept
    - row 4: preseason game (GAME_ID starts with 1), dropped
    - row 5: regular-season game with missing PTS_home, dropped
    """
    return pd.DataFrame({
        "GAME_DATE_EST": ["2020-01-01", "2020-01-01", "2020-07-30",
                          "2020-10-11", "2020-10-12", "2021-01-05"],
        "GAME_ID": [22000001, 22000001, 21900002,
                    41900003, 12000004, 22000005],
        "SEASON": [2019, 2019, 2019, 2019, 2020, 2020],
        "PTS_home": [110.0, 999.0, 100.0, 120.0, 90.0, math.nan],
        "PTS_away": [100.0, 0.0, 105.0, 101.0, 80.0, 100.0],
    })


def make_raw_teams():
    """
    Returns a 4-row raw teams DataFrame. It has extra columns (OWNER,
    HEADCOACH) that clean_teams should drop, and one missing
    ARENACAPACITY that should stay missing.
    """
    return pd.DataFrame({
        "TEAM_ID": [1, 2, 3, 4],
        "ABBREVIATION": ["LAL", "BOS", "PHX", "MIA"],
        "CITY": ["Los Angeles", "Boston", "Phoenix", "Miami"],
        "NICKNAME": ["Lakers", "Celtics", "Suns", "Heat"],
        "ARENA": ["Arena A", "Arena B", "Arena C", "Arena D"],
        "ARENACAPACITY": [19060.0, 18624.0, math.nan, 19600.0],
        "OWNER": ["a", "b", "c", "d"],
        "HEADCOACH": ["e", "f", "g", "h"],
    })


def make_raw_ranking():
    """
    Returns a 6-row raw ranking DataFrame for testing clean_ranking:
    - row 0: team 1, 2019 regular season, kept
    - row 1: same team, season and date as row 0 (only W_PCT differs),
      dropped as a duplicate
    - row 2: preseason standings (SEASON_ID starts with 1), dropped
    - row 3: 2002 season, before FIRST_SEASON, dropped
    - row 4: team 2, 2019 regular season, kept
    - row 5: team 1, 2020 regular season, kept
    """
    return pd.DataFrame({
        "TEAM_ID": [1, 1, 2, 1, 2, 1],
        "SEASON_ID": [22019, 22019, 12019, 22002, 22019, 22020],
        "STANDINGSDATE": ["2020-08-01", "2020-08-01", "2019-10-10",
                          "2003-10-02", "2020-08-01", "2021-01-01"],
        "W_PCT": [0.583, 0.5833, 0.5, 0.6, 0.454, 0.5],
        "HOME_RECORD": ["20-10", "20-10", "1-1", "30-11", "7-12", "1-0"],
        "ROAD_RECORD": ["15-15", "15-15", "0-1", "20-21", "3-0", "0-1"],
    })


def make_clean_games_for_details():
    """
    Returns a 2-row DataFrame shaped like the output of clean_games,
    with only the columns that clean_details needs. Game 22000001 is
    team 10 (home) against team 20; game 42000002 is a bubble playoff
    game with team 30 at home against team 10.
    """
    return pd.DataFrame({
        "GAME_ID": [22000001, 42000002],
        "HOME_TEAM_ID": [10, 30],
        "VISITOR_TEAM_ID": [20, 10],
        "SEASON": [2020, 2019],
        "GAME_TYPE": [2, 4],
        "BUBBLE": [False, True],
    })


def make_raw_details():
    """
    Returns a 6-row raw games_details DataFrame for testing
    clean_details together with make_clean_games_for_details:
    - row 0: player 101 (team 10, home) in game 22000001, kept
    - row 1: same player and game as row 0, dropped as a duplicate
    - row 2: player 201 did not play (PTS missing), dropped
    - row 3: player 101 (team 10, away) in game 42000002, kept
    - row 4: game 12000003 is not in the cleaned games, dropped
    - row 5: team 40 did not play in game 42000002, dropped
    """
    return pd.DataFrame({
        "GAME_ID": [22000001, 22000001, 22000001,
                    42000002, 12000003, 42000002],
        "TEAM_ID": [10, 10, 20, 10, 10, 40],
        "PLAYER_ID": [101, 101, 201, 101, 102, 401],
        "MIN": ["30:30", "30:30", math.nan, "12", "20:00", "5:00"],
        "PTS": [20.0, 20.0, math.nan, 4.0, 8.0, 2.0],
    })


def test_load_games():
    """
    Writes a 4-row CSV to a temporary folder and checks that load_games
    reads back the same shape and values.
    """
    df = pd.DataFrame({
        "GAME_ID": [22000001, 22000002, 42000003, 12000004],
        "PTS_home": [110, 95, 120, 88],
        "PTS_away": [100, 99, 101, 90],
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "games.csv")
        df.to_csv(path, index=False)
        result = load_games(path)

    assert result.shape == (4, 3)
    assert list(result.columns) == ["GAME_ID", "PTS_home", "PTS_away"]
    assert list(result["GAME_ID"]) == [22000001, 22000002, 42000003, 12000004]
    assert list(result["PTS_home"]) == [110, 95, 120, 88]


def test_load_csv():
    """
    Writes a 4-row CSV whose MIN column mixes text and numbers (like
    games_details.csv) and checks that load_csv reads it back intact.
    """
    df = pd.DataFrame({
        "PLAYER_ID": [1, 2, 3, 4],
        "MIN": ["24:56", "18", "", "30:00"],
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "details.csv")
        df.to_csv(path, index=False)
        result = load_csv(path)

    assert result.shape == (4, 2)
    assert list(result["PLAYER_ID"]) == [1, 2, 3, 4]
    assert result.loc[0, "MIN"] == "24:56"
    assert pd.isna(result.loc[2, "MIN"])


def test_clean_games():
    """
    Checks that clean_games keeps exactly rows 0, 2 and 3 of
    make_raw_games and computes GAME_TYPE, POINT_DIFF and BUBBLE
    correctly for them.
    """
    raw = make_raw_games()
    result = clean_games(raw)

    # Duplicate, preseason and missing-score rows are gone.
    assert list(result["GAME_ID"]) == [22000001, 21900002, 41900003]
    # The first copy of the duplicate GAME_ID is the one kept.
    assert result.loc[0, "PTS_home"] == 110.0

    assert pd.api.types.is_datetime64_any_dtype(result["GAME_DATE_EST"])
    assert result.loc[0, "GAME_DATE_EST"] == pd.Timestamp("2020-01-01")

    assert list(result["GAME_TYPE"]) == [2, 2, 4]
    # 110 - 100, 100 - 105, 120 - 101
    assert list(result["POINT_DIFF"]) == [10.0, -5.0, 19.0]
    # Both bubble boundary dates count as bubble games.
    assert list(result["BUBBLE"]) == [False, True, True]
    # Non-bubble SEASON 2019 games are excluded.
    assert list(result["PERIOD"]) == ["exclude", "bubble", "bubble"]

    # The original DataFrame is not changed.
    assert len(raw) == 6


def test_add_period():
    """
    Checks add_period on a 7-row DataFrame that covers all four PERIOD
    labels, including both excluded edge seasons (2003 and 2022) and
    the bubble / non-bubble split inside SEASON 2019.
    """
    df = pd.DataFrame({
        "SEASON": [2003, 2004, 2019, 2019, 2020, 2021, 2022],
        "BUBBLE": [False, False, True, False, False, False, False],
    })
    result = add_period(df)

    assert list(result["PERIOD"]) == ["exclude", "normal", "bubble",
                                      "exclude", "covid_2021", "normal",
                                      "exclude"]
    assert "PERIOD" not in df.columns


def test_clean_teams():
    """
    Checks that clean_teams keeps the 4 teams, drops OWNER and
    HEADCOACH, builds TEAM_NAME and leaves the missing capacity missing.
    """
    raw = make_raw_teams()
    result = clean_teams(raw)

    assert list(result.columns) == ["TEAM_ID", "ABBREVIATION", "CITY",
                                    "NICKNAME", "ARENA", "ARENACAPACITY",
                                    "TEAM_NAME"]
    assert list(result["TEAM_NAME"]) == ["Los Angeles Lakers",
                                         "Boston Celtics", "Phoenix Suns",
                                         "Miami Heat"]
    assert pd.isna(result.loc[2, "ARENACAPACITY"])
    assert result.loc[0, "ARENACAPACITY"] == 19060.0
    assert "OWNER" in raw.columns


def test_split_record():
    """
    Checks that split_record turns "W-L" strings into integer wins and
    losses, including 0-0 and multi-digit records.
    """
    records = pd.Series(["13-2", "0-0", "41-0", "7-12"])
    wins, losses = split_record(records)

    assert list(wins) == [13, 0, 41, 7]
    assert list(losses) == [2, 0, 0, 12]


def test_clean_ranking():
    """
    Checks that clean_ranking keeps exactly rows 0, 4 and 5 of
    make_raw_ranking and computes SEASON and the home / road win and
    loss columns for them.
    """
    raw = make_raw_ranking()
    result = clean_ranking(raw)

    assert list(result["TEAM_ID"]) == [1, 2, 1]
    assert list(result["SEASON"]) == [2019, 2019, 2020]
    # The first copy of the duplicate row is the one kept.
    assert result.loc[0, "W_PCT"] == 0.583

    assert pd.api.types.is_datetime64_any_dtype(result["STANDINGSDATE"])
    assert result.loc[2, "STANDINGSDATE"] == pd.Timestamp("2021-01-01")

    # "20-10", "7-12", "1-0" and "15-15", "3-0", "0-1"
    assert list(result["HOME_W"]) == [20, 7, 1]
    assert list(result["HOME_L"]) == [10, 12, 0]
    assert list(result["ROAD_W"]) == [15, 3, 0]
    assert list(result["ROAD_L"]) == [15, 0, 1]

    assert len(raw) == 6


def test_parse_minutes():
    """
    Checks parse_minutes on each MIN format found in games_details.csv,
    plus negative and missing values.
    """
    assert parse_minutes("24:30") == 24.5
    assert parse_minutes("0:15") == 0.25
    assert parse_minutes("29.000000:45") == 29.75
    assert parse_minutes("18") == 18.0
    assert math.isnan(parse_minutes("-9"))
    assert math.isnan(parse_minutes(math.nan))


def test_clean_details():
    """
    Checks that clean_details keeps exactly rows 0 and 3 of
    make_raw_details, converts MIN, and adds the right HOME, BUBBLE,
    SEASON and GAME_TYPE values from the games table.
    """
    raw = make_raw_details()
    games = make_clean_games_for_details()
    result = clean_details(raw, games)

    assert list(result["GAME_ID"]) == [22000001, 42000002]
    assert list(result["PLAYER_ID"]) == [101, 101]
    # "30:30" is 30.5 minutes, "12" is 12 minutes.
    assert list(result["MIN"]) == [30.5, 12.0]
    # Team 10 is at home in the first game and away in the second.
    assert list(result["HOME"]) == [True, False]
    assert list(result["BUBBLE"]) == [False, True]
    assert list(result["SEASON"]) == [2020, 2019]
    assert list(result["GAME_TYPE"]) == [2, 4]
    assert "HOME_TEAM_ID" not in result.columns

    assert len(raw) == 6
    assert raw.loc[0, "MIN"] == "30:30"


def test_main():
    """
    Runs main on a temporary data folder with 4 rows in each raw file
    and checks the printed counts and the saved files.
    - games: the play-in game is dropped, and the games one day before
      and one day after the bubble are not bubble games, so 3 rows
      remain and 1 is a bubble game
    - details: the play-in row and the did-not-play row are dropped
    - ranking: the preseason row and the duplicate row are dropped
    - teams: all 4 rows are kept
    """
    games = pd.DataFrame({
        "GAME_DATE_EST": ["2020-07-29", "2020-08-01",
                          "2020-08-15", "2020-10-12"],
        "GAME_ID": [21900010, 21900011, 51900012, 42000013],
        "HOME_TEAM_ID": [1, 2, 1, 3],
        "VISITOR_TEAM_ID": [2, 1, 3, 1],
        "SEASON": [2019, 2019, 2019, 2020],
        "PTS_home": [100.0, 102.0, 126.0, 99.0],
        "PTS_away": [90.0, 104.0, 122.0, 97.0],
    })
    details = pd.DataFrame({
        "GAME_ID": [21900010, 21900011, 51900012, 21900011],
        "TEAM_ID": [1, 1, 1, 2],
        "PLAYER_ID": [11, 11, 11, 22],
        "MIN": ["10:00", "20:00", "15:00", math.nan],
        "PTS": [5.0, 7.0, 3.0, math.nan],
    })
    ranking = make_raw_ranking().iloc[[0, 1, 2, 4]]
    teams = make_raw_teams()

    with tempfile.TemporaryDirectory() as tmp:
        games.to_csv(os.path.join(tmp, "games.csv"), index=False)
        details.to_csv(os.path.join(tmp, "games_details.csv"), index=False)
        ranking.to_csv(os.path.join(tmp, "ranking.csv"), index=False)
        teams.to_csv(os.path.join(tmp, "teams.csv"), index=False)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(tmp)
        saved_games = pd.read_csv(os.path.join(tmp, "games_clean.csv"))
        saved_details = pd.read_csv(
            os.path.join(tmp, "games_details_clean.csv"))
        saved_ranking = pd.read_csv(os.path.join(tmp, "ranking_clean.csv"))
        saved_teams = pd.read_csv(os.path.join(tmp, "teams_clean.csv"))

    printed = output.getvalue()
    assert "Rows before cleaning: 4" in printed
    assert "Rows after cleaning: 3" in printed
    assert "Bubble games: 1" in printed
    assert "games_details rows: 4 -> 2" in printed
    assert "ranking rows: 4 -> 2" in printed
    assert "teams rows: 4 -> 4" in printed

    assert list(saved_games["GAME_ID"]) == [21900010, 21900011, 42000013]
    assert list(saved_games["POINT_DIFF"]) == [10.0, -2.0, 2.0]
    assert list(saved_games["BUBBLE"]) == [False, True, False]
    assert list(saved_games["PERIOD"]) == ["exclude", "bubble",
                                           "covid_2021"]
    assert "Games by PERIOD and GAME_TYPE" in printed

    assert list(saved_details["GAME_ID"]) == [21900010, 21900011]
    # Team 1 is home in game 21900010 and away in game 21900011.
    assert list(saved_details["HOME"]) == [True, False]
    assert list(saved_details["MIN"]) == [10.0, 20.0]

    assert list(saved_ranking["TEAM_ID"]) == [1, 2]
    assert len(saved_teams) == 4


def run_all_tests():
    """
    Runs every test function in this file and prints a summary.
    """
    test_load_games()
    test_load_csv()
    test_clean_games()
    test_add_period()
    test_clean_teams()
    test_split_record()
    test_clean_ranking()
    test_parse_minutes()
    test_clean_details()
    test_main()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()
