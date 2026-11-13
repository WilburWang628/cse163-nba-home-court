"""
CSE 163 Final Project: NBA home-court advantage.

Tests for the functions in eda_details.py.
Plotting functions and functions that only format text are not tested.
"""

import pandas as pd

from eda_details import (aggregate_team_stats, calculate_differences)

TEST_DATA = pd.DataFrame({
    'GAME_ID': [1, 1, 1, 1, 2, 2],
    'HOME': [True, True, False, False, True, False],
    'PERIOD': ['normal', 'normal', 'normal', 'normal',
               'covid_2021', 'covid_2021'],
    'FTA': [5, 3, 6, 4, 7, 5],
    'PF': [2, 1, 3, 2, 4, 3],
    'FGA': [10, 8, 12, 9, 15, 11]
})


def test_aggregate():
    '''
    Tests the aggregate_team_stats function.
    '''

    result = aggregate_team_stats(TEST_DATA)

    assert len(result) == 4
    assert list(result.columns) == [
        'GAME_ID', 'HOME', 'PERIOD', 'FTA', 'PF', 'FGA'
    ]

    home = result[(result['GAME_ID'] == 1) & result['HOME']]
    assert home['FTA'].loc[home.index[0]] == 8
    assert home['PF'].loc[home.index[0]] == 3
    assert home['FGA'].loc[home.index[0]] == 18


def test_diff():
    '''
    Tests the calculate_differences function.
    '''

    team_stats = aggregate_team_stats(TEST_DATA)
    result = calculate_differences(team_stats)

    assert len(result) == 2

    game1 = result[result['GAME_ID'] == 1]
    assert game1['FTA_DIFF'].loc[game1.index[0]] == -2
    assert game1['PF_DIFF'].loc[game1.index[0]] == -2
    assert game1['FGA_DIFF'].loc[game1.index[0]] == -3


def main():
    test_aggregate()
    test_diff()
    print("All tests passed.")


if __name__ == '__main__':
    main()
