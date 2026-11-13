'''
CSE 163 Final Project: NBA home-court advantage.
This file takes in a data frame of NBA game details, explores the statistics
of player performance based on free throw percentage(FTA), number of points
scored per player(PF), field goal percentage(FGA), and field goals made(MIN).
And generates two figures about free throw difference in home-court and
non-home-court, and free throw distribution by periods.
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

DATA_DIR = "data"


def explore_data(data: pd.DataFrame, file) -> None:
    '''
    This function takes in a data frame and an output file,
    and prints dataset dimensions, missing values,
    and statistical summaries of FTA, PF, FGA, and MIN.
    '''

    print('Dataset Dimensions', file=file)
    print('Rows: ', data.shape[0], file=file)
    print('Columns: ', data.shape[1], file=file)
    print('Each row represents an NBA game.', file=file)

    print('\nMissing Values', file=file)
    print(data.isna().sum(), file=file)
    print('NICKNAME, START_POSITION, and COMMENT presents a significant' \
          'amount of missing data, whileh all other columns have no' \
          'missing data.', file=file)

    print('\nMissing Value Percentages', file=file)
    print(data.isna().mean() * 100, file=file)
    print('NICKNAME misses more than 90 percent of content, START_POSITION' \
          'misses more than half content, and COMMENT misses all content,' \
          'while other columns have no missing data.', file=file)

    columns = ['FTA', 'PF', 'FGA', 'MIN']
    print('\nStatistical Summary', file=file)
    print(data[columns].describe(), file=file)


def aggregate_team_stats(data: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes a data frame and
    aggregates players' field goal percentage (FGA),
    free throw percentage (FTA), and number of points scored (PF)
    to get their respective teams' data.
    It then returns a data frame of the team's data.
    '''

    columns = ['FTA', 'PF', 'FGA']
    data = data[data['PERIOD'] != 'exclude'].copy()
    data[columns] = data[columns].fillna(0)
    team_stats = (data.groupby(['GAME_ID', 'HOME', 'PERIOD'])[columns].sum()
                  .reset_index())
    return team_stats


def calculate_differences(team_stats: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in a data frame of the aggregate team data,
    and calculates the home-away difference based on FGA, FTA, and PF.
    It then returns a data frame that has the home-away differences
    of the three columns.
    '''

    home = team_stats[team_stats['HOME']].copy()
    away = team_stats[~team_stats['HOME']].copy()

    home = home.drop(columns=['HOME'])
    away = away.drop(columns=['HOME'])

    home = home.rename(columns={'FTA': 'HOME_FTA',
                                'FGA': 'HOME_FGA',
                                'PF': 'HOME_PF'})
    away = away.rename(columns={'FTA': 'AWAY_FTA',
                                'FGA': 'AWAY_FGA',
                                'PF': 'AWAY_PF'})

    games = home.merge(away, on=['GAME_ID', 'PERIOD'])
    games['FTA_DIFF'] = games['HOME_FTA'] - games['AWAY_FTA']
    games['FGA_DIFF'] = games['HOME_FGA'] - games['AWAY_FGA']
    games['PF_DIFF'] = games['HOME_PF'] - games['AWAY_PF']

    return games


def plot_fta_periods(games: pd.DataFrame) -> None:
    '''
    This function takes in the data frame that has the home-away difference
    and plots the FTA difference across periods.
    '''

    plt.figure(figsize=(8, 4))
    sns.barplot(data=games, x='PERIOD', y='FTA_DIFF',
                order=['normal', 'covid_2021', 'bubble'])
    plt.title('Free Throw Differences by Period')
    plt.xlabel('Period')
    plt.ylabel('Home FTA - AWAY FTA')
    plt.savefig('figures/fig7_Home_Away_FTA_Diff.png')
    plt.close()


def plot_fta_distribution(games: pd.DataFrame) -> None:
    '''
    This function takes in the data frame that has the home-away difference
    and plots the FTA distribution by periods.
    '''

    periods = ['normal', 'covid_2021', 'bubble']
    fig, axs = plt.subplots(nrows=3, figsize=(8, 15))
    fig.suptitle('Distribution of Home-Away Free Throw Difference by Period',
                 fontsize=16, y=0.98)

    for i in range(len(periods)):
        subset = games[games['PERIOD'] == periods[i]]
        sns.histplot(x='FTA_DIFF', data=subset, ax=axs[i])
        axs[i].set_title(periods[i], pad=10)
        axs[i].set_xlabel('Difference in Free Throws Number (Home - Away)')
        axs[i].set_ylabel('Number of Games')

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3)
    fig.savefig('figures/fig8_Home_Away_FTA_Dist.png')
    plt.close(fig)


def plot_fta_histogram(games: pd.DataFrame) -> None:
    '''
    This funcion takes in a data frame of the game details containing
    home-away differences, and plots the distribution of FTA.
    '''

    plt.figure(figsize=(8, 4.5))
    sns.histplot(data=games, x='FTA_DIFF', bins=30)
    plt.title('Distribution of Free Throw Difference')
    plt.xlabel('Home FTA - Away FTA')
    plt.ylabel('Frequency')

    plt.savefig('figures/fig9_FTA_Histogram.png')
    plt.close()


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    game_data = pd.read_csv("data/games_details_clean.csv", low_memory=False)
    team_stats = aggregate_team_stats(game_data)
    games = calculate_differences(team_stats)

    with open('results/eda_details.txt', 'w') as file:
        explore_data(game_data, file)

        print('\nTeam Statistics', file=file)
        print(team_stats.head(), file=file)

        print('\nGame Differences', file=file)
        print(games.head(), file=file)

        print('\nAverage Differences by Period', file=file)
        print(games.groupby('PERIOD')[['FTA_DIFF', 'PF_DIFF']]
              .agg(['mean', 'median', 'std', 'count']), file=file)

        print('\nNormality Check', file=file)
        for period in ['normal', 'covid_2021', 'bubble']:
            subset = games[games['PERIOD'] == period]
            print(period, 'FTA Histogram Sample Size: ', len(subset),
                  file=file)

        print('\nPaired T-test', file=file)
        t_stat, p_value = stats.ttest_1samp(games['FTA_DIFF'], 0)

        print('T-statistic:', t_stat, file=file)
        print('P-value:', p_value, file=file)

    plot_fta_periods(games)
    plot_fta_distribution(games)
    plot_fta_histogram(games)


if __name__ == '__main__':
    main()
