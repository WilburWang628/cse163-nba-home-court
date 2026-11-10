# Does the Crowd Win the Game? Measuring NBA Home-Court Advantage with the 2020 Bubble

CSE 163 final project by **Wilbur Wang, Amber Yang, Zhexin Cai**.

We measure how large the NBA home-court advantage is from the 2003-04 season
to 2022, and whether it disappeared during the 2020 "bubble" in Orlando
(July 30 – October 11, 2020), when games were played with no fans at a
neutral site.

## 1. Get the data

The data comes from the Kaggle dataset
[NBA games data (nathanlauga/nba-games)](https://www.kaggle.com/datasets/nathanlauga/nba-games).
It is not included in this repository because the files are large.

1. Open the link above, sign in to Kaggle, and click **Download**. You will
   get a file named `archive.zip`.
2. Unzip it and put the CSV files into the `data/` folder of this project:

   ```bash
   mkdir -p data
   unzip archive.zip -d data/
   ```

   (Safari may unzip the download for you into a folder named `archive`; in
   that case copy the CSV files from that folder into `data/`.)

After this step `data/` should contain:

```
data/
├── games.csv
├── games_details.csv
├── players.csv
├── ranking.csv
└── teams.csv
```

## 2. Set up the Python environment

You need Python 3.10 or newer. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run `source .venv/bin/activate` again every time you open a new terminal.

## 3. Run the scripts

Run every command from the project folder with the virtual environment
activated, **in this order**. Each analysis script cleans `data/games.csv`
itself (by calling `clean.py`), so steps 3–5 do not depend on the
`*_clean.csv` files, but steps 1 and 2 are the quickest way to confirm the
setup works.

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python check_setup.py` | Checks that the data loads and the libraries are installed. | Prints `Setup OK!` |
| 2 | `python clean.py` | Cleans `games.csv`, `games_details.csv`, `ranking.csv` and `teams.csv`, adds `GAME_TYPE`, `POINT_DIFF`, `BUBBLE` and `PERIOD`, and prints row counts and games per `PERIOD` x `GAME_TYPE`. | `data/*_clean.csv` |
| 3 | `python eda_games.py` | Exploratory data analysis of the games: data size, missing values, summaries of the research variables, and home win rates by season and by period. | `results/eda_games.txt`, `figures/fig1`–`fig3` |
| 4 | `python validity_checks.py` | Result Validity checks for RQ1 and RQ2: runs each planned test (binomial test, OLS trend, two-proportion z-test, Welch's t-test, permutation test) and checks its assumptions. | `results/validity_checks.txt`, `figures/fig4`–`fig5` |
| 5 | `python robustness_checks.py` | Follow-up checks for problems found in step 4: game-level logistic regression for the RQ1 trend, series-level permutation tests for playoff games, Holm / Bonferroni adjustment for multiple comparisons, and power analysis for the small bubble groups. | `results/robustness_checks.txt`, `figures/fig6` |

### Tests and style

Each test file uses small hand-made DataFrames whose correct results can be
worked out by hand. Each prints a "passed" message at the end.

| Command | Tests |
|---|---|
| `python test_clean.py` | `clean.py` |
| `python test_eda.py` | calculation functions in `eda_games.py` |
| `python test_validity_checks.py` | calculation functions in `validity_checks.py` |
| `python test_robustness_checks.py` | calculation functions in `robustness_checks.py` |
| `flake8 *.py` | code style of every script (no output means no problems) |

## Cleaned data

`python clean.py` keeps only regular-season (`GAME_TYPE` 2) and playoff
(`GAME_TYPE` 4) games. Every game in `games_clean.csv`, and every player row
in `games_details_clean.csv`, gets a `PERIOD` label:

| `PERIOD` | Games | Why |
|---|---|---|
| `bubble` | Played from 2020-07-30 to 2020-10-11 | Orlando bubble: no fans, neutral site |
| `covid_2021` | `SEASON` 2020 (the 2020-21 season) | Very few or no fans in most arenas |
| `normal` | `SEASON` 2004–2018 and 2021 | Regular crowds; the comparison group |
| `exclude` | `SEASON` 2003, `SEASON` 2022, and non-bubble `SEASON` 2019 games | 2022 is incomplete in the data (it ends in December 2022). 2003 and the non-bubble 2019 games are left out by design; they could also be counted as `normal` |

Other useful columns:

- `games_clean.csv`: `POINT_DIFF` (home points minus away points),
  `BUBBLE` (True for bubble games)
- `games_details_clean.csv`: `MIN` (minutes played as a number), `HOME`
  (True if the player's team was the home team)
- `ranking_clean.csv`: `HOME_W`, `HOME_L`, `ROAD_W`, `ROAD_L`
- `teams_clean.csv`: `TEAM_NAME` (e.g. "Los Angeles Lakers")

## Files

| File | Purpose |
|---|---|
| `check_setup.py` | Environment and data sanity check |
| `clean.py` | Data cleaning functions and the cleaning pipeline |
| `eda_games.py` | Exploratory data analysis of `games.csv` |
| `validity_checks.py` | Hypothesis tests and assumption checks for RQ1 and RQ2 |
| `robustness_checks.py` | Logistic regression, series-level permutation tests, multiple-comparison adjustment and power analysis |
| `test_clean.py`, `test_eda.py`, `test_validity_checks.py`, `test_robustness_checks.py` | Tests for the scripts above |
| `figures/` | Figures produced by steps 3–5 |
| `results/` | Text results produced by steps 3–5 |
| `requirements.txt` | Python packages used by the project |
