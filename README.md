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
activated.

| Command | What it does |
|---|---|
| `python check_setup.py` | Checks that the data loads and the libraries are installed. Prints `Setup OK!` at the end. |
| `python clean.py` | Cleans `games.csv`, `games_details.csv`, `ranking.csv` and `teams.csv`, prints row counts, and writes `games_clean.csv`, `games_details_clean.csv`, `ranking_clean.csv` and `teams_clean.csv` into `data/`. |
| `python test_clean.py` | Runs the tests for `clean.py` on small hand-made DataFrames. Prints `All tests passed!` at the end. |
| `flake8 clean.py test_clean.py check_setup.py` | Checks code style. No output means no problems. |

## Files

- `check_setup.py` – environment and data sanity check
- `clean.py` – data cleaning functions and the cleaning pipeline
- `test_clean.py` – tests for `clean.py`
- `requirements.txt` – Python packages used by the project
