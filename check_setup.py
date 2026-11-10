"""
Quick sanity check for the CSE 163 NBA final project environment.
Loads the main datasets and prints library versions.
"""

from pathlib import Path

import pandas as pd
import scipy
import statsmodels

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    for name in ["games.csv", "games_details.csv"]:
        df = pd.read_csv(DATA_DIR / name, low_memory=False)
        rows, cols = df.shape
        print(f"{name}: {rows} rows, {cols} columns")

    print(f"scipy version: {scipy.__version__}")
    print(f"statsmodels version: {statsmodels.__version__}")
    print("Setup OK!")


if __name__ == "__main__":
    main()
