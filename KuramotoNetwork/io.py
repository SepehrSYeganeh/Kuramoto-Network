import os
import csv
import pandas as pd


def init_data(filename: str, N: int) -> None:
    FILENAME = f"data/{filename}.csv"
    HEADER = ["t", 'r', 'psi'] + [f"theta{i}" for i in range(N)]
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    with open(FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def append_data(filename: str, time: float, r: float, psi: float, angles) -> None:
    with open(f"data/{filename}.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time, r, psi] + list(angles))


def load_data(filename: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(f"data/{filename}.csv")
    constants = {}
    # TODO: complete this
    return df, constants
