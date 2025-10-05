import os
import csv
import pandas as pd


def make_filename(dim: int, N: int, kappa: float, sigma: float, xi: float):
    return f"D{dim}-N{N}-K{kappa:.3f}-std{sigma:.3f}-xi{xi:.3f}"


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


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(f"data/{filename}.csv")
