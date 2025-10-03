import os
import csv
import pandas as pd


def init_data(N: int, dim: int, constants: dict) -> None:
    FILENAME = f"data/data{dim}D.csv"
    HEADER = ["t", 'r', 'psi'] + [f"theta{i}" for i in range(N)]
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)

    with open(FILENAME, mode="w", newline="") as f:
        for key, val in constants.items():
            f.write(f"# {key}={val}\n")

        writer = csv.writer(f)
        writer.writerow(HEADER)


def append_data(dim: int, time: float, r: float, psi: float, angles) -> None:
    with open("data/data" + str(dim) + "D.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time, r, psi] + list(angles))


def load_data(dim: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(f"data/data{dim}D.csv", comment="#")

    constants = {}
    with open(f"data/data{dim}D.csv") as f:
        for line in f:
            if line.startswith("#"):
                key, val = line[1:].strip().split("=")
                constants[key] = float(val)
            else:
                break

    return df, constants
