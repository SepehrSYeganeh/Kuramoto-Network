import os
import csv


def init_data(N: int, dim: int):
    FILENAME = "data/data" + str(dim) + "D.csv"
    HEADER = ["t", 'r', 'psi'] + ['theta' + str(i) for i in range(N)]
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    with open(FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def append_data(dim: int, time: float, r: float, psi: float, angles):
    with open("data/data" + str(dim) + "D.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time, r, psi] + list(angles))
