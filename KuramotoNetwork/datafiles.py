import os
import csv


def init_theta(N: int, dim: int):
    FILENAME = "data/theta-" + str(dim) + "d.csv"
    HEADER = ["t"] + ['theta' + str(i) for i in range(N)]
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    with open(FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def init_param(dim: int):
    FILENAME = "data/param-" + str(dim) + "d.csv"
    HEADER = ["t", 'r', 'psi']
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    with open(FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def append_theta(timestamp, angles, dim):
    with open("data/theta-" + str(dim) + "d.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp] + list(angles))


def append_param(timestamp, r, psi, dim):
    with open("data/param-" + str(dim) + "d.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, r, psi])
