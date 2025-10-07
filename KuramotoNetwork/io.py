import os
import csv
import pandas as pd

SIM_PATH = "simulations"


def init_simulation_directory(name: str) -> tuple[os.PathLike, os.PathLike]:
    path = os.path.join(SIM_PATH, name)
    os.makedirs(path, exist_ok=True)
    data_path = os.path.join(path, "data")
    os.makedirs(data_path, exist_ok=True)
    fig_path = os.path.join(path, "fig")
    os.makedirs(fig_path, exist_ok=True)
    return data_path, fig_path


def make_filename(dim: int, N: int, kappa: float, sigma: float, xi: float, T: int) -> str:
    return f"D{dim}-N{N}-K{kappa:.3f}-std{sigma:.3f}-xi{xi:.3f}-T{T}"


def init_data_file(data_path: os.PathLike, filename: str, N: int) -> os.PathLike:
    PATH = os.path.join(data_path, f"{filename}.csv")
    HEADER = ["t", 'r', 'psi'] + [f"theta{i}" for i in range(N)]
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
    return PATH


def append_data(path: os.PathLike, time: float, r: float, psi: float, angles) -> None:
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time, r, psi] + list(angles))


def load_data(data_path: os.PathLike, filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, f"{filename}.csv"))


def init_anim_directory(fig_path: os.PathLike) -> os.PathLike:
    PATH = os.path.join(fig_path, "animations")
    os.makedirs(PATH, exist_ok=True)
    return PATH
