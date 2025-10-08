from KuramotoNetwork import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def animate(args) -> None:
    anim_path, data_path, dim, N, kappa, sigma, xi, steps = args
    filename = io.make_filename(dim, N, kappa, sigma, xi, steps)
    df = io.load_data(data_path, filename)

    theta = df.iloc[:, 3:].to_numpy()
    r = df.iloc[:, 1].to_numpy()
    psi = df.iloc[:, 2].to_numpy()
    time = df.iloc[:, 0].to_numpy()
    frames = len(time)

    x = np.cos(theta)
    y = np.sin(theta)
    xr = r * np.cos(psi)
    yr = r * np.sin(psi)
    X = np.column_stack((x, xr))
    Y = np.column_stack((y, yr))

    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), 1.0, color='lightgray', fill=False)
    line = plt.plot([0])
    ax.add_artist(circle)
    line, = ax.plot([], [], color='red', lw=1.5)
    title = ax.set_title(f'time = {time[0]}')
    fig.suptitle(f'dim={dim}, kappa={kappa:.2f}, xi={xi:.2f}', fontsize=14)

    colors = plt.cm.viridis(np.linspace(0, 1, N))
    colors = np.vstack((colors, np.array([0, 0, 0, 1])))
    scat = ax.scatter(X[0], Y[0], c=colors)

    def update(frame):
        line.set_data([0, xr[frame]], [0, yr[frame]])
        coords = np.column_stack((X[frame], Y[frame]))
        scat.set_offsets(coords)
        title.set_text(f'time = {time[frame]:.2f}')
        return line, scat, title

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=20,
        blit=True
    )

    ani.save(os.path.join(anim_path, f"{filename}.gif"),
             writer='pillow', fps=15)

    print(f"animation done for kappa={kappa} and xi={xi}")


def plot_r_time(args) -> None:
    rt_path, data_path, dim, N, kappa, sigma, xi, steps = args
    filename = io.make_filename(dim, N, kappa, sigma, xi, steps)
    df = io.load_data(data_path, filename)

    plt.plot(df['t'], df['r'])
    plt.xlabel('time')
    plt.ylabel('r')
    plt.title(f"dim={dim}, kappa={kappa:.2f}, xi={xi:.2f}")
    plt.savefig(os.path.join(rt_path, f"{filename}.png"), dpi=300)
    plt.close()

    print(f"plot done for kappa={kappa} and xi={xi}")


def plot_r_infty_kappa(rk_path: os.PathLike,
                       k_arr: np.ndarray, xi_arr: np.ndarray, r_arr: np.ndarray) -> None:
    # TODO: only this
    plt.plot(k_arr, r_arr)
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$r_\infty$")
    plt.title("Stable State Order Parameter versus Coupling Strength")
    plt.savefig(os.path.join(rk_path, "r-k.png"), dpi=300)
    plt.close()


def plot_critical_kappa(kappa_arr, r_arr, dim, xi):
    plt.scatter(kappa_arr, r_arr)
    plt.xlabel(r'$\kappa$')
    plt.ylabel('$r$')
    plt.title(f'{dim} dimensional network')
    plt.savefig(f'fig/critical-kappa-xi{xi}-{dim}d.png', dpi=300)
    plt.show()
