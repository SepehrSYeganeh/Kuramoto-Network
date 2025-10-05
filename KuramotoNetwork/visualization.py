from KuramotoNetwork import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(dim: int, N: int, kappa: float, sigma: float, xi: float):
    filename = io.make_filename(dim, N, kappa, sigma, xi)
    df = io.load_data(filename)
    data_df, constants = io.load_data(dim)
    theta = df_theta.iloc[:, 3:].to_numpy()
    x = np.cos(theta)
    y = np.sin(theta)
    r = df_param.iloc[:, 1].to_numpy()
    psi = df_param.iloc[:, 2].to_numpy()
    xr = r * np.cos(psi)
    yr = r * np.sin(psi)
    X = np.column_stack((x, xr))
    Y = np.column_stack((y, yr))
    time = df_theta.iloc[:, 0].to_numpy()
    frames = len(time)
    N = theta.shape[1]

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
    fig.suptitle(f'kappa = {kappa}, xi = {xi}', fontsize=14)

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
        interval=20,  # milliseconds between frames
        blit=True  # faster redraw
    )

    ani.save('fig/trajectory-' + str(dim) + 'd-k' + str(kappa) + '-xi' + str(xi) + '.gif', writer='pillow', fps=5)


def plot_final_r(kappa_arr, r_arr, dim, xi):
    plt.scatter(kappa_arr, r_arr)
    plt.xlabel(r'$\kappa$')
    plt.ylabel('$r$')
    plt.title(f'{dim} dimensional network')
    plt.savefig(f'fig/final-r-xi{xi}-{dim}d.png', dpi=300)
    plt.show()


def plot_critical_kappa(kappa_arr, r_arr, dim, xi):
    plt.scatter(kappa_arr, r_arr)
    plt.xlabel(r'$\kappa$')
    plt.ylabel('$r$')
    plt.title(f'{dim} dimensional network')
    plt.savefig(f'fig/critical-kappa-xi{xi}-{dim}d.png', dpi=300)
    plt.show()


def plot_final_r_noise(xi_arr, r_arr, dim, kappa):
    plt.scatter(xi_arr, r_arr)
    plt.xlabel(r'$\xi$')
    plt.ylabel('$r$')
    plt.title(f'{dim} dimensional network, $\\kappa$ = {kappa}')
    plt.savefig(f'fig/final-r-kappa{kappa}-{dim}d.png', dpi=300)
    plt.show()
