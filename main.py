from KuramotoNetwork import *
from KuramotoNetwork import io
import numpy as np


def main():
    name = "r-k-2D"
    data_path, fig_path = io.init_simulation_directory(name)
    dim = 2
    L = 10
    N = 100
    sigma = 2

    # 1D init
    # init_theta = initialize_theta_1D(N)
    # init_omega = initialize_Normal_omega_1D(N, sigma)

    # 2D init
    init_theta = initialize_theta_2D(L)
    init_omega = initialize_Normal_omega_2D(L, sigma)

    kappa_arr = np.linspace(0, 15, 31)
    xi_arr = np.array([0])
    steps = 40_000
    dt = 0.001
    snapshot_frames = 10

    # 1D
    # generate_data(data_path, dim,
    #               N, sigma, kappa_arr, xi_arr,
    #               steps, dt, snapshot_frames,
    #               init_theta, init_omega)

    # 2D
    # generate_data(data_path, dim,
    #               L, sigma, kappa_arr, xi_arr,
    #               steps, dt, snapshot_frames,
    #               init_theta, init_omega)

    # 1D
    # generate_relaxed_data(data_path, dim,
    #                       N, sigma, kappa_arr, xi_arr,
    #                       steps, dt, snapshot_frames,
    #                       init_theta, init_omega)

    # 2D
    # generate_relaxed_data(data_path, dim,
    #                       L, sigma, kappa_arr, xi_arr,
    #                       steps, dt, snapshot_frames,
    #                       init_theta, init_omega)

    # animate_simulations(fig_path, data_path, dim, N, sigma, kappa_arr, xi_arr, steps)

    # r_in_time(fig_path, data_path, dim, N, sigma, kappa_arr, xi_arr, steps)

    r_infty_kappa(fig_path, data_path, dim, N, sigma, kappa_arr, xi_arr, steps)


if __name__ == '__main__':
    main()
