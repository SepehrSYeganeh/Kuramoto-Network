from KuramotoNetwork import *
from KuramotoNetwork import io
import numpy as np


def main():
    name = "test0"
    path = io.init_simulation_directory(name)
    dim = 2
    L = 10
    N = 100
    sigma = 2
    kappa_arr = np.linspace(0, 15, 16)
    xi_arr = np.array([0])
    steps = 3_000
    dt = 0.001
    snapshot_frames = 10
    init_theta = initialize_theta_2D(L)
    init_omega = initialize_Normal_omega_2D(L, sigma)
    steady_state = int(steps / 10)

    generate_data(path, dim,
                  L, sigma, kappa_arr, xi_arr,
                  steps, dt, snapshot_frames,
                  init_theta, init_omega)
    # animate_simulations(dim, N, sigma, kappa_arr, xi_arr, steps)
    # r_in_time(dim, N, sigma, kappa_arr, xi_arr, steps)
    # r_infty_kappa()


if __name__ == '__main__':
    main()
