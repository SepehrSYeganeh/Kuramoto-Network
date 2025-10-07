from KuramotoNetwork import *
import numpy as np


def main():
    dim = 2
    L = 10
    N = 100
    sigma = 2
    kappa_arr = np.linspace(0, 15, 16)
    xi_arr = np.array([0])
    # xi_arr = np.linspace(0, 1, 2)
    steps = 10_000
    dt = 0.001
    snapshot_frames = 10
    init_theta = initialize_theta_1D(N)
    init_omega = initialize_Normal_omega_1D(N, sigma)
    steady_state = int(steps / 10)

    # generate_data(dim, N, sigma, kappa_arr, xi_arr, steps, dt, snapshot_frames, init_theta, init_omega)
    # animate_simulations(dim, N, sigma, kappa_arr, xi_arr, steps)
    r_in_time(dim, N, sigma, kappa_arr, xi_arr, steps)
    # r_infty_kappa()


if __name__ == '__main__':
    main()
