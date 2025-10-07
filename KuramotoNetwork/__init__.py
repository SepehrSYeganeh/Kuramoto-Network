from KuramotoNetwork.simulation import *
from KuramotoNetwork import visualization
import numpy as np
from itertools import product
import multiprocessing as mp


def _simulation1D(args) -> None:
    """run a single simulation for 1-dimensional grid"""
    path, N, sigma, kappa, xi, steps, dt, snapshot_frames, init_theta, init_omega = args
    kuramoto1d = KuramotoGrid1D(path,
                                N, sigma, kappa, xi,
                                steps, dt, snapshot_frames,
                                init_theta, init_omega)
    kuramoto1d.run()
    print(f"simulation done for kappa={kappa} and xi={xi}")


def _simulation2D(args) -> None:
    """run a single simulation for 2-dimensional grid"""
    path, N, sigma, kappa, xi, steps, dt, snapshot_frames, init_theta, init_omega = args
    kuramoto2d = KuramotoGrid2D(path,
                                N, sigma, kappa, xi,
                                steps, dt, snapshot_frames,
                                init_theta, init_omega)
    kuramoto2d.run()
    print(f"simulation done for kappa={kappa}")


def generate_data(data_path: str,
                  dim: int,
                  N: int,  # L for 2d dimension
                  sigma: float,
                  kappa_arr: np.ndarray,
                  xi_arr: np.ndarray,
                  steps: int,
                  dt: float,
                  snapshot_frames: int,
                  init_theta: np.ndarray,
                  init_omega: np.ndarray
                  ) -> None:
    """
    :param data_path: simulation path
    :param dim: dimension
    :param N: number of oscillators
    :param sigma: std of omega distribution
    :param kappa_arr: coupling strength
    :param xi_arr: noise strength
    :param steps: number of simulation steps
    :param dt: time increment
    :param snapshot_frames: frames between snapshots
    generates data for given parameters
    """
    args_list = [
        (data_path,
         N, sigma, kappa, xi,
         steps, dt, snapshot_frames,
         init_theta, init_omega)
        for kappa, xi in product(kappa_arr, xi_arr)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        if dim == 1:
            pool.map(_simulation1D, args_list)
        elif dim == 2:
            pool.map(_simulation2D, args_list)

    print("All simulations finished")


def animate_simulations(fig_path: str,
                        data_path: str,
                        dim: int,
                        N: int,
                        sigma: float,
                        kappa_arr: np.ndarray,
                        xi_arr: np.ndarray,
                        steps: int
                        ) -> None:
    anim_path = io.init_anim_directory(fig_path)

    args_list = [
        (anim_path, data_path, dim, N, kappa, sigma, xi, steps)
        for kappa, xi in product(kappa_arr, xi_arr)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(visualization.animate, args_list)

    print("All animations finished")


def r_in_time(dim: int,
              N: int,
              sigma: float,
              kappa_arr: np.ndarray,
              xi_arr: np.ndarray,
              steps: int
              ) -> None:
    args_list = [
        (dim, N, kappa, sigma, xi, steps)
        for kappa, xi in product(kappa_arr, xi_arr)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(visualization.plot_r_time, args_list)

    print("All plots finished")


def r_infty_kappa(dim: int,
                  N: int,
                  sigma: float,
                  kappa_arr: np.ndarray,
                  xi_arr: np.ndarray,
                  steady_state: int  # number of steady state steps
                  ) -> None:
    # TODO: first make r = f(kappa,xi) then plot
    args_list = [
        (dim, N, kappa, sigma, xi, steady_state)
        for kappa, xi in product(kappa_arr, xi_arr)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(visualization.plot_r_infty_kappa, args_list)

    print("All plots finished")


def critical_kappa_1d():
    N = 100
    sigma = 2
    kappa_arr = np.arange(0, 4.6, 0.1)
    xi = 0
    transient_steps = 9000
    steady_steps = 1000
    dt = 0.001
    n_s = 0
    dim = 1
    ensemble = 100
    final_r_arr = np.zeros(len(kappa_arr))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, kappa in enumerate(kappa_arr):
            args_list = [
                (N, sigma, kappa, xi, transient_steps, dt, n_s, steady_steps)
                for _ in range(ensemble)
            ]
            r_arr = pool.map(_simulate_single_run_1d, args_list)
            r_arr = np.array(r_arr)
            final_r_arr[i] = np.mean(r_arr)

    print('simulation done')
    plot_critical_kappa(kappa_arr, final_r_arr, dim, xi)
    print('visualization done')


def critical_kappa_2d():
    N = 100
    sigma = 2
    kappa_arr = np.arange(0, 4.6, 0.1)
    xi = 0
    transient_steps = 9000
    steady_steps = 1000
    dt = 0.001
    n_s = 0
    dim = 2
    ensemble = 100
    final_r_arr = np.zeros(len(kappa_arr))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, kappa in enumerate(kappa_arr):
            args_list = [
                (N, sigma, kappa, xi, transient_steps, dt, n_s, steady_steps)
                for _ in range(ensemble)
            ]
            r_arr = pool.map(_simulate_single_run_2d, args_list)
            r_arr = np.array(r_arr)
            final_r_arr[i] = np.mean(r_arr)

    print('simulation done')
    plot_critical_kappa(kappa_arr, final_r_arr, dim, xi)
    print('visualization done')


def final_r_noise_1d():
    N = 100
    sigma = 2
    kappa = 60
    xi_arr = np.arange(26)
    transient_steps = 9000
    steady_steps = 1000
    dt = 0.001
    n_s = 0
    dim = 1
    ensemble = 100
    final_r_arr = np.zeros(len(xi_arr))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, xi in enumerate(xi_arr):
            args_list = [
                (N, sigma, kappa, xi, transient_steps, dt, n_s, steady_steps)
                for _ in range(ensemble)
            ]
            r_arr = pool.map(_simulate_single_run_1d, args_list)
            r_arr = np.array(r_arr)
            final_r_arr[i] = np.mean(r_arr)

    print('simulation done')
    plot_final_r_noise(xi_arr, final_r_arr, dim, kappa)
    print('visualization done')


def final_r_noise_2d():
    N = 100
    sigma = 2
    kappa = 30
    xi_arr = np.arange(16)
    transient_steps = 9000
    steady_steps = 1000
    dt = 0.001
    n_s = 0
    dim = 2
    ensemble = 100
    final_r_arr = np.zeros(len(xi_arr))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i, xi in enumerate(xi_arr):
            args_list = [
                (N, sigma, kappa, xi, transient_steps, dt, n_s, steady_steps)
                for _ in range(ensemble)
            ]
            r_arr = pool.map(_simulate_single_run_2d, args_list)
            r_arr = np.array(r_arr)
            final_r_arr[i] = np.mean(r_arr)

    print('simulation done')
    plot_final_r_noise(xi_arr, final_r_arr, dim, kappa)
    print('visualization done')
