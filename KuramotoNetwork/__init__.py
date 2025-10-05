from KuramotoNetwork.simulation import *
from KuramotoNetwork.visualization import *
import numpy as np
from itertools import product
import multiprocessing as mp


def _simulation1D(args) -> None:
    """run a single simulation for 1-dimensional grid"""
    N, sigma, kappa, xi, steps, dt, snapshot_frames = args
    kuramoto1d = KuramotoGrid1D(N, sigma, kappa, xi, steps, dt, snapshot_frames)
    kuramoto1d.run()
    print(f"simulation done for kappa={kappa} and xi={xi}")


def _simulation2D(args) -> None:
    """run a single simulation for 2-dimensional grid"""
    N, sigma, kappa, xi, steps, dt, snapshot_frames = args
    kuramoto2d = KuramotoGrid2D(N, sigma, kappa, xi, steps, dt, snapshot_frames)
    kuramoto2d.run()
    print(f"simulation done for kappa={kappa}")


def generate_data(dim: int,
                  N: int,
                  sigma: float,
                  kappa_arr: np.ndarray,
                  xi_arr: np.ndarray,
                  steps: int,
                  dt: float,
                  snapshot_frames: int
                  ) -> None:
    """
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
        (N, sigma, kappa, xi, steps, dt, snapshot_frames)
        for kappa, xi in product(kappa_arr, xi_arr)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        if dim == 1:
            pool.map(_simulation1D, args_list)
        elif dim == 2:
            pool.map(_simulation2D, args_list)

    print("All simulations finished")


def animate_simulations():
    pass


def _simulate_single_run_1d(args):
    N, sigma, kappa, xi, transient_steps, dt, n_s, steady_steps = args
    kuramoto1d = KuramotoGrid1D(N, sigma, kappa, xi, transient_steps, dt, n_s)
    return kuramoto1d.final_r(transient_steps, steady_steps)


def final_r_no_noise_1d():
    N = 100
    sigma = 2
    kappa_arr = np.arange(201)
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
    plot_final_r(kappa_arr, final_r_arr, dim, xi)
    print('visualization done')


def _simulate_single_run_2d(args):
    N, sigma, kappa, xi, transient_steps, dt, n_s, steady_steps = args
    kuramoto2d = KuramotoGrid2D(N, sigma, kappa, xi, transient_steps, dt, n_s)
    return kuramoto2d.final_r(transient_steps, steady_steps)


def final_r_no_noise_2d():
    N = 100
    sigma = 2
    kappa_arr = np.arange(0, 401, 5)
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
    plot_final_r(kappa_arr, final_r_arr, dim, xi)
    print('visualization done')


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
