from KuramotoNetwork.simulation import *
from KuramotoNetwork.visualization import *
import multiprocessing as mp


def trajectory_1d():
    N = 100
    sigma = 2
    kappa_list = [0, 0.1, 1, 5, 20, 100, 500]
    xi = 0
    steps = 500
    dt = 0.001
    n_s = 10
    dim = 1
    for kappa in kappa_list:
        kuramoto1d = KuramotoGrid1d(N, sigma, kappa, xi, steps, dt, n_s)
        kuramoto1d.run()
        print('simulation done')
        # animate(xi, kappa, dim)
        # print('visualize done')


def trajectory_with_noise_1d():
    N = 100
    sigma = 2
    kappa_list = [0, 0.1, 1, 5, 20, 100, 500]
    xi = 0.1
    steps = 5000
    dt = 0.001
    n_s = 10
    dim = 1
    for kappa in kappa_list:
        kuramoto1d = KuramotoGrid1d(N, sigma, kappa, xi, steps, dt, n_s)
        kuramoto1d.run()
        print('simulation done')
        animate(xi, kappa, dim)
        print('visualize done')


def trajectory_2d():
    N = 100
    sigma = 2
    kappa_list = [0, 0.1, 1, 5, 20, 100, 500]
    xi = 0.1
    steps = 5000
    dt = 0.001
    n_s = 10
    dim = 2
    for kappa in kappa_list:
        kuramoto2d = KuramotoGrid2d(N, sigma, kappa, xi, steps, dt, n_s)
        kuramoto2d.run()
        print('simulation done')
        animate(xi, kappa, dim)
        print('visualize done')


def trajectory_with_noise_2d():
    N = 100
    sigma = 2
    kappa_list = [0, 0.1, 1, 5, 20, 100, 500]
    xi = 0.1
    steps = 5000
    dt = 0.001
    n_s = 10
    dim = 2
    for kappa in kappa_list:
        kuramoto2d = KuramotoGrid2d(N, sigma, kappa, xi, steps, dt, n_s)
        kuramoto2d.run()
        print('simulation done')
        animate(xi, kappa, dim)
        print('visualize done')


def _simulate_single_run_1d(args):
    N, sigma, kappa, xi, transient_steps, dt, n_s, steady_steps = args
    kuramoto1d = KuramotoGrid1d(N, sigma, kappa, xi, transient_steps, dt, n_s)
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
    kuramoto2d = KuramotoGrid2d(N, sigma, kappa, xi, transient_steps, dt, n_s)
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
