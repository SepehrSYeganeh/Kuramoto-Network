from KuramotoNetwork import *


def main():
    dim = 1
    N = 100
    sigma = 2
    kappa_arr = np.linspace(0, 7, 8)
    xi_arr = np.linspace(0, 1, 8)
    steps = 500
    dt = 0.001
    snapshot_frames = 10

    generate_data(dim, N, sigma, kappa_arr, xi_arr, steps, dt, snapshot_frames)
    # final_r_no_noise_1d()
    # final_r_no_noise_2d()
    # critical_kappa_1d()
    # critical_kappa_2d()
    # final_r_noise_1d()
    # final_r_noise_2d()
    pass


if __name__ == '__main__':
    main()
