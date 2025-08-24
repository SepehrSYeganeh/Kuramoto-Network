from KuramotoNetwork import datafiles
import numpy as np


class KuramotoGrid1d:
    dim = 1

    def __init__(self, N: int, sigma: float, kappa: float, xi: float, steps: int, dt: float, n_s: int):
        # initialize time
        self.steps = steps
        self.dt = dt
        self.time = 0
        self.n_s = n_s  # snapshot steps

        # initialize oscillators
        self.N = N  # number of oscillators
        self.sigma = sigma  # std of omega dist
        self.kappa = kappa  # coupling constant
        self.xi = xi  # amplitude of noise
        self.theta_arr = np.linspace(0, 2 * np.pi, self.N, endpoint=False)  # phase
        np.random.shuffle(self.theta_arr)
        self.omega_arr = np.random.normal(size=self.N, scale=self.sigma)  # natural frequencies

        # initialize data files
        datafiles.init_theta(self.N, self.dim)
        datafiles.init_param(self.dim)
        self.update_files()

    #################################################
    # order parameter
    #################################################
    def order_parameter(self):
        x = np.mean(np.cos(self.theta_arr))
        y = np.mean(np.sin(self.theta_arr))
        r = np.sqrt(x ** 2 + y ** 2)
        psi = np.arctan2(y, x)
        return r, psi

    #################################################
    # update
    #################################################
    def update_phases(self):
        lag_left = np.roll(self.theta_arr, 1) - self.theta_arr
        lag_right = np.roll(self.theta_arr, -1) - self.theta_arr
        coupling_arr = self.kappa * (np.sin(lag_left) + np.sin(lag_right)) * self.dt

        noise_arr = self.xi * np.random.normal(size=self.N, scale=np.sqrt(self.dt))

        self.theta_arr += self.omega_arr * self.dt + coupling_arr + noise_arr

    def update_files(self):
        datafiles.append_theta(self.time, self.theta_arr, self.dim)
        r, psi = self.order_parameter()
        datafiles.append_param(self.time, r, psi, self.dim)

    #################################################
    # simulation
    #################################################
    def run(self):
        for t in range(self.steps):
            self.time += self.dt
            self.update_phases()
            if t % self.n_s == 0:
                self.update_files()

    #################################################
    # final_r
    #################################################
    def final_r(self, transient_steps, steady_state):
        # running until system relaxes
        for t in range(transient_steps):
            self.time += self.dt
            self.update_phases()

        # system is relaxed
        r_arr = np.zeros(steady_state)
        for t in range(steady_state):
            self.time += self.dt
            self.update_phases()
            r_arr[t] = self.order_parameter()[0]

        return np.mean(r_arr)


###################################################################################
###################################################################################
class KuramotoGrid2d:
    dim = 2

    def __init__(self, N: int, sigma: float, kappa: float, xi: float,
                 steps: int, dt: float, n_s: int):
        # initialize time
        self.steps = steps
        self.dt = dt
        self.time = 0
        self.n_s = n_s  # snapshot steps

        # initialize parameters
        self.N = N  # number of oscillators
        self.sigma = sigma  # std of omega dist
        self.kappa = kappa  # coupling constant
        self.xi = xi  # amplitude of noise

        # initialize oscillators
        self.theta_arr = np.linspace(0, 2 * np.pi, self.N, endpoint=False)  # phase
        np.random.shuffle(self.theta_arr)
        self.theta_arr = np.reshape(self.theta_arr, (np.sqrt(self.N).astype(int), np.sqrt(self.N).astype(int)))
        self.omega_arr = np.random.normal(size=self.N, scale=self.sigma)  # natural frequencies
        np.random.shuffle(self.omega_arr)
        self.omega_arr = np.reshape(self.omega_arr, (np.sqrt(self.N).astype(int), np.sqrt(self.N).astype(int)))

        # initialize data files
        datafiles.init_theta(self.N, self.dim)
        datafiles.init_param(self.dim)
        self.update_files()

    #################################################
    # order parameter
    #################################################
    def order_parameter(self):
        x = np.mean(np.cos(self.theta_arr))
        y = np.mean(np.sin(self.theta_arr))
        r = np.sqrt(x ** 2 + y ** 2)
        psi = np.arctan2(y, x)
        return r, psi

    #################################################
    # update
    #################################################
    def update_phases(self):
        lag_left = np.roll(self.theta_arr, 1, axis=1) - self.theta_arr
        lag_right = np.roll(self.theta_arr, -1, axis=1) - self.theta_arr
        lag_down = np.roll(self.theta_arr, 1, axis=0) - self.theta_arr
        lag_up = np.roll(self.theta_arr, -1, axis=0) - self.theta_arr
        coupling_arr = self.kappa * (np.sin(lag_left) + np.sin(lag_right) + np.sin(lag_up) + np.sin(lag_down)) * self.dt

        noise_arr = self.xi * np.random.normal(size=self.N, scale=np.sqrt(self.dt)).reshape(
            np.sqrt(self.N).astype(int), np.sqrt(self.N).astype(int))

        self.theta_arr += self.omega_arr * self.dt + coupling_arr + noise_arr

    def update_files(self):
        datafiles.append_theta(self.time, self.theta_arr.flatten(), self.dim)
        r, psi = self.order_parameter()
        datafiles.append_param(self.time, r, psi, self.dim)

    #################################################
    # run simulation
    #################################################
    def run(self):
        for t in range(1, self.steps + 1):
            self.time += self.dt
            self.update_phases()
            if t % self.n_s == 0:
                self.update_files()

    #################################################
    # final_r
    #################################################
    def final_r(self, transient_steps, steady_state):
        # running until system relaxes
        for t in range(transient_steps):
            self.time += self.dt
            self.update_phases()

        # system is relaxed
        r_arr = np.zeros(steady_state)
        for t in range(steady_state):
            self.time += self.dt
            self.update_phases()
            r_arr[t] = self.order_parameter()[0]

        return np.mean(r_arr)
