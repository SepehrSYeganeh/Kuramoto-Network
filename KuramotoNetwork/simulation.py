from KuramotoNetwork import io
import numpy as np


class KuramotoGrid1d:
    dim = 1

    def __init__(self, N: int, sigma: float, kappa: float, xi: float,
                 steps: int, dt: float, snapshot_frames: int):
        """
        :param N: number of oscillators
        :param sigma: std of omega distribution
        :param kappa: coupling strength
        :param xi: noise strength
        :param steps: simulation steps
        :param dt: time step
        :param snapshot_frames: frames between snapshots
        """
        # initialize time
        self.steps = steps
        self.dt = dt
        self.time = 0
        self.snapshot_frames = snapshot_frames

        # initialize oscillators
        self.N = N
        self.sigma = sigma
        self.kappa = kappa
        self.xi = xi
        self.theta_arr = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        np.random.shuffle(self.theta_arr)
        self.omega_arr = np.random.normal(size=self.N, scale=self.sigma)

        # initialize data files
        io.init_data(self.N, self.dim)
        self.update_files()

    #################################################
    # order parameter
    #################################################
    def order_parameter(self) -> tuple[float, float]:
        """
        r * exp(i * psi) = mean(exp(i * theta_j))
        """
        x = np.mean(np.cos(self.theta_arr))
        y = np.mean(np.sin(self.theta_arr))
        r = np.sqrt(x ** 2 + y ** 2)
        psi = np.arctan2(y, x)
        return r, psi

    #################################################
    # update
    #################################################
    def update_phases(self) -> None:
        """
        d(theta)/dt = omega + coupling + noise
        """
        # coupling term
        lag_left = np.roll(self.theta_arr, 1) - self.theta_arr
        lag_right = np.roll(self.theta_arr, -1) - self.theta_arr
        coupling_term = self.kappa * (np.sin(lag_left) + np.sin(lag_right)) * self.dt

        # noise term
        noise_term = self.xi * np.random.normal(size=self.N, scale=np.sqrt(self.dt))

        # omega term
        omega_term = self.omega_arr * self.dt

        self.theta_arr += omega_term + coupling_term + noise_term

    def update_files(self) -> None:
        r, psi = self.order_parameter()
        io.append_data(self.dim, r, psi, self.theta_arr)

    #################################################
    # simulation
    #################################################
    def run(self) -> None:
        for t in range(self.steps):
            self.time += self.dt
            self.update_phases()
            if t % self.snapshot_frames == 0:
                self.update_files()

    #################################################
    # final_r
    #################################################
    def final_r(self, transient_steps, steady_state) -> float:
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

    def __init__(self, L: int, sigma: float, kappa: float, xi: float,
                 steps: int, dt: float, snapshot_frames: int):
        """
        :param L: length of grid
        :param sigma: std of omega distribution
        :param kappa: coupling strength
        :param xi: noise strength
        :param steps: simulation steps
        :param dt: time step
        :param snapshot_frames: frames between snapshots
        """
        # initialize time
        self.steps = steps
        self.dt = dt
        self.time = 0
        self.snapshot_frames = snapshot_frames

        # initialize parameters
        self.L = L
        self.N = int(L * L)
        self.sigma = sigma
        self.kappa = kappa
        self.xi = xi

        # initialize oscillators
        self.theta_arr = np.linspace(0, 2 * np.pi, self.N, endpoint=False)  # phase
        np.random.shuffle(self.theta_arr)
        self.theta_arr = np.reshape(self.theta_arr, (L, L))
        self.omega_arr = np.random.normal(size=self.N, scale=self.sigma)  # natural frequencies
        np.random.shuffle(self.omega_arr)
        self.omega_arr = np.reshape(self.omega_arr, (L, L))

        # initialize data files
        io.init_data(self.N, self.dim)
        self.update_files()

    #################################################
    # order parameter
    #################################################
    def order_parameter(self) -> tuple[float, float]:
        """
        r * exp(i * psi) = mean(exp(i * theta_j))
        """
        x = np.mean(np.cos(self.theta_arr))
        y = np.mean(np.sin(self.theta_arr))
        r = np.sqrt(x ** 2 + y ** 2)
        psi = np.arctan2(y, x)
        return r, psi

    #################################################
    # update
    #################################################
    def update_phases(self) -> None:
        """
        d(theta)/dt = omega + coupling + noise
        """
        # coupling term
        lag_left = np.roll(self.theta_arr, 1, axis=1) - self.theta_arr
        lag_right = np.roll(self.theta_arr, -1, axis=1) - self.theta_arr
        lag_down = np.roll(self.theta_arr, 1, axis=0) - self.theta_arr
        lag_up = np.roll(self.theta_arr, -1, axis=0) - self.theta_arr
        coupling_term = self.kappa * (np.sin(lag_left) + np.sin(lag_right) +
                                      np.sin(lag_up) + np.sin(lag_down)) * self.dt

        # noise term
        noise_term = self.xi * np.random.normal(size=self.N, scale=np.sqrt(self.dt)).reshape(self.L, self.L)

        # omega term
        omega_term = self.omega_arr * self.dt

        self.theta_arr += omega_term + coupling_term + noise_term

    def update_files(self) -> None:
        r, psi = self.order_parameter()
        io.append_data(self.dim, self.time, r, psi, self.theta_arr.flatten())

    #################################################
    # run simulation
    #################################################
    def run(self) -> None:
        for t in range(1, self.steps + 1):
            self.time += self.dt
            self.update_phases()
            if t % self.snapshot_frames == 0:
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
