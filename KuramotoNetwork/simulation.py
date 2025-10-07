from KuramotoNetwork import io
import numpy as np


def initialize_theta_1D(N: int) -> np.ndarray:
    np.random.seed(42)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    np.random.shuffle(theta)
    return theta


def initialize_theta_2D(L: int) -> np.ndarray:
    N = int(L * L)
    theta = initialize_theta_1D(N)
    theta.resize((L, L))
    return theta


def initialize_Normal_omega_1D(N: int, sigma: float) -> np.ndarray:
    np.random.seed(42)
    omega = np.random.normal(size=N, scale=sigma)
    omega -= np.mean(omega)
    return omega


def initialize_Normal_omega_2D(L: int, sigma: float) -> np.ndarray:
    N = int(L * L)
    omega = initialize_Normal_omega_1D(N, sigma)
    omega.resize((L, L))
    return omega


class KuramotoGrid1D:
    dim = 1

    def __init__(self, path: str,
                 N: int, sigma: float, kappa: float, xi: float,
                 steps: int, dt: float, snapshot_frames: int,
                 init_theta: np.ndarray, init_omega: np.ndarray):
        """
        :param N: number of oscillators
        :param sigma: std of omega distribution
        :param kappa: coupling strength
        :param xi: noise strength
        :param steps: simulation steps
        :param dt: time increment
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
        self.theta_arr = init_theta
        self.omega_arr = init_omega

        # initialize data file
        self.filename = io.make_filename(self.dim, self.N, self.kappa, self.sigma, self.xi, self.steps)
        self.path = io.init_data_file(path, self.filename, self.N)
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
        # noise term
        noise_term = self.xi * np.random.normal(size=self.N, scale=np.sqrt(self.dt))

        # omega term
        omega_term = self.omega_arr * self.dt

        # coupling term
        lag_left = np.roll(self.theta_arr, 1) - self.theta_arr
        lag_right = np.roll(self.theta_arr, -1) - self.theta_arr
        coupling_term = self.kappa * (np.sin(lag_left) + np.sin(lag_right)) * self.dt

        theta_bar = self.theta_arr + omega_term + coupling_term + noise_term
        lag_left_bar = np.roll(self.theta_arr, 1) - theta_bar
        lag_right_bar = np.roll(self.theta_arr, -1) - theta_bar
        coupling_term_bar = self.kappa * (np.sin(lag_left_bar) + np.sin(lag_right_bar)) * self.dt

        self.theta_arr += omega_term + noise_term + (coupling_term + coupling_term_bar) / 2

    def update_files(self) -> None:
        r, psi = self.order_parameter()
        io.append_data(self.path, self.time, r, psi, self.theta_arr)

    #################################################
    # simulation
    #################################################
    def run(self) -> None:
        for t in range(self.steps):
            self.time += self.dt
            self.update_phases()
            if t % self.snapshot_frames == 0:
                self.update_files()


###################################################################################
###################################################################################
class KuramotoGrid2D:
    dim = 2

    def __init__(self, path: str,
                 L: int, sigma: float, kappa: float, xi: float,
                 steps: int, dt: float, snapshot_frames: int,
                 init_theta: np.ndarray, init_omega: np.ndarray):
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
        self.theta_arr = init_theta
        self.omega_arr = init_omega

        # initialize data files
        self.filename = io.make_filename(self.dim, self.N, self.kappa, self.sigma, self.xi, self.steps)
        self.path = io.init_data_file(path, self.filename, self.N)
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
        # omega term
        omega_term = self.omega_arr * self.dt

        # noise term
        noise_term = self.xi * np.random.normal(size=self.N, scale=np.sqrt(self.dt)).reshape(self.L, self.L)

        # coupling term
        lag_left = np.roll(self.theta_arr, 1, axis=1) - self.theta_arr
        lag_right = np.roll(self.theta_arr, -1, axis=1) - self.theta_arr
        lag_down = np.roll(self.theta_arr, 1, axis=0) - self.theta_arr
        lag_up = np.roll(self.theta_arr, -1, axis=0) - self.theta_arr
        coupling_term = self.kappa * (np.sin(lag_left) + np.sin(lag_right) +
                                      np.sin(lag_up) + np.sin(lag_down)) * self.dt

        theta_bar = self.theta_arr + omega_term + coupling_term + noise_term
        lag_left_bar = np.roll(self.theta_arr, 1, axis=1) - theta_bar
        lag_right_bar = np.roll(self.theta_arr, -1, axis=1) - theta_bar
        lag_down_bar = np.roll(self.theta_arr, 1, axis=0) - theta_bar
        lag_up_bar = np.roll(self.theta_arr, -1, axis=0) - theta_bar
        coupling_term_bar = self.kappa * (np.sin(lag_left_bar) + np.sin(lag_right_bar) +
                                          np.sin(lag_up_bar) + np.sin(lag_down_bar)) * self.dt

        self.theta_arr += omega_term + noise_term + (coupling_term + coupling_term_bar) / 2

    def update_files(self) -> None:
        r, psi = self.order_parameter()
        io.append_data(self.path, self.time, r, psi, self.theta_arr.flatten())

    #################################################
    # run simulation
    #################################################
    def run(self) -> None:
        for t in range(1, self.steps + 1):
            self.time += self.dt
            self.update_phases()
            if t % self.snapshot_frames == 0:
                self.update_files()
