import numpy as np
from activation_functions import relu, identity, tanh, sigmoid, softplus, softmax, heaviside
import numpy.linalg as alg
from scipy import sparse, stats
from numpy.random import Generator, PCG64


# all kinds of connection matrices
def CM_Initialise_Uniform(in_shape, out_shape, scale=1.0, bias=0.0):
    # uniform in [-1.0,1.0]
    CM = 2.0 * np.random.rand(in_shape, out_shape) - 1.0
    return scale * CM + bias


def CM_Initialise_Normal(in_shape, out_shape, scale=1.0, bias=0.0):
    CM = scale * np.random.randn(in_shape, out_shape) + bias
    return CM


def CM_Initialise_Orthogonal(in_shape, out_shape):
    n = max(in_shape, out_shape)
    H = np.random.randn(n, n)
    Q, R = alg.qr(H)
    return Q[:in_shape, :out_shape]


# spectral radius scaling
def CM_scale_specrad(CM, SR):
    # CM is the original connection matrix, SR is the desired spectral radius
    nCM = CM.copy()
    EV = np.max(np.absolute(alg.eigvals(CM)))
    return SR * nCM / EV

def init_matrices(n, input_connectivity, connectivity, spectral_radius=1, w_distribution=stats.uniform(0, 1),
                  win_distribution=stats.norm(1, 0.5), seed=111):
    #
    # The distribution generation functions
    #
    # stats.norm(1, 0.5)
    # stats.uniform(-1, 1)
    # stats.binom(n=1, p=0.5)
    bias_distribution = stats.uniform(0, 1)
    # To ensure reproducibility
    numpy_randomGen = Generator(PCG64(seed))
    w_distribution.random_state = numpy_randomGen
    win_distribution.random_state = numpy_randomGen
    bias_distribution.random_state = numpy_randomGen

    #
    # The generation of the matrices
    #
    if type(n) == int:
        n = (n, n)
    # Reservoir matrix
    W = sparse.random(n[0], n[1], density=connectivity, random_state=seed, data_rvs=w_distribution.rvs)
    # Input matrix
    Win = sparse.random(n[0], 1, density=input_connectivity, random_state=seed, data_rvs=win_distribution.rvs)

    # We set the diagonal to zero only for a square matrix
    if n[0] == n[1] and n[0] > 0:
        W.setdiag(0)
        W.eliminate_zeros()
        # Set the spectral radius
        # source : p104 David Verstraeten : largest eigenvalue = spectral radius
        eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
        sr = max(abs(eigen))
        W *= spectral_radius / sr

    # Bias matrice
    bias = bias_distribution.rvs(size=n[0])

    return Win, W, bias


# Generic ESN base class
# Requires external generation of the connection matrices:
#  * Input to reservoir (I2R)
#  * Bias to reservoir (bias_e)
#  * Reservoir to reservoir (R2R)
#  * Output to reservoir (O2R) - for output feedback only
#
# Other arguments:
#  * nonlinearity (function)
#  * stdev of noise (standard normal) on state values, input values (only for Batch processing)
#    and output values

class NN:
    def __init__(self, Win=np.array([]), bias_e=np.array([]), W_e=np.array([]),
                 W_i=np.array([]), W_ie=np.array([]),  W_ei=np.array([]), bias_i=np.array([]),
                 Wf=np.array([]), activation_function=lambda x: np.tanh(x), leaky_rate=1,
                 state_noise=0.0, inputnoise=0.0, readoutnoise=0.0):
        # number of neurons
        self.n_e = W_e.shape[0]
        self.n_i = W_e.shape[0]
        self.activation_function = activation_function
        self.leaky_rate = leaky_rate

        # Connection matrices: convention Row-to-Column !!
        self.Win = Win.copy()
        self.bias_e = bias_e.copy()
        self.W_e = W_e.copy()
        self.W_ei = W_ei.copy()
        self.W_ie = W_ie.copy()
        self.W_i = W_i.copy()
        self.bias_i = bias_i.copy()
        self.Wf = Wf.copy()

        self.inputnoise = inputnoise
        self.state_noise = state_noise
        self.readoutnoise = readoutnoise

        self.state_e = np.zeros((1, self.n_e))
        self.state_i = np.zeros((1, self.n_i))

    def Reset(self):
        self.state_e = np.zeros((1, self.n_e))
        self.state_i = np.zeros((1, self.n_i))

    def Copy(self):
        NewRes = NN(Win=self.Win.copy(), bias_e=self.bias_e.copy(), W_e=self.W_e.copy(),
                     W_i=np.array([]), W_ie=np.array([]),  W_ei=np.array([]), Wf=self.Wf.copy(),
                     activation_function=self.activation_function, state_noise=self.state_noise, inputnoise=self.inputnoise,
                     readoutnoise=self.readoutnoise)
        return NewRes

    def State(self):
        return self.state_e.copy(), self.state_i.copy()

    def _update(self, inputs, feedback):
        pre_s_e = (1 - self.leaky_rate) * self.state_e + self.leaky_rate * (self.W_e @ self.state_e - self.W_ie @ self.state_i) \
                  + self.Win @ inputs + self.bias_e + self.Wf @ feedback + self.state_noise * np.random.randn(1, self.n_e)
        pre_s_i = (1 - self.leaky_rate) * self.state_i + self.leaky_rate * self.W_ei @ self.state_i + self.bias_i \
                  + self.state_noise * np.random.randn(1, self.n_i)
        self.state_e = self.activation_function(pre_s_e)
        self.state_i = self.activation_function(pre_s_i)

    def Batch(self, inputs):
        steps = inputs.shape[0]
        states_e = np.zeros((steps, self.n_e))
        states_i = np.zeros((steps, self.n_i))
        for tt in range(steps):
            n_inp = inputs[tt:tt + 1, :]
            if self.inputnoise > 0:
                n_inp = n_inp + self.inputnoise * np.random.randn(n_inp.shape[0], n_inp.shape[1])
            self._update(n_inp, np.zeros((1, self.n_e)))
            states_e[tt:tt + 1, :], states_i[tt:tt + 1, :]= self.State()
        if self.readoutnoise > 0.0:
            states_e = states_e + self.readoutnoise * np.random.randn(states_e.shape[0].states.shape[1])
        return states_e, states_i

    def BatchWithFeedback(self, inputs, feedback):
        steps = inputs.shape[0]
        states_e = np.zeros((steps, self.n_e))
        states_i = np.zeros((steps, self.n_i))
        for tt in range(steps):
            self._update(inputs[tt:tt + 1, :], feedback[tt:tt + 1, :])
            states_e[tt:tt + 1, :], states_i[tt:tt + 1, :] = self.State()
        return states_e, states_i

