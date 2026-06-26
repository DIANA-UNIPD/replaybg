import numpy as np
from numba import float64

from environment import jitclass_, njit_

@jitclass_([
    ("mu", float64),
    ("sigma", float64),
])
class LogNormal(object):
    """Log-normal distribution helper.

    The class provides density evaluation and bounded random sampling for a
    log-normal distribution parameterized by the mean and standard deviation
    of the underlying normal distribution.

    ...
    Attributes
    ----------
    mu : float
        Mean of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal distribution.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, mu, sigma):
        """Constructs all the necessary attributes for the LogNormal object.

        Parameters
        ----------
        mu : float
            Mean of the underlying normal distribution.
        sigma : float
            Standard deviation of the underlying normal distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x):
        """Evaluates the log-normal probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density. Must be positive.

        Returns
        -------
        float
            The probability density value at ``x``.
        """
        return 1 / (x * self.sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((np.log(x) - self.mu) / self.sigma) ** 2)

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.lognormal(self.mu, self.sigma), min_val), max_val)

@jitclass_([
    ("mu", float64),
    ("sigma", float64),
])
class Normal(object):
    """Normal distribution helper.

    The class provides density evaluation and bounded random sampling for a
    Gaussian distribution parameterized by its mean and standard deviation.

    ...
    Attributes
    ----------
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, mu, sigma):
        """Constructs all the necessary attributes for the Normal object.

        Parameters
        ----------
        mu : float
            Mean of the normal distribution.
        sigma : float
            Standard deviation of the normal distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x):
        """Evaluates the normal probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density.

        Returns
        -------
        float
            The probability density value at ``x``.
        """
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - self.mu) / self.sigma) ** 2)

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.normal(self.mu, self.sigma), min_val), max_val)

@jitclass_([
    ("alpha", float64),
    ("beta", float64),
    ("_log_normalizer", float64),
])
class Gamma:
    """Gamma distribution helper.

    The class provides density evaluation and bounded random sampling for a
    Gamma distribution parameterized in rate form.

    ...
    Attributes
    ----------
    alpha : float
        Shape parameter.
    beta : float
        Rate parameter, where ``beta = 1 / scale``.
    _log_normalizer : float
        Cached log-normalization constant.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, alpha, beta):
        """Constructs all the necessary attributes for the Gamma object.

        Parameters
        ----------
        alpha : float
            Shape parameter.
        beta : float
            Rate parameter, where ``beta = 1 / scale``.
        """
        self.alpha = alpha
        self.beta = beta  # rate parameterization (beta = 1/scale)
        self._log_normalizer = alpha * np.log(beta) - _gammaln(alpha)

    def evaluate(self, x):
        """Evaluates the Gamma probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density. Must be positive.

        Returns
        -------
        float
            The probability density value at ``x``. Returns ``0.0`` if
            ``x <= 0``.
        """
        if x <= 0:
            return 0.0
        return np.exp(self._log_normalizer + (self.alpha - 1) * np.log(x) - self.beta * x)

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.gamma(self.alpha, 1.0 / self.beta), min_val), max_val)

@jitclass_([
    ("a", float64),
    ("b", float64),
])
class Uniform(object):
    """Uniform distribution helper.

    The class provides density evaluation and bounded random sampling for a
    uniform distribution over the interval ``[a, b]``.

    ...
    Attributes
    ----------
    a : float
        Lower bound of the distribution support.
    b : float
        Upper bound of the distribution support.

    Methods
    -------
    evaluate(x):
        Evaluates the probability density function at ``x``.
    sample(min_val, max_val, seed):
        Draws a bounded random sample from the distribution.
    """

    def __init__(self, a, b):
        """Constructs all the necessary attributes for the Uniform object.

        Parameters
        ----------
        a : float
            Lower bound of the distribution support.
        b : float
            Upper bound of the distribution support.
        """
        self.a = a
        self.b = b

    def evaluate(self, x):
        """Evaluates the uniform probability density function.

        Parameters
        ----------
        x : float
            Value at which to evaluate the density.

        Returns
        -------
        float
            The probability density value at ``x``. Returns ``0.0`` if ``x`` is
            outside ``[a, b]``.
        """
        if x < self.a or x > self.b:
            return 0.0
        return 1.0 / (self.b - self.a)

    def sample(self, min_val, max_val, seed=None):
        """Draws a bounded random sample from the distribution.

        Parameters
        ----------
        min_val : float
            Minimum allowed sampled value.
        max_val : float
            Maximum allowed sampled value.
        seed : int, optional, default : None
            Optional random seed. For reproducibility.

        Returns
        -------
        float
            A sample from the distribution clipped to the given bounds.
        """
        if seed is not None:
            np.random.seed(seed)
        return min(max(np.random.uniform(self.a, self.b), min_val), max_val)

@njit_
def _gammaln(x):
    """Approximates the natural logarithm of the gamma function.

    Uses the Lanczos approximation.

    Parameters
    ----------
    x : float
        Positive input value.

    Returns
    -------
    float
        An approximation of ``log(Gamma(x))``.
    """
    coeffs = np.array([
         76.18009172947146,
        -86.50532032941677,
         24.01409824083091,
         -1.231739572450155,
          0.1208650973866179e-2,
         -0.5395239384953e-5
    ])
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    ser = 1.000000000190015
    for c in coeffs:
        y += 1.0
        ser += c / y
    return -tmp + np.log(2.5066282746310005 * ser / x)