import numpy as np
from numba import float32
from environment.config import jitclass_, njit_
import math


@jitclass_([
    ("mu", float32),
    ("sigma", float32),
])
class LogNormal(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x):
        """
        Computes the logarithm of the normal pdf evaluated at given x with given mu and sigma.

        Parameters
        ----------
        x: float
            The value where to evaluate the normal pdf.
        mu: float
            The mean of the normal distribution.
        sigma: float
            The standard deviation of the normal distribution.

        Returns
        -------
        l_norm: float
            The logarithm of the normal pdf evaluated at given x with given mu and sigma.

        Raises
        ------
        None

        See Also
        --------
        None

        Examples
        --------
        None
        """
        return np.log(1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - self.mu) / self.sigma) ** 2))


@jitclass_([
    ("mu", float32),
    ("sigma", float32),
])
class LogLogNormal(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x):
        """
        Computes the logarithm of the normal pdf evaluated at given x with given mu and sigma.

        Parameters
        ----------
        x: float
            The value where to evaluate the normal pdf.
        mu: float
            The mean of the normal distribution.
        sigma: float
            The standard deviation of the normal distribution.

        Returns
        -------
        l_norm: float
            The logarithm of the normal pdf evaluated at given x with given mu and sigma.

        Raises
        ------
        None

        See Also
        --------
        None

        Examples
        --------
        None
        """
        return np.log(1 / (x * self.sigma * np.sqrt(2 * np.pi)) * np.exp(
            - ((np.log(x) - self.mu) ** 2) / (2 * (self.sigma ** 2))))


# lognormtype = LogNormal.class_type.instance_type


@jitclass_([
    ("mu", float32),
    ("sigma", float32),
])
class Normal(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, x):
        """
        Computes the logarithm of the normal pdf evaluated at given x with given mu and sigma.

        Parameters
        ----------
        x: float
            The value where to evaluate the normal pdf.
        mu: float
            The mean of the normal distribution.
        sigma: float
            The standard deviation of the normal distribution.

        Returns
        -------
        l_norm: float
            The logarithm of the normal pdf evaluated at given x with given mu and sigma.

        Raises
        ------
        None

        See Also
        --------
        None

        Examples
        --------
        None
        """
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - self.mu) / self.sigma) ** 2)


# normtype = Normal.class_type.instance_type

@jitclass_([
    ("alpha", float32),
    ("beta", float32),
])
class LogGamma(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, x):
        """
        Computes the logarithm of the gamma pdf evaluated at given x with given alpha and beta.

        Parameters
        ----------
        x: float
            The value where to evaluate the normal pdf.
        alpha: float
            The alpha value of the gamma distribution at hand.
        beta: float
            The beta value of the gamma distribution at hand.

        Returns
        -------
        l_gam: float
            The logarithm of the gamma pdf evaluated at given x with given alpha and beta.

        Raises
        ------
        None

        See Also
        --------
        None

        Examples
        --------
        None
        """
        return -np.inf if x < 0 else np.log(
            (self.beta ** self.alpha * x ** (self.alpha - 1) * np.exp(-self.beta * x)) / math.gamma(self.alpha))


@njit_
def to_constrained(x, a=0, b=1):
    """
    Passes from search space to physiological space.
    :param x:
    :param a:
    :param b:
    :return:
    """
    return a + (b - a) / (1. + np.exp(-x))


@njit_
def to_unconstrained(x, a, b):
    """
    Passes from physiological space to search space.
    :param x:
    :param a:
    :param b:
    :return:
    """
    t = (x - a) / (b - a)
    return np.log(t / (1 - t))


@njit_
def log_jacobian_single(x, a, b):
    s = 1 / (1 + np.exp(-x))
    return np.log((b - a) * s * (1 - s))
