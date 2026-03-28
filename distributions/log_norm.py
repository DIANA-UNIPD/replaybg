import numpy as np
from numba import float32
from environment.config import jitclass_

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


#lognormtype = LogNormal.class_type.instance_type


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


#normtype = Normal.class_type.instance_type
