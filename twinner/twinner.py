from typing import Any

import numpy as np
from scipy.optimize import minimize

from distributions import to_unconstrained, to_constrained, log_jacobian_single

from numba import float64, types
from numba.typed import Dict


class Twinner():
    """Estimate model parameters by maximizing a posterior objective (MAP estimation).

    The twinning procedure searches for the most likely set of model
    parameters given observed data and prior distributions. Parameters are
    optimized in an unconstrained space and mapped back to their constrained
    values during evaluation.

    Attributes:
        None
    """

    def __init__(self):
        """Initialize a ``Twinner`` instance."""
        pass

    def twin(self, model : Any, data, unknown_parameters_prior, environment) -> dict:
        """Run twinning estimation for a model.

        The method builds an initial guess from the current model parameters,
        transforms them to an unconstrained space, and then uses Powell
        optimization to minimize the negative log posterior.

        Args:
            model: Model instance whose parameters will be estimated. The model
                must expose attributes for every key in ``unknown_parameters_prior``
                and implement ``reset()``, ``step()``, and ``output()`` methods.
            data: Data object containing the observed inputs and measurements used
                to evaluate the likelihood.
            unknown_parameters_prior: Dictionary describing the parameters to fit.
                Each entry should include:
                - ``prior``: object with an ``evaluate(value)`` method
                - ``min``: lower bound for the parameter
                - ``max``: upper bound for the parameter

        Returns:
            ret: A dictionary containing the optimization result:
                - ``fun``: Final objective value.
                - ``x``: Optimized parameter values in unconstrained space.
        """

        options = dict()
        options['maxiter'] = 100000
        options['maxfev'] = 100000
        options['disp'] = True
        """
        best = None
        for _ in range(10):
            # random start within each parameter's range
            start_guess = []
            for k, v in unknown_parameters_prior.items():
                theta_rand = np.random.uniform(v['min'], v['max']) # TODO: change with sample method
                start_guess.append(to_unconstrained(theta_rand, v['min'], v['max']))

            result = minimize(self._neg_log_posterior, start_guess,
                              method='Powell', args=(model, data, unknown_parameters_prior), options=options)

            if best is None or result.fun < best.fun:
                best = result
        """
        # TODO: add multiple start guesses
        # TODO: parallelize over start_guesses. It must use a parallelize flag and a n_cores parameters to organize it
        rng = np.random.default_rng(environment.seed)
        start_guess = []
        for k, v in unknown_parameters_prior.items():
            start_guess.append(to_unconstrained(getattr(model, k), v['min'],
                                                v['max']))
        start_guess = np.array(start_guess)
        best = minimize(self._neg_log_posterior, start_guess, method='Powell',
                          args=(model, data, unknown_parameters_prior,), options=options)

        ret = dict()
        ret['fun'] = best.fun
        ret['x'] = best.x

        return ret

    def _log_prior(self, model, unknown_parameters_prior):
        """Compute the log prior probability of the model parameters.

        Args:
            model: Model instance containing the current parameter values.
            unknown_parameters_prior: Dictionary describing the priors for each
                parameter. Each prior must provide an ``evaluate(value)`` method.

        Returns:
            float: Sum of log prior contributions for all parameters.
        """
        lp = 0
        for up, v in unknown_parameters_prior.items():
            lp += v['prior'].evaluate(getattr(model, up))
        return lp

    def _log_likelihood(self, model, data, ):
        """Compute the log likelihood of the observed data under the model.

        The model is simulated forward using the input sequence in ``data``.
        The predicted output is then compared against the observed glucose
        measurements using a Gaussian error model.

        Args:
            model: Model instance to simulate.
            data: Data object containing inputs, output timestamps, and observed
                glucose values.

        Returns:
            float: Log likelihood value for the simulated trajectory.
        """
        out = np.zeros(data.tsteps, )
        for k in range(out.shape[0]):
            model.step(data.u[k], k)
            # TODO: add the possibiliy to track oall the states during twinning (mainly for debugging)
            out[k] = model.output()

        out = out[0::data.yts]
        sdn = 5 #TODO: mettere da altra parte
        return -0.5 * np.sum(((out[data.glucose_idxs] - data.glucose[data.glucose_idxs]) / sdn) ** 2) #TODO: far diventare data.glucose => data.y e data.glucose_idxs => data.y_idxs

    def _neg_log_posterior(self, theta, model, data, unknown_parameters_prior):
        """Return the negative log posterior for optimization.

        Args:
            theta: Parameter vector in unconstrained space.
            model: Model instance being fit.
            data: Data object used to compute the likelihood.
            unknown_parameters_prior: Dictionary describing priors and bounds for
                the parameters.

        Returns:
            float: Negative log posterior value.
        """
        return -self._log_posterior(theta, model, data, unknown_parameters_prior)

    def _log_posterior(self, theta, model, data, unknown_parameters_prior):
        """Compute the log posterior for a parameter vector.

        The input parameter vector is assumed to be in unconstrained space.
        Parameters are mapped back to constrained values before updating the
        model. A Jacobian correction is included for the change of variables.

        Args:
            theta: Parameter vector in unconstrained space.
            model: Model instance being updated with candidate parameters.
            data: Data object used to compute the likelihood.
            unknown_parameters_prior: Dictionary describing priors and bounds for
                the parameters.

        Returns:
            float: Log posterior value, or ``-np.inf`` if the prior is invalid.
        """
        # thetadict must be a numba typed dict
        thetadict = Dict.empty(key_type=types.unicode_type, value_type=float64)
        total_jacobian = 0.0

        for i, k in enumerate(unknown_parameters_prior.keys()):
            thetadict[k] = to_constrained(theta[i], unknown_parameters_prior[k]['min'],
                                          unknown_parameters_prior[k]['max'])
            total_jacobian += log_jacobian_single(theta[i], unknown_parameters_prior[k]['min'],
                                                  unknown_parameters_prior[k]['max'])

        model.reset(thetadict)
        lp = self._log_prior(model, unknown_parameters_prior)
        if lp == -np.inf:
            return -np.inf
        else:
            ll = self._log_likelihood(model, data, )
            if ll == -np.inf:
                return -np.inf
            return lp + ll + total_jacobian #TODO: study the theory behind the jacobian
