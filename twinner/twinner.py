import os
from typing import Any

import numpy as np
from scipy.optimize import minimize

import warnings

from tqdm import tqdm
import multiprocessing
from numba import float64, types
from numba.typed import Dict

_worker_args = None

class Twinner():
    """Estimate model parameters by maximizing a posterior objective.

    The twinning procedure searches for the most likely set of model
    parameters given observed data and prior distributions. Parameters are
    optimized in an unconstrained space and mapped back to constrained values
    during evaluation.

    Attributes:
        parallelize: Whether to run optimizations in parallel.
        n_jobs: Number of worker processes to use when parallelizing. Use ``-1``
            to select all available CPU cores.
        n_starts: Number of random initial guesses used to seed the optimizer.
    """

    def __init__(self,
                 parallelize: bool = True,
                 n_jobs: int = -1,
                 n_starts: int = 64,
    ):
        """Initialize a ``Twinner`` instance."""
        self.parallelize = parallelize
        self.n_jobs = n_jobs
        self.n_starts = n_starts


    def twin(self, model : Any, data, unknown_parameters_prior, environment) -> dict:
        """Run twinning for a model.

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
                - ``min``: lower bound for the parameter.
                - ``max``: upper bound for the parameter.
            environment: Environment object containing runtime settings used by
                the twinning pipeline.

        Returns:
            dict: Dictionary containing the optimization result with keys:
                - ``fun``: Final objective value.
                - ``x``: Optimized parameter values in unconstrained space.
        """

        # Set the worker arguments
        global _worker_args
        _worker_args = (self._neg_log_posterior, unknown_parameters_prior, model, data)

        # Build initial guesses for the parameters using their priors
        start_guesses = []
        for i in range(self.n_starts):
            start_guess = np.array([v['prior'].sample(v['min'], v['max'], i)
                                    for v in unknown_parameters_prior.values()])
            start_guesses.append((i, start_guess))

        if self.parallelize:
            # Set the number of jobs
            n_jobs = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
            # Set the context
            ctx = multiprocessing.get_context('fork' if os.name != 'nt' else 'spawn')
            # Run the optimization in parallel
            with ctx.Pool(processes=n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(_run_optimization, start_guesses),
                    total=self.n_starts,
                    desc='Twinning'
                ))
        else:
            # Run the optimization sequentially
            results = [_run_optimization(a) for a in tqdm(start_guesses, desc='Twinning')]

        # Get the best result
        best = min(results, key=lambda r: r.fun)

        # Return the best result
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
        # Iterate over the parameters and compute the log prior
        lp = 0
        for up, v in unknown_parameters_prior.items():
            parameter_value = getattr(model, up)
            # If the parameter is outside the valid range, return -inf
            if parameter_value > v['max'] or parameter_value < v['min']:
                return -np.inf
            # Otherwise, add the log prior contribution
            lp += np.log(v['prior'].evaluate(parameter_value))
        # Return the sum of log prior contributions
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
        # Simulate the model forward and get the output
        out = np.zeros(data.tsteps, )
        for k in np.arange(1,out.shape[0]):
            model.step(data.u[k], k)
            out[k] = model.output(k)
        # Subsample the output to match the sampling rate of the data
        out = out[0::data.yts]

        # Calculate the log-likelihood with a Gaussian error model (constant coefficient of variation)
        cv = 0.05  # constant coefficient of variation (5%) #TODO: make this a parameter
        residuals = out[data.y_idxs] - data.y[data.y_idxs]
        sdn = cv * np.abs(out[data.y_idxs])
        return -0.5 * np.sum((residuals / sdn) ** 2)

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
        # Just return the negative log-posterior
        return -self._log_posterior(theta, model, data, unknown_parameters_prior)

    def _log_posterior(self, theta, model, data, unknown_parameters_prior):
        """Compute the log posterior for a parameter vector.

        The input parameter vector is assumed to be in unconstrained space.
        Parameters are mapped back to constrained values before updating the
        model.

        Args:
            theta: Parameter vector in unconstrained space.
            model: Model instance being updated with candidate parameters.
            data: Data object used to compute the likelihood.
            unknown_parameters_prior: Dictionary describing priors and bounds for
                the parameters.

        Returns:
            float: Log-posterior value, or ``-np.inf`` if the prior is invalid.
        """

        # Create the theta input dictionary (note: order is maintained by construction)
        theta_dict = Dict.empty(key_type=types.unicode_type, value_type=float64)
        for i, k in enumerate(unknown_parameters_prior.keys()):
            val = theta[i]
            if unknown_parameters_prior[k].get('integer', False):
                val = float(round(val))
            theta_dict[k] = val

        # Reset the model with the new parameters
        model.reset(theta_dict)

        # Calculate log-prior
        lp = self._log_prior(model, unknown_parameters_prior)
        if lp == -np.inf or np.isnan(lp):
            return -np.inf
        # Calculate log-likelihood
        ll = self._log_likelihood(model, data)
        if ll == -np.inf or np.isnan(ll):
            return -np.inf

        # Return log-posterior
        return lp + ll

def _run_optimization(args):
    # Unpack the arguments
    i, start_guess = args

    # Get the worker arguments
    neg_log_posterior_fn, unknown_parameters_prior, model, data = _worker_args

    # Run the optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return minimize(neg_log_posterior_fn, start_guess, method='Powell',
                        args=(model, data, unknown_parameters_prior,),
                        options={
                            'maxiter': 100000,
                            'maxfev': 100000,
                            'disp': False
                        })