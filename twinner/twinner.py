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
    """

    def __init__(self,
                 parallelize: bool = True,
                 n_jobs: int | None = None,
                 n_starts: int = 64,
                 log_history: bool = False
    ):
        """Initialize a ``Twinner`` instance.
        """
        self.parallelize = parallelize
        self.n_jobs = -1 if n_jobs is None else n_jobs
        self.n_starts = n_starts

        self.log_history = log_history
        if self.log_history:
            self.history = dict()
            self.history['theta'] = []
            self.history['log_prior'] = []
            self.history['log_likelihood'] = []
            self.history['log_posterior'] = []

    def twin(self, model : Any, rbg_data, unknown_parameters_prior) -> dict:
        """Run twinning for a model.
        """

        # Set the worker arguments
        global _worker_args
        _worker_args = (self._neg_log_posterior, self._log_posterior, unknown_parameters_prior, model, rbg_data)

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
                raw = list(tqdm(
                    pool.imap(_run_optimization, start_guesses),
                    total=self.n_starts,
                    desc='Twinning using MAP'
                ))
            results = [r for r, _ in raw]
            if self.log_history:
                for _, h in raw:
                    if h is not None:
                        for k in self.history:
                            self.history[k].extend(h[k])
        else:
            # Run the optimization sequentially; history accumulates in-process via _neg_log_posterior
            raw = [_run_optimization(a) for a in tqdm(start_guesses, desc='Twinning')]
            results = [r for r, _ in raw]

        # Get the best result
        best = min(results, key=lambda r: r.fun)

        # Round integer parameters and clip to bounds (Powell has no native bound support)
        clipped_x = []
        for x, (k, v) in zip(best.x, unknown_parameters_prior.items()):
            if v.get('integer', False):
                x = float(round(x))
            clipped_x.append(np.clip(x, v['min'], v['max']))
        clipped_x = np.array(clipped_x)

        # Return the best result
        ret = dict()
        ret['fun'] = best.fun
        ret['x'] = clipped_x
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

    def _log_likelihood(self, model, rbg_data):
        """Compute the log likelihood of the observed data under the model.

        The model is simulated forward using the input sequence in ``data``.
        The predicted output is then compared against the observed glucose
        measurements using a Gaussian error model.

        Args:
            model: Model instance to simulate.
            rbg_data: Data object containing inputs, output timestamps, and observed
                glucose values.

        Returns:
            float: Log likelihood value for the simulated trajectory.
        """
        # Simulate the model forward and get the output
        out = np.zeros(rbg_data.tsteps, )
        out[0] = model.output(0)
        for k in np.arange(1,out.shape[0]):
            model.step(rbg_data.u[k], k)
            out[k] = model.output(k)
        # Subsample the output to match the sampling rate of the data
        out = out[0::rbg_data.yts]

        # TODO: enable the choice of multiple ll shapes
        # Two-term noise model: σ² = σ_add² + (cv·|ŷ|)²
        #sigma_add = 5.0          # mg/dL, literature default
        #cv        = 0.05         # 5%
        #sdn = np.sqrt(sigma_add**2 + (cv * np.abs(out[rbg_data.y_idxs]))**2)
        #residuals = out[rbg_data.y_idxs] - rbg_data.y[rbg_data.y_idxs]

        # Calculate the log-likelihood with a Gaussian error model (constant coefficient of variation)
        cv = 0.05  # constant coefficient of variation (5%) #TODO: make this a parameter
        residuals = out[rbg_data.y_idxs] - rbg_data.y[rbg_data.y_idxs]
        sdn = cv * np.abs(out[rbg_data.y_idxs])
        subsampling = 6 #TODO: decide where to evaluate ll (now every 30 minutes but maybe we can do something smarter)
        sdn = sdn[0::subsampling]
        residuals = residuals[0::subsampling]
        return -0.5 * np.sum((residuals / sdn) ** 2)

    def _neg_log_posterior(self, theta, model, rbg_data, unknown_parameters_prior):
        """Return the negative log posterior for optimization.

        Args:
            theta: Parameter vector in unconstrained space.
            model: Model instance being fit.
            rbg_data: Data object used to compute the likelihood.
            unknown_parameters_prior: Dictionary describing priors and bounds for
                the parameters.

        Returns:
            float: Negative log posterior value.
        """
        # Just return the negative log-posterior
        log_prior, log_likelihood, log_post = self._log_posterior(theta, model, rbg_data, unknown_parameters_prior)

        # log the history
        if self.log_history:
            self.history['theta'].append(theta)
            self.history['log_prior'].append(log_prior)
            self.history['log_likelihood'].append(log_likelihood)
            self.history['log_posterior'].append(log_post)

        return -log_post

    def _log_posterior(self, theta, model, rbg_data, unknown_parameters_prior):
        """Compute log-prior, log-likelihood, and log-posterior in one model pass.

        The input parameter vector is assumed to be in unconstrained space.
        Parameters are mapped back to constrained values before updating the
        model.

        Args:
            theta: Parameter vector in unconstrained space.
            model: Model instance being updated with candidate parameters.
            rbg_data: Data object used to compute the likelihood.
            unknown_parameters_prior: Dictionary describing priors and bounds for
                the parameters.

        Returns:
            tuple: ``(log_prior, log_likelihood, log_posterior)``. Any component
                that is invalid is returned as ``-np.inf``.
        """

        # Create the theta input dictionary (note: order is maintained by construction)
        theta_dict = Dict.empty(key_type=types.unicode_type, value_type=float64)
        for i, k in enumerate(unknown_parameters_prior.keys()):
            val = theta[i]
            #val = np.clip(theta[i], unknown_parameters_prior[k]['min'], unknown_parameters_prior[k]['max'])
            if unknown_parameters_prior[k].get('integer', False):
                val = int(round(val))
            theta_dict[k] = val

        # Reset the model with the new parameters
        model.reset(theta_dict)

        # Calculate log-prior
        lp = self._log_prior(model, unknown_parameters_prior)
        if lp == -np.inf or np.isnan(lp):
            return -np.inf, -np.inf, -np.inf
        # Calculate log-likelihood
        ll = self._log_likelihood(model, rbg_data)
        if ll == -np.inf or np.isnan(ll):
            return lp, -np.inf, -np.inf

        # Return log-posterior
        return lp, ll, lp + ll

def _run_optimization(args):
    # Unpack the arguments
    i, start_guess = args

    # Get the worker arguments
    neg_log_posterior_fn, log_posterior_components_fn, unknown_parameters_prior, model, rbg_data = _worker_args

    # Run the optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(neg_log_posterior_fn, start_guess, method='Powell',
                          args=(model, rbg_data, unknown_parameters_prior,),
                          options={
                              'maxiter': 100000,
                              'maxfev': 100000,
                              'disp': False,
                          })

    # Return a snapshot of history so the parent process can merge it (needed in parallel mode
    # where worker-side mutations to self.history never reach the parent).
    twinner = neg_log_posterior_fn.__self__
    history = {k: list(v) for k, v in twinner.history.items()} if twinner.log_history else None
    return result, history
