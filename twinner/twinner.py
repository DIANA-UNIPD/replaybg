import os
import time
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
    """A class that estimates model parameters by maximizing a posterior objective.

    The twinner performs MAP (maximum a posteriori) estimation: it minimizes the
    negative log-posterior (``-(log_prior + log_likelihood)``) over the unknown
    parameters using a multi-start Powell optimisation, optionally parallelised
    across CPUs with multiprocessing.

    ...
    Attributes
    ----------
    parallelize : bool
        A boolean that specifies whether to run the multi-start optimisation in
        parallel.
    n_jobs : int
        The number of parallel jobs to use (``-1`` means all available CPUs).
    n_starts : int
        The number of multi-start optimisations.
    verbose : bool
        A boolean that specifies the verbosity of the twinner.
    log_history : bool
        A boolean that specifies whether the optimisation history is recorded.
    history : dict
        The recorded optimisation history (only present when ``log_history`` is
        ``True``), with keys ``theta``, ``log_prior``, ``log_likelihood`` and
        ``log_posterior``.

    Methods
    -------
    twin(model, rbg_data, unknown_parameters_prior):
        Runs the multi-start MAP estimation and returns the best parameters.
    """

    def __init__(self,
                 parallelize: bool = True,
                 n_jobs: int | None = None,
                 n_starts: int = 64,
                 log_history: bool = False,
                 verbose: bool = True
    ):
        """Constructs all the necessary attributes for the Twinner object.

        Parameters
        ----------
        parallelize : bool, optional, default : True
            A boolean that specifies whether to run the multi-start optimisation
            in parallel using multiprocessing.
        n_jobs : int or None, optional, default : None
            The number of parallel jobs to use. If ``None``, all available CPUs
            are used (stored internally as ``-1``).
        n_starts : int, optional, default : 64
            An integer that specifies the number of multi-start optimisations.
        log_history : bool, optional, default : False
            A boolean that specifies whether to record the optimisation history.
        verbose : bool, optional, default : True
            A boolean that specifies the verbosity of the twinner.
        """
        self.parallelize = parallelize
        self.n_jobs = -1 if n_jobs is None else n_jobs
        self.n_starts = n_starts
        self.verbose = verbose

        self.log_history = log_history
        if self.log_history:
            self.history = dict()
            self.history['theta'] = []
            self.history['log_prior'] = []
            self.history['log_likelihood'] = []
            self.history['log_posterior'] = []

    def twin(self, model : Any, rbg_data, unknown_parameters_prior) -> dict:
        """Runs the multi-start MAP estimation for a model.

        Builds ``n_starts`` initial guesses by sampling the parameter priors,
        runs a bounded Powell optimisation of the negative log-posterior from
        each start (sequentially or in parallel), and selects the start with the
        lowest objective. Integer parameters are rounded and all parameters are
        clipped to their bounds before being returned.

        Parameters
        ----------
        model : object
            A model instance being fit, implementing ``reset(theta_dict)``,
            ``step(u, t)`` and ``output(t)``.
        rbg_data : object
            A data object used to compute the likelihood.
        unknown_parameters_prior : dict
            A dictionary describing the priors and bounds for each parameter to
            estimate (keys ``prior``, ``min``, ``max`` and optional ``integer``).

        Returns
        -------
        dict
            A dictionary with keys ``fun`` (the best negative log-posterior
            value) and ``x`` (the best-fit parameter vector, rounded and clipped
            to the bounds).
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

        # Resolve the effective number of jobs (used both for running and for the header)
        n_jobs = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs

        # Print a run header so the user can sanity-check the twinning setup
        if self.verbose:
            param_names = list(unknown_parameters_prior.keys())
            mode = f'parallel ({n_jobs} jobs)' if self.parallelize else 'sequential'
            print('--- Twinning (MAP estimation) ---')
            print(f'  starts          : {self.n_starts}')
            print(f'  mode            : {mode}')
            print(f'  parameters ({len(param_names)})  : {", ".join(param_names)}')

        t0 = time.perf_counter()

        if self.parallelize:
            # Set the context
            ctx = multiprocessing.get_context('fork' if os.name != 'nt' else 'spawn')
            # Run the optimization in parallel
            with ctx.Pool(processes=n_jobs) as pool:
                raw = list(tqdm(
                    pool.imap(_run_optimization, start_guesses),
                    total=self.n_starts,
                    desc='Twinning using MAP',
                    disable=not self.verbose
                ))
            results = [r for r, _ in raw]
            if self.log_history:
                for _, h in raw:
                    if h is not None:
                        for k in self.history:
                            self.history[k].extend(h[k])
        else:
            # Run the optimization sequentially; history accumulates in-process via _neg_log_posterior
            raw = [_run_optimization(a)
                   for a in tqdm(start_guesses, desc='Twinning', disable=not self.verbose)]
            results = [r for r, _ in raw]

        elapsed = time.perf_counter() - t0

        # Get the best result
        best = min(results, key=lambda r: r.fun)

        # Print a convergence summary across all starts
        if self.verbose:
            funs = np.array([r.fun for r in results], dtype=float)
            finite = funs[np.isfinite(funs)]
            n_failed = len(funs) - len(finite)
            print('--- Twinning summary ---')
            print(f'  elapsed         : {elapsed:.1f} s')
            print(f'  failed starts   : {n_failed} / {len(funs)}')
            if len(finite):
                print(f'  objective (neg log-posterior)')
                print(f'    best          : {best.fun:.3f}')
                print(f'    min/med/max   : {finite.min():.3f} / '
                      f'{np.median(finite):.3f} / {finite.max():.3f}')
            else:
                print('  WARNING: no start produced a finite objective '
                      '(check priors/bounds and the model).')

        # Round integer parameters and clip to bounds (Powell has no native bound support)
        clipped_x = []
        for x, (k, v) in zip(best.x, unknown_parameters_prior.items()):
            if v.get('integer', False):
                x = float(round(x))
            clipped_x.append(np.clip(x, v['min'], v['max']))
        clipped_x = np.array(clipped_x)

        # Print the best-fit parameters, flagging any that sit at their bounds
        if self.verbose:
            print('--- Best-fit parameters ---')
            for x, (k, v) in zip(clipped_x, unknown_parameters_prior.items()):
                lo, hi = v['min'], v['max']
                tol = 1e-6 * max(1.0, abs(hi - lo))
                flags = []
                if x <= lo + tol:
                    flags.append('at min bound')
                elif x >= hi - tol:
                    flags.append('at max bound')
                if v.get('integer', False):
                    flags.append('integer')
                suffix = f'   [{", ".join(flags)}]' if flags else ''
                print(f'  {k:<14} = {x:.4g}   (bounds: {lo:g}..{hi:g}){suffix}')

        # Return the best result
        ret = dict()
        ret['fun'] = best.fun
        ret['x'] = clipped_x
        return ret

    def _log_prior(self, model, unknown_parameters_prior):
        """Computes the log prior probability of the model parameters.

        Parameters
        ----------
        model : object
            A model instance containing the current parameter values.
        unknown_parameters_prior : dict
            A dictionary describing the priors for each parameter. Each prior
            must provide an ``evaluate(value)`` method.

        Returns
        -------
        float
            The sum of log prior contributions for all parameters, or ``-inf``
            when any parameter is outside its valid range.
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
        """Computes the log likelihood of the observed data under the model.

        The model is simulated forward using the input sequence in ``rbg_data``.
        The predicted output is then compared against the observed glucose
        measurements using a Gaussian error model with a constant coefficient of
        variation.

        Parameters
        ----------
        model : object
            A model instance to simulate.
        rbg_data : object
            A data object containing inputs, output timestamps, and observed
            glucose values.

        Returns
        -------
        float
            The log likelihood value for the simulated trajectory.
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
        """Returns the negative log posterior for optimization.

        Parameters
        ----------
        theta : numpy.ndarray
            A parameter vector in the natural (constrained) parameter space,
            ordered to match ``unknown_parameters_prior``.
        model : object
            A model instance being fit.
        rbg_data : object
            A data object used to compute the likelihood.
        unknown_parameters_prior : dict
            A dictionary describing priors and bounds for the parameters.

        Returns
        -------
        float
            The negative log posterior value.
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

        The parameter vector is used directly in the natural (constrained)
        parameter space: each entry is written into the model's typed parameter
        dict (integer parameters are rounded) before ``model.reset``. Bounds are
        enforced by the prior, which returns ``-inf`` outside ``[min, max]``.

        Parameters
        ----------
        theta : numpy.ndarray
            A parameter vector in the natural (constrained) parameter space,
            ordered to match ``unknown_parameters_prior``.
        model : object
            A model instance being updated with candidate parameters.
        rbg_data : object
            A data object used to compute the likelihood.
        unknown_parameters_prior : dict
            A dictionary describing priors and bounds for the parameters.

        Returns
        -------
        tuple
            The triple ``(log_prior, log_likelihood, log_posterior)``. Any
            component that is invalid is returned as ``-np.inf``.
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
    """Runs a single Powell optimisation start (module-level worker).

    Defined at module scope so it can be pickled and dispatched to worker
    processes by multiprocessing. Reads the shared optimisation arguments from
    the module-global ``_worker_args``.

    Parameters
    ----------
    args : tuple
        The pair ``(i, start_guess)`` where ``i`` is the start index and
        ``start_guess`` is the initial parameter vector.

    Returns
    -------
    tuple
        The pair ``(result, history)`` where ``result`` is the
        ``scipy.optimize.OptimizeResult`` of the start and ``history`` is a
        snapshot of the twinner history (or ``None`` when history logging is
        disabled).
    """
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
