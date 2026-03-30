import numpy as np
from scipy.optimize import minimize

from distributions import to_unconstrained, to_constrained, log_jacobian_single
from numba import float32, types
from numba.typed import Dict

class Twinner():

    def __init__(self):
        pass

    def twin(self, model, data, unknown_parameters_prior):

        start_guess = []
        for k, v in unknown_parameters_prior.items():
            start_guess.append(to_unconstrained(getattr(model, k), v['min'],
                                                v['max']))
        start_guess = np.array(start_guess)

        options = dict()
        options['maxiter'] = 100000
        options['maxfev'] = 100000
        options['disp'] = True

        result = minimize(self._neg_log_posterior, start_guess, method='Powell',
                          args=(model, data, unknown_parameters_prior,), options=options)
        ret = dict()
        ret['fun'] = result.fun
        ret['x'] = result.x

        return ret

    def _log_prior(self, model, unknown_parameters_prior):
        lp = 0
        for up, v in unknown_parameters_prior.items():
            lp += v['prior'].evaluate(getattr(model, up))
        return lp

    def _log_likelihood(self, model, data, ):
        out = np.zeros(data.tsteps, )
        for k in range(out.shape[0]):
            model.step(data.u[k], k)
            out[k] = model.output()

        out = out[0::5]
        sdn = 5
        return -0.5 * np.sum(((out - data.glucose) / sdn) ** 2)

    def _neg_log_posterior(self, theta, model, data, unknown_parameters_prior):
        return -self._log_posterior(theta, model, data, unknown_parameters_prior)

    def _log_posterior(self, theta, model, data, unknown_parameters_prior):
        # thetadict must be a numba typed dict
        thetadict = Dict.empty(key_type=types.unicode_type, value_type=float32)
        total_jacobian = 0.

        for i, k in enumerate(unknown_parameters_prior.keys()):
            thetadict[k] = to_constrained(theta[i], unknown_parameters_prior[k]['min'],
                                          unknown_parameters_prior[k]['max'])
            total_jacobian += log_jacobian_single(thetadict[k], unknown_parameters_prior[k]['min'],
                                                  unknown_parameters_prior[k]['max'])

        model.reset(thetadict)
        lp = self._log_prior(model, unknown_parameters_prior)
        print(model.Gb)
        if lp == -np.inf:
            return -np.inf
        else:
            # TODO: fix total jacobian
            return lp + self._log_likelihood(model, data, ) #+ total_jacobian
