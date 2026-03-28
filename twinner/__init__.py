import numpy as np
from scipy.optimize import minimize


class Twinner():

    def __init__(self):
        pass

    def twin(self, model, data, unknown_parameters_prior):

        start_guess = []
        for up in unknown_parameters_prior.keys():
            start_guess.append(getattr(model, up))
        start_guess = np.array(start_guess)

        options = dict()
        options['maxiter'] = 100000
        options['maxfev'] = 100000
        options['disp'] = True

        result = minimize(self._neg_log_posterior, start_guess, method='Powell', args=(model, data, unknown_parameters_prior,), options=options)
        ret = dict()
        ret['fun'] = result.fun
        ret['x'] = result.x

        return ret

    def _log_prior(self, model, unknown_parameters_prior):
        lp = 0
        for up in unknown_parameters_prior.keys():
            lp += unknown_parameters_prior[up].evaluate(getattr(model, up))
        return lp

    def _log_likelihood(self, theta, model, data, unknown_parameters_prior):
        for i, up in enumerate(unknown_parameters_prior.keys()):
            setattr(model, up, theta[i])

        out = np.zeros(data.cho.shape[0],)
        for k in range(out.shape[0]):
            model.step([data.cho[k], data.bolus[k]])
            out[k] = model.output()

        #out = out[0::5]
        sdn = 5
        return -0.5 * np.sum(((out - data.glucose) / sdn) ** 2)

    def _neg_log_posterior(self, theta, model, data, unknown_parameters_prior):
        return -self._log_posterior(theta, model, data, unknown_parameters_prior)

    def _log_posterior(self, theta, model, data, unknown_parameters_prior):
        lp = self._log_prior(model, unknown_parameters_prior)
        if lp == -np.inf:
            return -np.inf
        else:
            return lp + self._log_likelihood(theta, model, data, unknown_parameters_prior)