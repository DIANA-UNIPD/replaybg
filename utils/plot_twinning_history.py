import numpy as np
import matplotlib.pyplot as plt


def plot_twinning_history(history: dict, param_names: list[str] | None = None) -> plt.Figure:
    """Plot optimization history from a twinning run.

    Args:
        history: Dict with keys 'theta', 'log_prior', 'log_likelihood',
                 'log_posterior'. Each value is a list of recorded values per
                 function evaluation. 'theta' entries are numpy arrays.
        param_names: Optional list of parameter names for the theta legend.
                     Defaults to 'theta_0', 'theta_1', ...

    Returns:
        matplotlib Figure.
    """
    thetas = np.array(history['theta'])          # (n_evals, n_params)
    log_prior = np.array(history['log_prior'])
    log_likelihood = np.array(history['log_likelihood'])
    log_posterior = np.array(history['log_posterior'])

    n_evals = len(log_posterior)
    steps = np.arange(n_evals)

    if thetas.ndim == 1:
        thetas = thetas[:, np.newaxis]

    n_params = thetas.shape[1]
    if param_names is None:
        param_names = [f'theta_{i}' for i in range(n_params)]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # --- theta ---
    ax = axes[0]
    for i in range(n_params):
        mask = np.isfinite(thetas[:, i])
        ax.plot(steps[mask], thetas[mask, i], label=param_names[i], linewidth=0.8)
    ax.set_ylabel('Parameter value')
    ax.set_title('Parameter trajectory')
    if n_params <= 10:
        ax.legend(fontsize='small', ncol=min(n_params, 4))

    # --- log-prior ---
    ax = axes[1]

    ax.plot(steps[mask], log_prior[mask], color='tab:blue', linewidth=0.8)
    ax.set_ylabel('Log-prior')
    ax.set_title('Log-prior')

    # --- log-likelihood ---
    ax = axes[2]
    mask = np.isfinite(log_likelihood)
    ax.plot(steps[mask], log_likelihood[mask], color='tab:orange', linewidth=0.8)
    ax.set_ylabel('Log-likelihood')
    ax.set_title('Log-likelihood')

    # --- log-posterior ---
    ax = axes[3]
    mask = np.isfinite(log_posterior)
    ax.plot(steps[mask], log_posterior[mask], color='tab:green', linewidth=0.8)
    ax.set_ylabel('Log-posterior')
    ax.set_title('Log-posterior')
    ax.set_xlabel('Function evaluation')

    fig.tight_layout()
    return fig
