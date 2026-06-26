import numpy as np
import matplotlib.pyplot as plt

from utils.numba_dicts import to_typed_f32_dict


def plot_twinning(
    rbg_data,
    model,
    theta: dict | None = None,
    thresholds: list[float] | None = None,
    input_groups: list[list[int]] | None = None,
    ts_min: float = 1.0,
    output_label: str = "Fit",
    observation_label: str = "Data",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot the outcome of a ``ReplayBG.twin()`` run.

    Twinning fits the model parameters to the observed data; this utility shows
    how well the fitted model reproduces them. The model is simulated forward
    with the estimated ``theta`` (the same ``reset``/``step``/``output`` pass the
    twinner uses for the likelihood), and the resulting trace is compared against
    the observations that were fitted.

    The figure follows the same rationale as :func:`plot_replay`: a vertical
    stack of subplots sharing a common time axis.

    1. **Fit** (first subplot): the simulated model ``output`` for the estimated
       ``theta`` as a continuous line and the observed data (``rbg_data.y`` at the
       non-missing indices ``rbg_data.y_idxs``) as markers. Each value in
       ``thresholds`` is drawn as a dashed horizontal line spanning the whole
       width, acting as a visual reference level.
    2. **Inputs** (one subplot per input *group*): the inputs that drove the
       fit, labelled via ``data_to_input``. By default every channel gets its own
       subplot; pass ``input_groups`` to overlay channels together.

    The utility is model-agnostic: nothing about the meaning of the output or
    inputs is assumed. Channel names come straight from ``data_to_input``.

    Args:
        rbg_data: The data object passed to :meth:`ReplayBG.twin` (e.g. a
            ``MultiMealT1DData``). Must expose ``u``, ``data_to_input``,
            ``tsteps``, ``yts``, ``y`` and ``y_idxs``.
        model: A model instance implementing the model interface contract
            (``reset(theta_dict)``, ``step(u, t)``, ``output(t)``).
        theta: The estimated parameters as returned in ``twin()['theta']``. When
            given, the model is reset with them before simulating. When ``None``,
            the model is simulated in its current state (e.g. already constructed
            with ``theta0``).
        thresholds: Optional reference levels drawn as dashed horizontal lines on
            the fit subplot.
        input_groups: Optional list of channel-index groups. Each group becomes
            one subplot with its channels overlaid. ``None`` plots one channel
            per subplot, in ``data_to_input`` order.
        ts_min: Minutes represented by one integration step, used to scale the
            time axis. Defaults to 1.
        output_label: Legend label for the simulated fit line.
        observation_label: Legend label for the observed-data markers.
        figsize: Optional figure size. Defaults to a height that grows with the
            number of subplots.

    Returns:
        matplotlib Figure.
    """
    inputs = np.asarray(rbg_data.u)
    data_to_input = rbg_data.data_to_input

    # --- simulate the fitted model forward (matches the twinner's pass) -------
    if theta is not None:
        model.reset(to_typed_f32_dict(theta))

    tsteps = int(rbg_data.tsteps)
    output = np.zeros(tsteps)
    output[0] = model.output(0)
    for k in range(1, tsteps):
        model.step(inputs[k], k)
        output[k] = model.output(k)

    t = np.arange(tsteps) * ts_min

    # --- resolve the subplot layout ------------------------------------------
    if input_groups is None:
        input_groups = [[idx] for idx in sorted(data_to_input.keys())]

    n_rows = 1 + len(input_groups)
    if figsize is None:
        figsize = (10, 2.2 * n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)

    # --- 1. fit subplot ------------------------------------------------------
    ax = axes[0]
    ax.plot(t, output, color="tab:blue", linewidth=1.2, label=output_label)

    # Observations live at data resolution; each sample i sits yts integration
    # steps apart. Only the non-missing samples (y_idxs) were used in the fit.
    y = np.asarray(rbg_data.y, dtype=float)
    y_idxs = np.asarray(rbg_data.y_idxs)
    obs_t = y_idxs * rbg_data.yts * ts_min
    ax.plot(
        obs_t, y[y_idxs],
        linestyle="none", marker="o", markersize=3,
        color="tab:red", alpha=0.7, label=observation_label,
    )

    for level in (thresholds or []):
        ax.axhline(level, linestyle="--", color="gray", linewidth=0.8)

    ax.set_ylabel(output_label)
    ax.set_title("Twinning fit")
    ax.legend(fontsize="small", loc="upper right")

    # --- 2. input subplots ---------------------------------------------------
    # Inputs are typically sparse impulses (boluses, meals) held over a sample
    # window, so they read best as stems at the non-zero steps rather than as a
    # dense line. Only non-zero values are stemmed to avoid clutter.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for row, group in enumerate(input_groups, start=1):
        ax = axes[row]
        for j, idx in enumerate(group):
            name = data_to_input.get(idx, f"input_{idx}")
            color = colors[j % len(colors)]
            channel = inputs[:, idx]
            nz = np.flatnonzero(channel)
            if nz.size:
                ax.stem(
                    t[nz], channel[nz],
                    linefmt=color, markerfmt="none", basefmt=" ", label=name,
                )
            else:
                # Keep the channel in the legend even when it never fires.
                ax.plot([], [], color=color, label=name)
        ax.set_ylabel("Input")
        names = ", ".join(data_to_input.get(idx, f"input_{idx}") for idx in group)
        ax.set_title(names)
        if len(group) > 1:
            ax.legend(fontsize="small", loc="upper right")

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    return fig
