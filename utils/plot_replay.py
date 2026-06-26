import numpy as np
import matplotlib.pyplot as plt


def plot_replay(
    replay_results: dict,
    thresholds: list[float] | None = None,
    input_groups: list[list[int]] | None = None,
    action_field: str = "u",
    ts_min: float = 1.0,
    output_label: str = "Output",
    measurement_label: str = "Measurement",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot the outcome of a ``ReplayBG.replay()`` run.

    The figure is a vertical stack of subplots sharing a common time axis:

    1. **Output** (first subplot): the replayed ``output`` as a continuous line
       and, when a sensor was used, the replayed ``measurement`` as markers.
       Each value in ``thresholds`` is drawn as a dashed horizontal line
       spanning the whole width, acting as a visual reference level.
    2. **Inputs** (one subplot per input *group*): the applied ``input``
       channels, labelled via ``data_to_input``. By default every channel gets
       its own subplot; pass ``input_groups`` to overlay channels together.
    3. **Callback actions** (one subplot per distinct callback): the values a
       callback logged, stemmed at the integration minute of each action.

    The utility is model-agnostic: nothing about the meaning of the output,
    inputs, or actions is assumed. Channel names come straight from
    ``data_to_input`` and callback labels from the action log.

    Args:
        replay_results: The dict returned by :meth:`ReplayBG.replay`. Must
            contain ``output``, ``input``, ``data_to_input`` and ``actions``;
            ``measurement`` and ``measurement_time`` are used when present.
        thresholds: Optional reference levels drawn as dashed horizontal lines
            on the output subplot.
        input_groups: Optional list of channel-index groups. Each group becomes
            one subplot with its channels overlaid. ``None`` plots one channel
            per subplot, in ``data_to_input`` order.
        action_field: Name of the scalar field to plot for each callback action.
            When a record lacks it, the function falls back to the record's lone
            non-metadata field (anything other than ``k``/``callback``).
        ts_min: Minutes represented by one integration step, used to scale the
            time axis. Defaults to 1.
        output_label: Legend label for the replayed output line.
        measurement_label: Legend label for the replayed measurement markers.
        figsize: Optional figure size. Defaults to a height that grows with the
            number of subplots.

    Returns:
        matplotlib Figure.
    """
    output = np.asarray(replay_results["output"])
    inputs = np.asarray(replay_results["input"])
    data_to_input = replay_results["data_to_input"]
    actions = replay_results.get("actions", []) or []

    tsteps = output.shape[0]
    t = np.arange(tsteps) * ts_min

    # --- resolve the subplot layout -----------------------------------------
    if input_groups is None:
        input_groups = [[idx] for idx in sorted(data_to_input.keys())]

    # Distinct callbacks, in first-seen order, that actually logged something.
    callback_names = list(dict.fromkeys(rec["callback"] for rec in actions))

    n_rows = 1 + len(input_groups) + len(callback_names)
    if figsize is None:
        figsize = (10, 2.2 * n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)

    # --- 1. output subplot ---------------------------------------------------
    ax = axes[0]
    ax.plot(t, output, color="tab:blue", linewidth=1.2, label=output_label)

    measurement = replay_results.get("measurement")
    measurement_time = replay_results.get("measurement_time")
    if measurement is not None and measurement_time is not None:
        measurement = np.asarray(measurement)
        measurement_time = np.asarray(measurement_time)
        mask = np.isfinite(measurement)
        ax.plot(
            measurement_time[mask] * ts_min, measurement[mask],
            linestyle="none", marker="o", markersize=3,
            color="tab:red", alpha=0.7, label=measurement_label,
        )

    for level in (thresholds or []):
        ax.axhline(level, linestyle="--", color="gray", linewidth=0.8)

    ax.set_ylabel(output_label)
    ax.set_title("Output")
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

    # --- 3. callback action subplots ----------------------------------------
    base = 1 + len(input_groups)
    for row, cb_name in enumerate(callback_names, start=base):
        ax = axes[row]
        ks, values = [], []
        for rec in actions:
            if rec["callback"] != cb_name:
                continue
            value = _action_value(rec, action_field)
            if value is None:
                continue
            ks.append(rec["k"] * ts_min)
            values.append(value)
        if ks:
            ax.stem(ks, values, basefmt=" ")
        ax.set_ylabel(action_field)
        ax.set_title(f"Callback: {cb_name}")

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    return fig


def _action_value(record: dict, action_field: str):
    """Return the scalar to plot for an action record.

    Prefers ``action_field``; otherwise falls back to the record's single
    non-metadata field so the utility stays agnostic to callback internals.
    """
    if action_field in record:
        return record[action_field]
    extras = [k for k in record if k not in ("k", "callback")]
    if len(extras) == 1:
        return record[extras[0]]
    return None
