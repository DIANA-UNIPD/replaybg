"""Tests for the plotting utilities.

These exercise what the figures actually contain rather than how they look: that
every logged callback action reaches the canvas, that each channel is drawn with
the right mark, that masking and grouping resolve as documented, and that each
history panel is filtered by its own values. The rendering itself runs on the
``Agg`` backend, so no display is needed.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from utils.plot_common import (
    IMPULSE_MAX_DUTY,
    KIND_DENSE,
    attach_hover,
    resolve_input_layout,
    series,
)
from utils.plot_replay import _resolve_action_field, plot_replay
from utils.plot_twinning_history import plot_twinning_history


DATA_TO_INPUT = {0: "meal", 1: "bolus", 2: "basal", 3: "t_hour"}


@pytest.fixture(autouse=True)
def close_figures():
    """Closes every figure a test opened, so the suite does not leak them."""
    yield
    plt.close("all")


@pytest.fixture
def replay_results():
    """A minimal ``replay()``-shaped dict with two callbacks logging actions.

    The channels cover both drawing regimes: ``meal``/``bolus`` fire on a single
    step (impulses) while ``basal``/``t_hour`` are non-zero throughout (dense).
    """
    tsteps = 100
    inputs = np.zeros((tsteps, 4))
    inputs[10, 0] = 40.0        # meal: one impulse
    inputs[20, 1] = 2.0         # bolus: one impulse
    inputs[:, 2] = 0.3          # basal: non-zero everywhere
    inputs[:, 3] = np.arange(tsteps) / 60.0

    return {
        "output": 120 + 30 * np.sin(np.arange(tsteps) / 10.0),
        "input": inputs,
        "data_to_input": DATA_TO_INPUT,
        "actions": [
            {"k": 30, "callback": "CorrectionBolus", "cb_u": 1.5, "prev_u": 0.0, "new_u": 15.0},
            {"k": 70, "callback": "CorrectionBolus", "cb_u": 2.5, "prev_u": 0.0, "new_u": 25.0},
            {"k": 50, "callback": "HypoTreatment", "ht_g": 15, "recent_bolus_u": 0.0,
             "prev_u": 0.0, "new_u": 150.0},
        ],
    }


def _panel(fig, title):
    """Returns the axes whose (left-aligned) panel title matches ``title``."""
    return next(ax for ax in fig.axes if ax.get_title(loc="left") == title)


def _markers(ax):
    """Returns the (x, y) of the marker line drawn by ``stem`` on ``ax``."""
    line = next(l for l in ax.lines if l.get_marker() not in ("", "None"))
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


# --- callback actions -------------------------------------------------------

def test_plot_replay_draws_every_logged_action(replay_results):
    """Each callback gets a panel carrying exactly the actions it logged.

    Regression: the utility used to default ``action_field`` to ``"u"``, a field
    no callback logs, and silently drew empty panels.
    """
    fig = plot_replay(replay_results)

    x, y = _markers(_panel(fig, "Callback: CorrectionBolus"))
    assert list(x) == [30, 70]
    assert list(y) == [1.5, 2.5]

    x, y = _markers(_panel(fig, "Callback: HypoTreatment"))
    assert list(x) == [50]
    assert list(y) == [15]


def test_plot_replay_labels_actions_with_the_field_plotted(replay_results):
    """The y label names the field each callback actually logged."""
    fig = plot_replay(replay_results)
    assert _panel(fig, "Callback: CorrectionBolus").get_ylabel() == "cb_u"
    assert _panel(fig, "Callback: HypoTreatment").get_ylabel() == "ht_g"


def test_plot_replay_scales_action_times_by_ts_min(replay_results):
    """Action steps are placed on the same scaled time axis as the traces."""
    fig = plot_replay(replay_results, ts_min=5.0)
    x, _ = _markers(_panel(fig, "Callback: CorrectionBolus"))
    assert list(x) == [150, 350]


def test_plot_replay_without_actions_has_no_callback_panel(replay_results):
    """A replay with no callbacks gets output + input panels only."""
    replay_results["actions"] = []
    fig = plot_replay(replay_results)
    assert not [ax for ax in fig.axes if ax.get_title(loc="left").startswith("Callback")]


# --- action field resolution ------------------------------------------------

def test_resolve_action_field_prefers_first_numeric_field():
    """``None`` picks the first numeric field, the quantity the action delivered."""
    records = [{"k": 1, "callback": "cb", "cb_u": 1.5, "prev_u": 0.0}]
    assert _resolve_action_field(records, None) == "cb_u"


def test_resolve_action_field_skips_non_numeric_fields():
    """Non-numeric fields are never chosen: they cannot be stemmed."""
    records = [{"k": 1, "callback": "cb", "reason": "high", "cb_u": 1.5}]
    assert _resolve_action_field(records, None) == "cb_u"


def test_resolve_action_field_honours_explicit_choice():
    """An explicit field wins over the automatic one."""
    records = [{"k": 1, "callback": "cb", "cb_u": 1.5, "new_u": 15.0}]
    assert _resolve_action_field(records, "new_u") == "new_u"


def test_resolve_action_field_falls_back_when_field_absent():
    """A callback that does not log the requested field still gets a panel."""
    records = [{"k": 1, "callback": "cb", "ht_g": 15.0}]
    assert _resolve_action_field(records, "cb_u") == "ht_g"


def test_resolve_action_field_returns_none_without_numeric_fields():
    """A callback logging nothing numeric resolves to no field at all."""
    records = [{"k": 1, "callback": "cb", "reason": "high"}]
    assert _resolve_action_field(records, None) is None


def test_plot_replay_survives_actions_with_no_numeric_field(replay_results):
    """A non-numeric-only callback gets an empty panel instead of raising."""
    replay_results["actions"] = [{"k": 5, "callback": "Noisy", "reason": "high"}]
    fig = plot_replay(replay_results)
    assert _panel(fig, "Callback: Noisy").get_ylabel() == ""


# --- input channels ---------------------------------------------------------

def test_impulse_channels_are_stemmed_and_dense_ones_stepped(replay_results):
    """Sparse channels read as stems; dense ones as a step line.

    Regression: stemming a channel that fires on nearly every step painted a
    solid block (e.g. basal).
    """
    fig = plot_replay(replay_results)

    # bolus fires once -> stem, which carries a marker line.
    assert [l for l in _panel(fig, "bolus").lines if l.get_marker() not in ("", "None")]

    # basal is non-zero throughout -> a plain step line, no markers, all points.
    basal_lines = _panel(fig, "basal").lines
    assert not [l for l in basal_lines if l.get_marker() not in ("", "None")]
    assert len(basal_lines[0].get_xdata()) == len(replay_results["output"])


def test_channel_at_the_duty_threshold_is_still_stemmed():
    """The impulse/step decision follows the documented duty fraction."""
    tsteps = 100
    n_fire = int(IMPULSE_MAX_DUTY * tsteps)
    inputs = np.zeros((tsteps, 1))
    inputs[:n_fire, 0] = 1.0

    fig = plot_replay({
        "output": np.zeros(tsteps), "input": inputs,
        "data_to_input": {0: "edge"}, "actions": [],
    })
    assert [l for l in _panel(fig, "edge").lines if l.get_marker() not in ("", "None")]


def test_never_firing_channel_keeps_its_legend_entry():
    """An all-zero channel still gets a panel and a labelled (empty) artist."""
    # hover=False keeps the crosshair's own lines out of the artist count.
    fig = plot_replay({
        "output": np.zeros(10), "input": np.zeros((10, 1)),
        "data_to_input": {0: "quiet"}, "actions": [],
    }, hover=False)
    ax = _panel(fig, "quiet")
    assert [l.get_label() for l in ax.lines] == ["quiet"]


# --- layout -----------------------------------------------------------------

def test_masked_channels_get_no_panel(replay_results):
    """Masked channels are dropped from the default one-per-panel layout."""
    fig = plot_replay(replay_results, mask_inputs=[3])
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert "t_hour" not in titles
    assert "basal" in titles


def test_masking_preserves_channel_colours():
    """A channel keeps its colour when another is masked out."""
    unmasked, colors = resolve_input_layout(DATA_TO_INPUT, None, None)
    _, masked_colors = resolve_input_layout(DATA_TO_INPUT, None, [0])
    assert colors[2] == masked_colors[2]


def test_grouping_overlays_channels_on_one_panel(replay_results):
    """An explicit group puts its channels on a single panel."""
    fig = plot_replay(replay_results, input_groups=[[0, 1], [2]], mask_inputs=[3])
    assert _panel(fig, "meal, bolus")
    assert _panel(fig, "basal")


def test_group_emptied_by_masking_is_dropped(replay_results):
    """A group whose every channel is masked gets no panel at all."""
    fig = plot_replay(replay_results, input_groups=[[0, 3], [3]], mask_inputs=[3])
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert "meal" in titles          # group 1 survives, minus the masked channel
    assert titles.count("") == 0     # group 2 vanished rather than drawing empty


def test_thresholds_shade_the_span_between_outermost_levels(replay_results):
    """Two or more reference levels also shade the band between them."""
    fig = plot_replay(replay_results, thresholds=[70, 180])
    assert _panel(fig, "Output").patches

    fig = plot_replay(replay_results, thresholds=[70])
    assert not _panel(fig, "Output").patches


# --- twinning history -------------------------------------------------------

def test_history_panels_are_masked_by_their_own_values():
    """Each term drops only its own non-finite samples.

    Regression: the log-prior panel reused the ``mask`` left over from the
    parameter loop, so it was filtered by the last parameter's finite values --
    dropping valid log-priors and plotting NaN ones.
    """
    n = 20
    thetas = np.ones((n, 2))
    thetas[5:10, 1] = np.nan        # last parameter missing here ...
    log_prior = np.arange(n, dtype=float)
    log_prior[15:18] = np.nan       # ... log-prior missing somewhere else

    fig = plot_twinning_history({
        "theta": list(thetas), "log_prior": log_prior,
        "log_likelihood": np.arange(n, dtype=float),
        "log_posterior": np.arange(n, dtype=float),
    })

    x, y = fig.axes[1].lines[0].get_xdata(), fig.axes[1].lines[0].get_ydata()
    assert np.array_equal(x, np.flatnonzero(np.isfinite(log_prior)))
    assert not np.isnan(y).any()


def test_history_handles_a_single_parameter():
    """A 1-D theta history is treated as one parameter rather than one eval."""
    n = 10
    fig = plot_twinning_history({
        "theta": [np.array([float(i)]) for i in range(n)],
        "log_prior": np.arange(n, dtype=float),
        "log_likelihood": np.arange(n, dtype=float),
        "log_posterior": np.arange(n, dtype=float),
    }, hover=False)
    assert len(fig.axes[0].lines) == 1
    assert np.array_equal(fig.axes[0].lines[0].get_ydata(), np.arange(n, dtype=float))


# --- hover readout ----------------------------------------------------------

def test_hover_reports_dense_series_anywhere():
    """A dense series is read at the nearest sample wherever the cursor is."""
    fig = plot_replay({
        "output": np.arange(100, dtype=float), "input": np.zeros((100, 1)),
        "data_to_input": {0: "quiet"}, "actions": [],
    })
    ax = _panel(fig, "Output")
    text = fig._rbg_crosshair._readout(ax, 42.4)
    assert "Output: 42" in text


def test_hover_ignores_sparse_series_away_from_a_sample():
    """A sparse series is reported only near one of its samples.

    Between stems it has no value, so reporting the nearest would invent a hold
    that was never there.
    """
    tsteps = 100
    inputs = np.zeros((tsteps, 1))
    inputs[10, 0] = 40.0
    fig = plot_replay({
        "output": np.zeros(tsteps), "input": inputs,
        "data_to_input": {0: "meal"}, "actions": [],
    })
    ax = _panel(fig, "meal")
    assert "meal: 40" in fig._rbg_crosshair._readout(ax, 10.2)
    assert fig._rbg_crosshair._readout(ax, 80.0) == ""


def test_hover_moves_the_readout_between_panels_at_the_same_x(replay_results):
    """Crossing into another panel at the same x moves the readout there."""
    from matplotlib.backend_bases import MouseEvent

    fig = plot_replay(replay_results, mask_inputs=[3])
    crosshair = fig._rbg_crosshair
    output_ax, basal_ax = _panel(fig, "Output"), _panel(fig, "basal")

    def move_to(ax, x, y):
        px, py = ax.transData.transform((x, y))
        fig.canvas.callbacks.process(
            "motion_notify_event", MouseEvent("motion_notify_event", fig.canvas, px, py))

    move_to(output_ax, 40.0, 120.0)
    assert crosshair._annots[output_ax].get_visible()

    move_to(basal_ax, 40.0, 0.3)
    assert crosshair._annots[basal_ax].get_visible()
    assert not crosshair._annots[output_ax].get_visible()


def test_hover_tracks_every_panel_together(replay_results):
    """One cursor position is drawn on all panels, so they can be read down."""
    from matplotlib.backend_bases import MouseEvent

    fig = plot_replay(replay_results, mask_inputs=[3])
    crosshair = fig._rbg_crosshair
    ax = _panel(fig, "Output")
    px, py = ax.transData.transform((40.0, 120.0))
    fig.canvas.callbacks.process(
        "motion_notify_event", MouseEvent("motion_notify_event", fig.canvas, px, py))

    assert all(v.get_visible() for v in crosshair._vlines)
    assert {float(v.get_xdata()[0]) for v in crosshair._vlines} == {40.0}


def test_hover_can_be_disabled(replay_results):
    """``hover=False`` leaves no crosshair on the figure."""
    fig = plot_replay(replay_results, hover=False)
    assert not hasattr(fig, "_rbg_crosshair")


def test_hover_does_not_disturb_the_data_limits():
    """Adding the cursor artists leaves the axis limits the data established."""
    results = {
        "output": np.arange(100, dtype=float), "input": np.zeros((100, 1)),
        "data_to_input": {0: "quiet"}, "actions": [],
    }
    without = plot_replay(results, hover=False).axes[0]
    with_hover = plot_replay(results, hover=True).axes[0]
    assert without.get_xlim() == with_hover.get_xlim()
    assert without.get_ylim() == with_hover.get_ylim()


def test_hover_skips_empty_series():
    """An empty series contributes nothing rather than raising."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([], [])
    crosshair = attach_hover(fig, [ax], {ax: [series("gone", [], [], KIND_DENSE)]})
    assert crosshair._readout(ax, 1.0) == ""
