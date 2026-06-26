class ReplayContext:
    """Per-step view of a replay simulation passed to every callback.

    A single instance is created once per ``replay()`` call and refreshed in
    place each minute (no per-step allocation). Callbacks address input channels
    by *name* via ``get_input`` / ``set_input`` / ``add_input`` and read history
    directly from ``output_history`` / ``input_history`` using ``k``.

    The context is model-agnostic: inputs are read and written in the model's own
    units. Any human-unit conversion (e.g. insulin U, carbs grams) is the
    callback's responsibility, using data it carries itself (e.g. body weight).

    Note:
        The data layer holds a bolus/meal value across the whole ``yts`` window.
        A callback that sets an input for a single minute therefore delivers a
        one-minute pulse, which is an acceptable approximation for correction
        boluses; spread the value over several steps yourself if it matters.

    Attributes:
        k: Current integration minute (the step being decided).
        t_hour: Hour-of-day at step ``k``.
        yts: Data/sensor cadence in minutes (e.g. 5); use it to self-gate.
        u: Live, mutable input vector for this step (model units). Callbacks edit
            it through the helpers; later callbacks see earlier edits.
        output_history: Interstitial glucose outputs; valid up to index ``k-1``.
        input_history: Applied input rows; valid up to index ``k-1``.
        measurement: Latest sensor measurement available at step ``k`` (held between
            sensor samples). When ``replay()`` is called without a sensor this carries
            the true model output, so policies reading it behave as if perfectly
            observed. Prefer this over ``output_history`` in control policies.
        measurement_history: Sensor measurement at integration resolution (zero-order
            held between samples); valid up to index ``k-1``. Mirrors ``output_history``
            but reflects the noisy sensor signal when a sensor is supplied.
        data_to_input: Mapping from channel index to channel name.
        model: The live model (read-only by convention) for advanced state such
            as plasma insulin (``model.Ip``) to derive insulin-on-board.
    """

    def __init__(self, rbg_data, model, output_history, input_history):
        self.k = 0
        self.t_hour = rbg_data.t_hour[0]
        self.yts = rbg_data.yts
        self.u = rbg_data.u[0].copy()
        self.output_history = output_history
        self.input_history = input_history
        self.measurement = output_history[0]
        self.measurement_history = output_history
        self.data_to_input = rbg_data.data_to_input
        self.model = model

        self._t_hour_all = rbg_data.t_hour
        self._name_to_idx = {name: idx for idx, name in rbg_data.data_to_input.items()}
        self._actions = []
        self._active_cb = None

    # -- lifecycle (called by the replay loop) --------------------------------

    def _advance(self, k, u):
        """Refresh the context for integration minute ``k`` with baseline ``u``."""
        self.k = k
        self.t_hour = self._t_hour_all[k]
        self.u = u

    # -- input accessors (name-based) -----------------------------------------

    def get_input(self, name) -> float:
        """Return the current value of input channel ``name`` (model units)."""
        return float(self.u[self._name_to_idx[name]])

    def set_input(self, name, value) -> None:
        """Set input channel ``name`` to ``value`` (model units)."""
        self.u[self._name_to_idx[name]] = value

    def add_input(self, name, value) -> None:
        """Add ``value`` (model units) to input channel ``name``."""
        self.u[self._name_to_idx[name]] += value

    # -- action logging -------------------------------------------------------

    def log(self, **fields) -> None:
        """Append an action record (``k``, ``callback`` auto-filled).

        Pass whatever fields make the action reconstructable later, e.g. the
        previous and new input values if you want to track what changed.
        """
        record = {
            "k": self.k,
            "callback": self._active_cb,
        }
        record.update(fields)
        self._actions.append(record)
