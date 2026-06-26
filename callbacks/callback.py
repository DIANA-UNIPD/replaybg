class ReplayCallback:
    """Base class for replay control policies.

    A callback is invoked once per integration minute, *before* the model steps,
    so it can inspect the simulation so far and modify the inputs the model is
    about to consume. Subclasses:

    - store **hyperparameters** in ``__init__`` (supplied at the ``replay()`` call
      site when the instance is created);
    - keep **memory** in instance attributes, which persist across all timesteps
      of a single ``replay()`` call and remain inspectable on the instance after
      the run finishes;
    - override :meth:`action` to read the simulation state from the context and
      modify the current-step inputs via ``ctx.set_input`` / ``ctx.add_input``.

    Callbacks run in the order they are passed to ``replay()``; each mutates the
    same input vector in place, so later callbacks observe earlier edits. The
    return value of :meth:`action` is ignored.
    """

    #: Identifies this callback in the action log. Defaults to the class name.
    #: Set a distinct ``name`` when using two instances of the same class.
    name: str = None

    #: The replay ``rbg_data`` is injected here before the run starts, so
    #: callbacks can access data-specific quantities (e.g. ``rbg_data.body_weight``)
    #: for their own unit conversions. ``None`` until ``replay()`` is called.
    rbg_data = None

    def action(self, ctx) -> None:
        """Inspect ``ctx`` and modify the current-step inputs in place.

        Args:
            ctx: The :class:`~callbacks.context.ReplayContext` for the current step.
        """
        raise NotImplementedError
