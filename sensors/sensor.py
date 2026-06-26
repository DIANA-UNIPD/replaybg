from abc import ABC, abstractmethod


class Sensor(ABC):
    """Output-agnostic sensor that turns a model output into a measurement.

    A sensor observes the model's output trace during ``replay()`` and produces a
    (typically noisy, sub-sampled) measurement of it. It is deliberately generic:
    it knows nothing about glucose or any specific model. Concrete subclasses (e.g.
    :class:`~sensors.vettoretti19_cgm.Vettoretti19CGM`) implement the actual error
    model in :meth:`measure`.

    Attributes:
        ts: Sample time of the sensor, in integration steps. The replay loop takes a
            measurement when ``k % ts == 0``.
        t_offset: Time offset of the sensor (0 when new). Used when a sensor object is
            shared across multiple replay runs.
        max_lifetime: Maximum lifetime of the sensor, in integration steps. When it is
            exceeded the replay loop reconnects a fresh sensor realization.
        connected_at: Integration step at which the current sensor was connected
            (utility to manage sensor lifetime across multiple replay calls).
    """

    def __init__(self):
        """Constructs all the necessary attributes for the Sensor object."""
        self.ts = 5
        self.t_offset = 0

        self.connect_new()

        # Set max lifetime (integration steps)
        self.max_lifetime = 1400 * 10
        self.connected_at = 0

    def connect_new(self, connected_at=0):
        """Connects a fresh sensor by (re)sampling its error realization.

        Args:
            connected_at: Integration step at which the sensor is connected.
        """
        # Set the offset
        self.t_offset = 0

        # Set the connection time
        self.connected_at = connected_at

    @abstractmethod
    def measure(self, value: float, past_values, t: float) -> float:
        """Produce a measurement from the current and past model outputs.

        Args:
            value: The current model output. This is the value being measured.
            past_values: The series of past model outputs (``out[:k]``).
            t: Current time, in days from when the sensor was connected.

        Returns:
            The measurement at the current time.
        """
        pass

    def add_offset(self, to_add: float) -> None:
        """Add an offset to the sensor life.

        Used when the sensor object is shared across multiple replay runs.

        Args:
            to_add: The offset to add to the sensor life.
        """
        self.t_offset += to_add
