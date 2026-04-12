import numpy as np
from numba import float64, int16
from numba.typed import Dict
from numba import types

from environment.config import jitclass_

theta0_type = types.DictType(types.unicode_type, float64)
x0_type = types.DictType(types.unicode_type, float64)
JITCLASS_SPEC = [
    ("_G0", float64),
    ("_X0", float64),
    ("_Qsto1_0", float64),
    ("_Qsto2_0", float64),
    ("_Qgut_0", float64),
    ("_Isc10", float64),
    ("_Isc20", float64),
    ("_Ip0", float64),
    ("_IG0", float64),
    ("G", float64[:]),
    ("X", float64[:]),
    ("Qsto1", float64[:]),
    ("Qsto2", float64[:]),
    ("Qgut", float64[:]),
    ("Isc1", float64[:]),
    ("Isc2", float64[:]),
    ("Ip", float64[:]),
    ("IG", float64[:]),
    ("SI", float64),
    ("SG", float64),
    ("Gb", float64),
    ("p2", float64),
    ("r1", float64),
    ("r2", float64),
    ("tau", int16),
    ("ka2", float64),
    ("kd", float64),
    ("ke", float64),
    ("kabs", float64),
    ("beta", int16),
    ("kempt", float64),
    ("f", float64),
    ("VG", float64),
    ("VI", float64),
    ("alpha", int16),
    ("Ipb", float64),
    ("u2ss", float64),
    ("theta0", theta0_type),
    ("x0", x0_type),
    ("tsteps", int16),
    ("n_u", int16),
    ("u", float64[:,:]),
]


@jitclass_(JITCLASS_SPEC)
class SingleMealT1DModel:
    """Physiological model of glucose-insulin dynamics for a type 1 diabetic patient with a single meal.

    Insulin pharmacokinetics follow a two-compartment subcutaneous absorption model.
    Glucose risk is captured via a non-symmetric risk function that penalises
    hypoglycaemia more than hyperglycaemia.

    State variables (all stored as arrays indexed by minute):
        G:        Plasma glucose concentration (mg/dL).
        X:        Remote insulin action (1/min).
        IG:       Interstitial glucose concentration (mg/dL).
        Qsto1:  Carbohydrate in the stomach solid phase (mg).
        Qsto2:  Carbohydrate in the stomach liquid phase (mg).
        Qgut:   Carbohydrate in the intestine (mg).
        Isc1:     Subcutaneous insulin in compartment 1 (pmol/kg).
        Isc2:     Subcutaneous insulin in compartment 2 (pmol/kg).
        Ip:       Plasma insulin (pmol/kg).

    Attributes:
        u2ss: Steady-state basal insulin infusion rate (pmol/kg/min).
        tsteps: Length of the simulation in minutes.
        n_u: Number of input channels (3: 1 meal, 2 insulin).
        Gb: Basal plasma glucose concentration (mg/dL).
        SG: Glucose effectiveness (1/min).
        SI: Insulin sensitivity (dL/kg/pmol/min).
        p2: Rate constant of the remote insulin compartment (1/min).
        r1: Risk function scaling parameter.
        r2: Risk function shape parameter.
        ka2: Absorption rate from subcutaneous compartment 2 to plasma (1/min).
        kd: Transfer rate from subcutaneous compartment 1 to 2 (1/min).
        ke: Elimination rate of plasma insulin (1/min).
        tau: Insulin absorption delay (minutes, integer).
        kempt: Gastric emptying rate constant shared across all meal slots (1/min).
        kabs: Intestinal absorption rate (1/min).
        beta: Meal announcement delay (minutes, integer).
        f: Carbohydrate bioavailability fraction (dimensionless).
        VG: Glucose distribution volume (dL/kg).
        VI: Insulin distribution volume (L/kg).
        alpha: Time constant of the interstitial glucose filter (minutes, integer).
        Ipb: Basal plasma insulin concentration (pmol/kg).
    """

    def __init__(self,
                 u2ss,
                 theta0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 x0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 tsteps=1440,
                 ):
        """Initialize the model and allocate state arrays.

        Args:
            u2ss: Steady-state basal insulin infusion rate (pmol/kg/min). Used
                to compute the basal plasma insulin ``Ipb`` and the default
                initial conditions for the insulin compartments.
            theta0: Optional dictionary of named parameter overrides. Any key
                not present falls back to the physiological default defined in
                ``reset()``.
            x0: Optional dictionary of named initial-condition overrides. Any
                key not present falls back to the steady-state default.
            tsteps: Length of the simulation in minutes. Determines the size of
                all state arrays.

        Returns:
            None
        """
        self.u2ss = np.float64(u2ss)
        self.x0 = x0 # TODO: check if this is the right way to do it
        self.tsteps = tsteps
        self.n_u = 3
        self.reset(theta0)


    def reset(self,
              theta0=Dict.empty(key_type=types.unicode_type, value_type=float64)
              ):
        """Reset all model parameters and re-initialise state arrays.

        Parameters are loaded from ``theta0`` when present; otherwise
        population-average defaults are used. Initial conditions are loaded
        from ``self.x0`` when present; otherwise steady-state values derived
        from ``u2ss`` are used. All state arrays are re-allocated and set to
        their initial conditions.

        Args:
            theta0: Dictionary of named parameter values to apply. Keys absent
                from the dictionary fall back to physiological defaults.

        Returns:
            None
        """
        # --- Structural / volume parameters ---
        self.f = theta0["f"] if "f" in theta0 else np.float64(0.9)
        self.VG = theta0["VG"] if "VG" in theta0 else np.float64(1.45)
        self.VI = theta0["VI"] if "VI" in theta0 else np.float64(0.135)
        # Alpha governs the lag between plasma and interstitial glucose
        self.alpha = np.int16(theta0["alpha"]) if "alpha" in theta0 else np.int16(7)

        # --- Insulin sensitivity ---
        self.SI = theta0["SI"] if "SI" in theta0 else np.float64(10.35e-4 / self.VG)

        # --- Glucose kinetics ---
        self.SG = theta0["SG"] if "SG" in theta0 else np.float64(2.5e-2)
        self.Gb = theta0["Gb"] if "Gb" in theta0 else np.float64(119.13)
        # p2: rate constant of the remote insulin compartment
        self.p2 = theta0["p2"] if "p2" in theta0 else np.float64(0.012)
        # Risk function parameters (non-symmetric hypoglycaemia penalty)
        self.r1 = theta0["r1"] if "r1" in theta0 else np.float64(1.4407)
        self.r2 = theta0["r2"] if "r2" in theta0 else np.float64(0.8124)

        # --- Insulin pharmacokinetics ---
        self.ka2 = theta0["ka2"] if "ka2" in theta0 else np.float64(0.014)
        self.kd = theta0["kd"] if "kd" in theta0 else np.float64(0.026)
        self.ke = theta0["ke"] if "ke" in theta0 else np.float64(0.127)
        # tau: integer delay (minutes) applied to the subcutaneous insulin input
        self.tau = np.int16(theta0["tau"]) if "tau" in theta0 else np.int16(8)

        # --- Meal gut-absorption rates ---
        self.kabs = theta0["kabs"] if "kabs" in theta0 else np.float64(0.012)
        # Shared gastric emptying rate constant across all meal slots
        self.kempt = theta0["kempt"] if "kempt" in theta0 else np.float64(0.18)

        # --- Meal announcement delays (integer minutes) ---
        self.beta = np.int16(theta0["beta"]) if "beta" in theta0 else np.int16(0)

        # --- Initial conditions (fall back to steady state if not provided) ---
        self._G0 = self.x0["G0"] if "G0" in self.x0 else np.float64(self.Gb)
        self._X0 = self.x0["X0"] if "X0" in self.x0 else np.float64(0)
        # Gut compartments start empty (no meal in progress at t=0)
        self._Qsto1_0 = self.x0["Qsto1_0"] if "Qsto1_0" in self.x0 else np.float64(0)
        self._Qsto2_0 = self.x0["Qsto2_0"] if "Qsto2_0" in self.x0 else np.float64(0)
        self._Qgut_0 = self.x0["Qgut_0"] if "Qgut_0" in self.x0 else np.float64(0)

        # Insulin compartments start at the basal steady state derived from u2ss
        ki1 = self.u2ss / self.kd
        ki2 = self.kd / self.ka2 * ki1
        self.Ipb = self.ka2 / self.ke * ki2  # basal plasma insulin

        self._Isc10 = self.x0["Isc10"] if "Isc10" in self.x0 else np.float64(ki1)
        self._Isc20 = self.x0["Isc20"] if "Isc20" in self.x0 else np.float64(ki2)
        self._Ip0 = self.x0["Ip0"] if "Ip0" in self.x0 else np.float64(self.Ipb)
        self._IG0 = self.x0["IG0"] if "IG0" in self.x0 else self.Gb

        # --- Allocate state arrays and set t=0 values ---
        self.G = np.empty((self.tsteps,), dtype=np.float64)
        self.G[0] = self._G0
        self.X = np.empty((self.tsteps,), dtype=np.float64)
        self.X[0] = self._X0

        self.Qsto1 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1[0] = self._Qsto1_0
        self.Qsto2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2[0] = self._Qsto2_0
        self.Qgut = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut[0] = self._Qgut_0

        self.Isc1 = np.empty((self.tsteps,), dtype=np.float64)
        self.Isc1[0] = self._Isc10
        self.Isc2 = np.empty((self.tsteps,), dtype=np.float64)
        self.Isc2[0] = self._Isc20
        self.Ip = np.empty((self.tsteps,), dtype=np.float64)
        self.Ip[0] = self._Ip0
        self.IG = np.empty((self.tsteps,), dtype=np.float64)
        self.IG[0] = self._IG0

        # Input buffer used by step() to retrieve delayed input values
        self.u = np.zeros((self.n_u, self.tsteps), dtype=np.float64)
        self.u[1, 0] = self.u2ss
        self.u[2, 0] = 0.0  # no bolus at t=0

    def step(self, u: float64[:], t: float64):
        """Advance the model by one minute using Backward Euler integration.

        This is the primary integration method used during twinning and replay.
        The full input history is stored in ``self.u`` to support integer-minute
        meal announcement delays (``beta_*``) and insulin absorption delay
        (``tau``). The Backward Euler discretisation is applied sequentially
        along the physiological cascade (Gauss-Seidel order), which is
        equivalent to a simultaneous implicit solve for this DAG-structured
        system. The nonlinear risk term is treated semi-implicitly — its
        coefficient is evaluated at ``G[t-1]`` to avoid a nonlinear solve while
        preserving first-order accuracy.

        Args:
            u: Input vector of length ``n_u`` containing:
                - u[0]: Carbohydrate rate (mg/kg/min).
                - u[1]: Basal insulin infusion rate (pmol/kg/min).
                - u[2]: Bolus insulin rate (pmol/kg/min).
            t: Current simulation time step (integer minute index, >= 1).

        Returns:
            None
        """
        # Store current inputs in the history buffer so delayed values can be
        # retrieved at this or future timesteps via index arithmetic.
        self.u[0, t] = u[0]
        # Apply meal announcement delay beta: use the carb input from beta
        # minutes ago. If we are still within the first beta minutes of the
        # simulation (t < beta), treat the delayed input as zero.
        u_m = self.u[0, t - self.beta] if (t - self.beta) >= 0 else 0
        self.u[1, t] = u[1]
        self.u[2, t] = u[2]
        # Apply insulin absorption delay tau: use the insulin input from tau
        # minutes ago. Before tau minutes have elapsed, fall back to u2ss so
        # the plasma insulin chain starts at its basal steady state.
        u_i = self.u[1, t - self.tau] + self.u[2, t - self.tau] if (t - self.tau) >= 0 else self.u2ss

        # Non-symmetric risk function evaluated at the previous glucose value
        # (semi-implicit treatment of the nonlinear term to avoid a nonlinear
        # solve). Risk is zero above Gb, increasing below it, and constant
        # below 60 mg/dL.
        g_prev = self.G[t-1]
        logGb = np.log(self.Gb)
        log60 = np.log(60.0)
        if (g_prev < self.Gb) and (g_prev >= 60.0):
            lg = np.log(g_prev)
            diff = lg ** self.r2 - logGb ** self.r2
            risk = 1.0 + 10 * self.r1 * diff * diff
        elif g_prev < 60.0:
            diff = log60 ** self.r2 - logGb ** self.r2  # constant below 60 mg/dL
            risk = 1.0 + 10 * self.r1 * diff * diff
        else:
            risk = 1.0

        # Pre-compute implicit denominators.
        # For dQ/dt = -k*Q + input the BE update is Q[t] = (Q[t-1] + input) / (1 + k).
        k1 = 1.0 / (1.0 + self.kempt)   # Qsto1: -kempt * Qsto1
        k2 = 1.0 / (1.0 + self.kempt)   # Qsto2: -kempt * Qsto2
        kd_fac = 1.0 / (1.0 + self.kd)  # Isc1:  -kd * Isc1

        # --- Gut absorption chains (Backward Euler, Gauss-Seidel order) ---
        # Each meal slot: solid stomach (Qsto1) → liquid stomach (Qsto2) → intestine (Qgut).
        # Each compartment uses the current-step value of its upstream input,
        # which is valid because the system is a strict cascade (no feedback).
        self.Qsto1[t] = (self.Qsto1[t-1] + u_m) * k1
        self.Qsto2[t] = (self.Qsto2[t-1] + self.kempt * self.Qsto1[t]) * k2
        self.Qgut[t] = (self.Qgut[t-1] + self.kempt * self.Qsto2[t]) / (1 + self.kabs)

        # --- Insulin pharmacokinetic chain (Backward Euler, Gauss-Seidel order) ---
        # Subcutaneous compartment 1 → compartment 2 → plasma
        self.Isc1[t] = (self.Isc1[t-1] + u_i) * kd_fac
        self.Isc2[t] = (self.Isc2[t-1] + self.kd * self.Isc1[t]) / (1 + self.ka2)
        self.Ip[t] = (self.Ip[t-1] + self.ka2 * self.Isc2[t]) / (1 + self.ke)

        # --- Insulin action and glucose (Backward Euler) ---
        # X depends on Ip[t] (just computed above)
        self.X[t] = (self.X[t-1] + self.p2 * (self.SI / self.VI) * (self.Ip[t] - self.Ipb)) / (1 + self.p2)
        # G depends on X[t] and all Qgut[t] (just computed). The risk coefficient
        # is frozen at G[t-1] (semi-implicit) to keep the update linear in G[t].
        self.G[t] = (self.G[t-1] + self.SG * self.Gb + self.f * self.kabs * self.Qgut[t] / self.VG) / (1 + self.SG + risk * self.X[t])
        # Interstitial glucose: first-order low-pass filter on plasma glucose
        self.IG[t] = (self.alpha * self.IG[t-1] + self.G[t]) / (1 + self.alpha)

    def output(self, t: float64):
        """Return the model output at time step ``t``.

        The observable output is the interstitial glucose concentration, which
        corresponds to the signal measured by a continuous glucose monitor.

        Args:
            t: Time step index (integer minute).

        Returns:
            float: Interstitial glucose concentration at time ``t`` (mg/dL).
        """
        return self.IG[t]