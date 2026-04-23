import numpy as np
from numba import float64, int16, int64
from numba.typed import Dict
from numba import types

from utils.numba_dicts import to_typed_f32_dict

from environment.config import jitclass_

theta0_type = types.DictType(types.unicode_type, float64)
x0_type = types.DictType(types.unicode_type, float64)
JITCLASS_SPEC = [
    ("_kd_prev", float64),
    ("_ka2_prev", float64),
    ("_SI_last_prev", float64),
    ("_VI_prev", float64),
    ("_G0", float64),
    ("_X0", float64),
    ("_Qsto1_B_0", float64),
    ("_Qsto2_B_0", float64),
    ("_Qgut_B_0", float64),
    ("_Qsto1_L_0", float64),
    ("_Qsto2_L_0", float64),
    ("_Qgut_L_0", float64),
    ("_Qsto1_D_0", float64),
    ("_Qsto2_D_0", float64),
    ("_Qgut_D_0", float64),
    ("_Qsto1_S_0", float64),
    ("_Qsto2_S_0", float64),
    ("_Qgut_S_0", float64),
    ("_Qsto1_H_0", float64),
    ("_Qsto2_H_0", float64),
    ("_Qgut_H_0", float64),
    ("_Isc10", float64),
    ("_Isc20", float64),
    ("_Ip0", float64),
    ("_IG0", float64),
    ("G", float64[:]),
    ("X", float64[:]),
    ("Qsto1_B", float64[:]),
    ("Qsto2_B", float64[:]),
    ("Qgut_B", float64[:]),
    ("Qsto1_L", float64[:]),
    ("Qsto2_L", float64[:]),
    ("Qgut_L", float64[:]),
    ("Qsto1_D", float64[:]),
    ("Qsto2_D", float64[:]),
    ("Qgut_D", float64[:]),
    ("Qsto1_S", float64[:]),
    ("Qsto2_S", float64[:]),
    ("Qgut_S", float64[:]),
    ("Qsto1_H", float64[:]),
    ("Qsto2_H", float64[:]),
    ("Qgut_H", float64[:]),
    ("Isc1", float64[:]),
    ("Isc2", float64[:]),
    ("Ip", float64[:]),
    ("IG", float64[:]),
    ("SI_B", float64),
    ("SI_L", float64),
    ("SI_D", float64),
    ("SG", float64),
    ("Gb", float64),
    ("p2", float64),
    ("r1", float64),
    ("r2", float64),
    ("tau", int16),
    ("ka2", float64),
    ("kd", float64),
    ("ke", float64),
    ("kabs_B", float64),
    ("kabs_L", float64),
    ("kabs_D", float64),
    ("kabs_S", float64),
    ("kabs_H", float64),
    ("beta_B", int16),
    ("beta_L", int16),
    ("beta_D", int16),
    ("beta_S", int16),
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
    ("t_start", int16),
    ("n_u", int16),
    ("u", float64[:,:]),
]


@jitclass_(JITCLASS_SPEC)
class MultiMealT1DModel:
    """Physiological model of glucose-insulin dynamics for a type 1 diabetic patient with multiple daily meals.
    """

    def __init__(self,
                 u2ss,
                 theta0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 x0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 tsteps=1440,
                 t_start=240,
                 ):
        """Initialize the model and allocate state arrays.
        """
        self._kd_prev  = np.float64(0.026)
        self._ka2_prev = np.float64(0.014)
        self._VI_prev  = np.float64(0.135)
        self._SI_last_prev = np.float64(10.35e-4 / 1.45) # 1.45 instead of dynamic VG

        self.theta0 = theta0
        self.x0 = x0
        self.tsteps = tsteps

        self.t_start = np.int16(t_start)

        self.n_u = 10

        self.u2ss = u2ss

        self.reset(theta0)

    def reset(self,
              theta0=Dict.empty(key_type=types.unicode_type, value_type=float64)
              ):
        """Reset all model parameters and re-initialise state arrays.
        """
        # --- Structural / volume parameters ---
        self.f = theta0["f"] if "f" in theta0 else np.float64(0.9)
        self.VG = theta0["VG"] if "VG" in theta0 else np.float64(1.45)
        self.VI = theta0["VI"] if "VI" in theta0 else np.float64(0.135)
        # Alpha governs the lag between plasma and interstitial glucose
        self.alpha = np.int16(theta0["alpha"]) if "alpha" in theta0 else np.int16(7)

        # --- Insulin sensitivity (time-of-day dependent) ---
        self.SI_B = theta0["SI_B"] if "SI_B" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SI_L = theta0["SI_L"] if "SI_L" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SI_D = theta0["SI_D"] if "SI_D" in theta0 else np.float64(10.35e-4 / self.VG)

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
        self.kabs_B = theta0["kabs_B"] if "kabs_B" in theta0 else np.float64(0.012)
        self.kabs_L = theta0["kabs_L"] if "kabs_L" in theta0 else np.float64(0.012)
        self.kabs_D = theta0["kabs_D"] if "kabs_D" in theta0 else np.float64(0.012)
        self.kabs_S = theta0["kabs_S"] if "kabs_S" in theta0 else np.float64(0.012)
        self.kabs_H = theta0["kabs_H"] if "kabs_H" in theta0 else np.float64(0.012)
        # Shared gastric emptying rate constant across all meal slots
        self.kempt = theta0["kempt"] if "kempt" in theta0 else np.float64(0.18)

        # --- Meal announcement delays (integer minutes) ---
        self.beta_B = np.int16(theta0["beta_B"]) if "beta_B" in theta0 else np.int16(0)
        self.beta_L = np.int16(theta0["beta_L"]) if "beta_L" in theta0 else np.int16(0)
        self.beta_D = np.int16(theta0["beta_D"]) if "beta_D" in theta0 else np.int16(0)
        self.beta_S = np.int16(theta0["beta_S"]) if "beta_S" in theta0 else np.int16(0)

        # --- Initial conditions (fall back to steady state if not provided) ---
        self._G0 = self.x0["G0"] if "G0" in self.x0 else np.float64(self.Gb)

        # X = (SI/VI)*(Ip - Ipb): when SI or VI change across segments, X0 must be
        # rescaled by the ratio of (SI_first/VI) for the current segment to
        # (SI_last/VI) for the previous segment, where SI_last is the sensitivity
        # active during the last minute of the previous segment (hour t_start-1).
        h_first = int(self.t_start // 60) % 24
        if h_first < 4 or h_first >= 17:
            SI_first = self.SI_D
        elif h_first < 11:
            SI_first = self.SI_B
        else:
            SI_first = self.SI_L
        if "X0" in self.x0:
            self._X0 = (SI_first / self.VI) / (self._SI_last_prev / self._VI_prev) * self.x0["X0"]
        else:
            self._X0 = np.float64(0)

        # Gut compartments start empty (no meal in progress at t=0)
        self._Qsto1_B_0 = self.x0["Qsto1_B_0"] if "Qsto1_B_0" in self.x0 else np.float64(0)
        self._Qsto2_B_0 = self.x0["Qsto2_B_0"] if "Qsto2_B_0" in self.x0 else np.float64(0)
        self._Qgut_B_0 = self.x0["Qgut_B_0"] if "Qgut_B_0" in self.x0 else np.float64(0)
        self._Qsto1_L_0 = self.x0["Qsto1_L_0"] if "Qsto1_L_0" in self.x0 else np.float64(0)
        self._Qsto2_L_0 = self.x0["Qsto2_L_0"] if "Qsto2_L_0" in self.x0 else np.float64(0)
        self._Qgut_L_0 = self.x0["Qgut_L_0"] if "Qgut_L_0" in self.x0 else np.float64(0)
        self._Qsto1_D_0 = self.x0["Qsto1_D_0"] if "Qsto1_D_0" in self.x0 else np.float64(0)
        self._Qsto2_D_0 = self.x0["Qsto2_D_0"] if "Qsto2_D_0" in self.x0 else np.float64(0)
        self._Qgut_D_0 = self.x0["Qgut_D_0"] if "Qgut_D_0" in self.x0 else np.float64(0)
        self._Qsto1_S_0 = self.x0["Qsto1_S_0"] if "Qsto1_S_0" in self.x0 else np.float64(0)
        self._Qsto2_S_0 = self.x0["Qsto2_S_0"] if "Qsto2_S_0" in self.x0 else np.float64(0)
        self._Qgut_S_0 = self.x0["Qgut_S_0"] if "Qgut_S_0" in self.x0 else np.float64(0)
        self._Qsto1_H_0 = self.x0["Qsto1_H_0"] if "Qsto1_H_0" in self.x0 else np.float64(0)
        self._Qsto2_H_0 = self.x0["Qsto2_H_0"] if "Qsto2_H_0" in self.x0 else np.float64(0)
        self._Qgut_H_0 = self.x0["Qgut_H_0"] if "Qgut_H_0" in self.x0 else np.float64(0)

        # Insulin compartments start at the basal steady state derived from u2ss.
        # When x0 carries values from the previous day, they are scaled by the ratio of
        # the current trial's ki1/ki2 to the previous day's ki1_prev/ki2_prev, mirroring
        # the py_replay_bg approach. When _kd_prev == kd (cold start default) the ratio
        # is 1 and behaviour is unchanged.
        ki1 = self.u2ss / self.kd
        ki2 = self.kd / self.ka2 * ki1
        self.Ipb = self.ka2 / self.ke * ki2  # basal plasma insulin

        ki1_prev = self.u2ss / self._kd_prev
        ki2_prev = self._kd_prev / self._ka2_prev * ki1_prev


        if "Isc10" in self.x0:
            self._Isc10 = ki1 / ki1_prev * self.x0["Isc10"]
        else:
            self._Isc10 = ki1
        if "Isc20" in self.x0:
            self._Isc20 = ki2 / ki2_prev * self.x0["Isc20"]
        else:
            self._Isc20 = ki2

        # Ipb = u2ss/ke regardless of kd/ka2 — no scaling needed
        self._Ip0 = self.x0["Ip0"] if "Ip0" in self.x0 else np.float64(self.Ipb)
        self._IG0 = self.x0["IG0"] if "IG0" in self.x0 else self.Gb

        # --- Allocate state arrays and set t=0 values ---
        self.G = np.empty((self.tsteps,), dtype=np.float64)
        self.G[0] = self._G0
        self.X = np.empty((self.tsteps,), dtype=np.float64)
        self.X[0] = self._X0

        self.Qsto1_B = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_B[0] = self._Qsto1_B_0
        self.Qsto2_B = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_B[0] = self._Qsto2_B_0
        self.Qgut_B = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_B[0] = self._Qgut_B_0

        self.Qsto1_L = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_L[0] = self._Qsto1_L_0
        self.Qsto2_L = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_L[0] = self._Qsto2_L_0
        self.Qgut_L = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_L[0] = self._Qgut_L_0

        self.Qsto1_D = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_D[0] = self._Qsto1_D_0
        self.Qsto2_D = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_D[0] = self._Qsto2_D_0
        self.Qgut_D = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_D[0] = self._Qgut_D_0

        self.Qsto1_S = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_S[0] = self._Qsto1_S_0
        self.Qsto2_S = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_S[0] = self._Qsto2_S_0
        self.Qgut_S = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_S[0] = self._Qgut_S_0

        self.Qsto1_H = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto1_H[0] = self._Qsto1_H_0
        self.Qsto2_H = np.empty((self.tsteps,), dtype=np.float64)
        self.Qsto2_H[0] = self._Qsto2_H_0
        self.Qgut_H = np.empty((self.tsteps,), dtype=np.float64)
        self.Qgut_H[0] = self._Qgut_H_0

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
        self.u[5, 0] = self.u2ss
        self.u[6, 0] = 0.0  # no bolus at t=0

    def step(self, u: float64[:], t: int64):
        """Advance the model by one minute using Backward Euler integration.
        """
        # Store current inputs in the history buffer so delayed values can be
        # retrieved at this or future timesteps via index arithmetic.
        self.u[0, t] = u[0]
        # Apply meal announcement delay beta_B: use the carb input from beta_B
        # minutes ago. If we are still within the first beta_B minutes of the
        # simulation (t < beta_B), treat the delayed input as zero.
        u_m_b = self.u[0, t - self.beta_B] if (t - self.beta_B) >= 0 else 0
        self.u[1, t] = u[1]
        u_m_l = self.u[1, t - self.beta_L] if (t - self.beta_L) >= 0 else 0
        self.u[2, t] = u[2]
        u_m_d = self.u[2, t - self.beta_D] if (t - self.beta_D) >= 0 else 0
        self.u[3, t] = u[3]
        u_m_s = self.u[3, t - self.beta_S] if (t - self.beta_S) >= 0 else 0
        self.u[4, t] = u[4]
        # Hypo-treatment has no announcement delay
        u_m_h = self.u[4, t]
        self.u[5, t] = u[5]
        self.u[6, t] = u[6]
        # Apply insulin absorption delay tau: use the insulin input from tau
        # minutes ago. Before tau minutes have elapsed, fall back to u2ss so
        # the plasma insulin chain starts at its basal steady state.
        u_i = self.u[5, t - self.tau] + self.u[6, t - self.tau] if (t - self.tau) >= 0 else self.u2ss
        self.u[7, t] = u[7]
        u_h = self.u[7, t]
        self.u[8, t] = u[8]
        forcing_ra = self.u[8, t]
        self.u[9, t] = u[9]
        forcing_ip = self.u[9, t]

        # Select insulin sensitivity based on hour of day:
        # breakfast window 04:00–10:59, lunch 11:00–16:59, dinner/night otherwise
        if u_h < 4 or u_h >= 17:
            SI = self.SI_D
        elif 4 <= u_h < 11:
            SI = self.SI_B
        else:
            SI = self.SI_L

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
        self.Qsto1_B[t] = (self.Qsto1_B[t-1] + u_m_b) * k1
        self.Qsto2_B[t] = (self.Qsto2_B[t-1] + self.kempt * self.Qsto1_B[t]) * k2
        self.Qgut_B[t] = (self.Qgut_B[t-1] + self.kempt * self.Qsto2_B[t]) / (1 + self.kabs_B)

        self.Qsto1_L[t] = (self.Qsto1_L[t-1] + u_m_l) * k1
        self.Qsto2_L[t] = (self.Qsto2_L[t-1] + self.kempt * self.Qsto1_L[t]) * k2
        self.Qgut_L[t] = (self.Qgut_L[t-1] + self.kempt * self.Qsto2_L[t]) / (1 + self.kabs_L)

        self.Qsto1_D[t] = (self.Qsto1_D[t-1] + u_m_d) * k1
        self.Qsto2_D[t] = (self.Qsto2_D[t-1] + self.kempt * self.Qsto1_D[t]) * k2
        self.Qgut_D[t] = (self.Qgut_D[t-1] + self.kempt * self.Qsto2_D[t]) / (1 + self.kabs_D)

        self.Qsto1_S[t] = (self.Qsto1_S[t-1] + u_m_s) * k1
        self.Qsto2_S[t] = (self.Qsto2_S[t-1] + self.kempt * self.Qsto1_S[t]) * k2
        self.Qgut_S[t] = (self.Qgut_S[t-1] + self.kempt * self.Qsto2_S[t]) / (1 + self.kabs_S)

        self.Qsto1_H[t] = (self.Qsto1_H[t-1] + u_m_h) * k1
        self.Qsto2_H[t] = (self.Qsto2_H[t-1] + self.kempt * self.Qsto1_H[t]) * k2
        self.Qgut_H[t] = (self.Qgut_H[t-1] + self.kempt * self.Qsto2_H[t]) / (1 + self.kabs_H)

        # --- Insulin pharmacokinetic chain (Backward Euler, Gauss-Seidel order) ---
        # Subcutaneous compartment 1 → compartment 2 → plasma
        self.Isc1[t] = (self.Isc1[t-1] + u_i) * kd_fac
        self.Isc2[t] = (self.Isc2[t-1] + self.kd * self.Isc1[t]) / (1 + self.ka2)
        self.Ip[t] = (self.Ip[t-1] + self.ka2 * self.Isc2[t] + forcing_ip) / (1 + self.ke)

        # --- Insulin action and glucose (Backward Euler) ---
        # X depends on Ip[t] (just computed above)
        self.X[t] = (self.X[t-1] + self.p2 * (SI / self.VI) * (self.Ip[t] - self.Ipb)) / (1 + self.p2)
        # G depends on X[t] and all Qgut[t] (just computed). The risk coefficient
        # is frozen at G[t-1] (semi-implicit) to keep the update linear in G[t].
        self.G[t] = (self.G[t-1] + self.SG * self.Gb + self.f * (
                self.kabs_B * self.Qgut_B[t] + self.kabs_L * self.Qgut_L[t] + self.kabs_D * self.Qgut_D[t] +
                self.kabs_S * self.Qgut_S[t] + self.kabs_H * self.Qgut_H[t]) / self.VG + forcing_ra / self.VG) / (1 + self.SG + risk * self.X[t])
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


def _setup_x0(x0, previous_theta=None):
    """Factory returning a ``(model, data) -> None`` callable for ``ReplayBG.twin(x0_setup=...)``.

    Computes the carry-over glucose appearance rate (``previous_Ra``) from the
    meal gut-compartment values stored in *x0*, injects it into
    ``data.forcing_ra`` / ``data.u[:, 8]``, resets the meal compartments in *x0*
    to zero, and stores the previous-day insulin kinetic parameters on the model
    so that :meth:`MultiMealT1DModel.reset` can correctly scale the ``Isc1``/
    ``Isc2`` initial conditions on every optimizer trial.

    This function is defined outside the ``@jitclass_``-decorated class body so
    that Numba never tries to compile it (it must accept a plain Python
    ``MultiMealT1DData`` object).  All spec'd model attributes and methods are
    accessible from Python on a compiled jitclass instance.

    Args:
        x0: Numba typed dict of initial conditions taken from the end of the
            previous simulation (keys match ``MultiMealT1DModel`` attribute
            names, e.g. ``"G0"``, ``"Isc10"``, ``"Qsto1_B_0"``).
        previous_theta: Plain Python dict of the previous day's estimated
            parameters.  Keys used: ``'kempt'``, ``'kabs_B'``, ``'kabs_L'``,
            ``'kabs_D'``, ``'kabs_S'``, ``'kabs_H'``, ``'f'``, ``'kd'``,
            ``'ka2'``.  Missing keys fall back to population defaults.

    Returns:
        Callable[(model, data) -> None]: ready to pass as the ``x0_setup``
        argument of :meth:`ReplayBG.twin`.
    """
    def setup(model, data):
        pt = previous_theta or {}

        kempt  = pt.get('kempt',  0.18)
        kabs_B = pt.get('kabs_B', 0.012)
        kabs_L = pt.get('kabs_L', 0.012)
        kabs_D = pt.get('kabs_D', 0.012)
        kabs_S = pt.get('kabs_S', 0.012)
        kabs_H = pt.get('kabs_H', 0.012)
        f      = pt.get('f',      0.9)

        # Free Backward-Euler evolution of meal gut compartments (no new meal input).
        # Layout: [Qsto1_B, Qsto2_B, Qgut_B, Qsto1_L, ..., Qgut_H]
        xk = [
            float(x0.get("Qsto1_B_0", 0.0)), float(x0.get("Qsto2_B_0", 0.0)), float(x0.get("Qgut_B_0", 0.0)),
            float(x0.get("Qsto1_L_0", 0.0)), float(x0.get("Qsto2_L_0", 0.0)), float(x0.get("Qgut_L_0", 0.0)),
            float(x0.get("Qsto1_D_0", 0.0)), float(x0.get("Qsto2_D_0", 0.0)), float(x0.get("Qgut_D_0", 0.0)),
            float(x0.get("Qsto1_S_0", 0.0)), float(x0.get("Qsto2_S_0", 0.0)), float(x0.get("Qgut_S_0", 0.0)),
            float(x0.get("Qsto1_H_0", 0.0)), float(x0.get("Qsto2_H_0", 0.0)), float(x0.get("Qgut_H_0", 0.0)),
        ]
        previous_Ra = np.zeros(data.tsteps)
        for k in range(data.tsteps):
            xk[0]  = xk[0]  / (1 + kempt)
            xk[1]  = (xk[1]  + kempt * xk[0])  / (1 + kempt)
            xk[2]  = (xk[2]  + kempt * xk[1])  / (1 + kabs_B)
            xk[3]  = xk[3]  / (1 + kempt)
            xk[4]  = (xk[4]  + kempt * xk[3])  / (1 + kempt)
            xk[5]  = (xk[5]  + kempt * xk[4])  / (1 + kabs_L)
            xk[6]  = xk[6]  / (1 + kempt)
            xk[7]  = (xk[7]  + kempt * xk[6])  / (1 + kempt)
            xk[8]  = (xk[8]  + kempt * xk[7])  / (1 + kabs_D)
            xk[9]  = xk[9]  / (1 + kempt)
            xk[10] = (xk[10] + kempt * xk[9])  / (1 + kempt)
            xk[11] = (xk[11] + kempt * xk[10]) / (1 + kabs_S)
            xk[12] = xk[12] / (1 + kempt)
            xk[13] = (xk[13] + kempt * xk[12]) / (1 + kempt)
            xk[14] = (xk[14] + kempt * xk[13]) / (1 + kabs_H)
            previous_Ra[k] = f * (
                kabs_B * xk[2]  + kabs_L * xk[5]  + kabs_D * xk[8] +
                kabs_S * xk[11] + kabs_H * xk[14]
            )

        # Inject carry-over Ra into data (data.u channel 8 maps to forcing_ra).
        data.forcing_ra = previous_Ra
        data.u[:, 8]    = previous_Ra

        # Reset meal compartments in x0 — carry-over is now handled via forcing_ra.
        for key in [
            "Qsto1_B_0", "Qsto2_B_0", "Qgut_B_0",
            "Qsto1_L_0", "Qsto2_L_0", "Qgut_L_0",
            "Qsto1_D_0", "Qsto2_D_0", "Qgut_D_0",
            "Qsto1_S_0", "Qsto2_S_0", "Qgut_S_0",
            "Qsto1_H_0", "Qsto2_H_0", "Qgut_H_0",
        ]:
            x0[key] = np.float64(0.0)

        # Store previous-day insulin params so reset() can scale Isc1/Isc2 and X.
        model._kd_prev  = np.float64(pt.get('kd',  0.026))
        model._ka2_prev = np.float64(pt.get('ka2', 0.014))
        model._VI_prev = np.float64(pt.get('VI', 0.135))

        # SI active during the last minute of the previous segment is SI at
        # hour (t_start - 1) % 24, using the time-of-day windows from step().
        h_last = ((int(model.t_start) - 1) // 60) % 24
        vg_prev = pt.get('VG', 1.45)
        si_default = 10.35e-4 / vg_prev
        if h_last < 4 or h_last >= 17:
            model._SI_last_prev = np.float64(pt.get('SI_D', si_default))
        elif h_last < 11:
            model._SI_last_prev = np.float64(pt.get('SI_B', si_default))
        else:
            model._SI_last_prev = np.float64(pt.get('SI_L', si_default))

        # Apply updated x0 and re-initialise state arrays, preserving fitted theta.
        model.x0 = x0
        model.reset(model.theta0)

    return setup


MultiMealT1DModel.setup_x0 = staticmethod(_setup_x0)


def _extract_final_x0(model):
    """Return the final state of *model* as a typed dict suitable for ``setup_x0``.

    Reads the last time-step of every state array and packs the values into a
    plain Python dict, then converts it to a Numba-typed dict via
    ``to_typed_f32_dict``.  The returned dict can be passed directly as the
    ``x0`` argument of ``MultiMealT1DModel.setup_x0``.

    Args:
        model: A ``MultiMealT1DModel`` instance whose ``step()`` loop has
            already been run to completion.

    Returns:
        Numba typed dict[str, float64] with keys matching the ``x0`` contract
        expected by ``_setup_x0``.
    """


    state = {
        "G0":        float(model.G[-1]),
        "X0":        float(model.X[-1]),
        "IG0":       float(model.IG[-1]),
        "Isc10":     float(model.Isc1[-1]),
        "Isc20":     float(model.Isc2[-1]),
        "Ip0":       float(model.Ip[-1]),
        "Qsto1_B_0": float(model.Qsto1_B[-1]),
        "Qsto2_B_0": float(model.Qsto2_B[-1]),
        "Qgut_B_0":  float(model.Qgut_B[-1]),
        "Qsto1_L_0": float(model.Qsto1_L[-1]),
        "Qsto2_L_0": float(model.Qsto2_L[-1]),
        "Qgut_L_0":  float(model.Qgut_L[-1]),
        "Qsto1_D_0": float(model.Qsto1_D[-1]),
        "Qsto2_D_0": float(model.Qsto2_D[-1]),
        "Qgut_D_0":  float(model.Qgut_D[-1]),
        "Qsto1_S_0": float(model.Qsto1_S[-1]),
        "Qsto2_S_0": float(model.Qsto2_S[-1]),
        "Qgut_S_0":  float(model.Qgut_S[-1]),
        "Qsto1_H_0": float(model.Qsto1_H[-1]),
        "Qsto2_H_0": float(model.Qsto2_H[-1]),
        "Qgut_H_0":  float(model.Qgut_H[-1]),
    }
    return to_typed_f32_dict(state)

MultiMealT1DModel.extract_final_x0 = staticmethod(_extract_final_x0)