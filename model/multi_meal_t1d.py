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
    ("n_u", int16),
    ("u", float64[:,:]),
]


@jitclass_(JITCLASS_SPEC)
class MultiMealT1DModel:
    """Physiological model of glucose-insulin dynamics for a type 1 diabetic patient with multiple daily meals.

    The model extends the Hovorka / UVA-Padova single-meal structure to five
    separate meal slots (Breakfast, Lunch, Dinner, Snack, Hypo-treatment), each
    with its own gut-absorption chain and absorption rate constant. Insulin
    pharmacokinetics follow a two-compartment subcutaneous absorption model.
    Glucose risk is captured via a non-symmetric risk function that penalises
    hypoglycaemia more than hyperglycaemia.

    State variables (all stored as arrays indexed by minute):
        G:        Plasma glucose concentration (mg/dL).
        X:        Remote insulin action (1/min).
        IG:       Interstitial glucose concentration (mg/dL).
        Qsto1_*:  Carbohydrate in the stomach solid phase for each meal slot (mg).
        Qsto2_*:  Carbohydrate in the stomach liquid phase for each meal slot (mg).
        Qgut_*:   Carbohydrate in the intestine for each meal slot (mg).
        Isc1:     Subcutaneous insulin in compartment 1 (pmol/kg).
        Isc2:     Subcutaneous insulin in compartment 2 (pmol/kg).
        Ip:       Plasma insulin (pmol/kg).

    Attributes:
        u2ss: Steady-state basal insulin infusion rate (pmol/kg/min).
        tsteps: Length of the simulation in minutes.
        n_u: Number of input channels (8: 5 meal slots, 2 insulin channels, 1 hour-of-day).
        Gb: Basal plasma glucose concentration (mg/dL).
        SG: Glucose effectiveness (1/min).
        SI_B: Insulin sensitivity during breakfast hours (dL/kg/pmol/min).
        SI_L: Insulin sensitivity during lunch hours (dL/kg/pmol/min).
        SI_D: Insulin sensitivity during dinner/night hours (dL/kg/pmol/min).
        p2: Rate constant of the remote insulin compartment (1/min).
        r1: Risk function scaling parameter.
        r2: Risk function shape parameter.
        ka2: Absorption rate from subcutaneous compartment 2 to plasma (1/min).
        kd: Transfer rate from subcutaneous compartment 1 to 2 (1/min).
        ke: Elimination rate of plasma insulin (1/min).
        tau: Insulin absorption delay (minutes, integer).
        kempt: Gastric emptying rate constant shared across all meal slots (1/min).
        kabs_B: Intestinal absorption rate for breakfast (1/min).
        kabs_L: Intestinal absorption rate for lunch (1/min).
        kabs_D: Intestinal absorption rate for dinner (1/min).
        kabs_S: Intestinal absorption rate for snack (1/min).
        kabs_H: Intestinal absorption rate for hypo-treatment (1/min).
        beta_B: Meal announcement delay for breakfast (minutes, integer).
        beta_L: Meal announcement delay for lunch (minutes, integer).
        beta_D: Meal announcement delay for dinner (minutes, integer).
        beta_S: Meal announcement delay for snack (minutes, integer).
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
        self.n_u = 8
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
        self._X0 = self.x0["X0"] if "X0" in self.x0 else np.float64(0)
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
                - u[0]: Breakfast carbohydrate rate (mg/kg/min).
                - u[1]: Lunch carbohydrate rate (mg/kg/min).
                - u[2]: Dinner carbohydrate rate (mg/kg/min).
                - u[3]: Snack carbohydrate rate (mg/kg/min).
                - u[4]: Hypo-treatment carbohydrate rate (mg/kg/min).
                - u[5]: Basal insulin infusion rate (pmol/kg/min).
                - u[6]: Bolus insulin rate (pmol/kg/min).
                - u[7]: Hour of day (0–23), used to select insulin sensitivity.
            t: Current simulation time step (integer minute index, >= 1).

        Returns:
            None
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
        self.Ip[t] = (self.Ip[t-1] + self.ka2 * self.Isc2[t]) / (1 + self.ke)

        # --- Insulin action and glucose (Backward Euler) ---
        # X depends on Ip[t] (just computed above)
        self.X[t] = (self.X[t-1] + self.p2 * (SI / self.VI) * (self.Ip[t] - self.Ipb)) / (1 + self.p2)
        # G depends on X[t] and all Qgut[t] (just computed). The risk coefficient
        # is frozen at G[t-1] (semi-implicit) to keep the update linear in G[t].
        self.G[t] = (self.G[t-1] + self.SG * self.Gb + self.f * (
                self.kabs_B * self.Qgut_B[t] + self.kabs_L * self.Qgut_L[t] + self.kabs_D * self.Qgut_D[t] +
                self.kabs_S * self.Qgut_S[t] + self.kabs_H * self.Qgut_H[t]) / self.VG) / (1 + self.SG + risk * self.X[t])
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