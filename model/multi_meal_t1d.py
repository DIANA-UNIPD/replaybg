import numpy as np
from numba import float64
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
    ("G", float64),
    ("X", float64),
    ("Qsto1_B", float64),
    ("Qsto2_B", float64),
    ("Qgut_B", float64),
    ("Qsto1_L", float64),
    ("Qsto2_L", float64),
    ("Qgut_L", float64),
    ("Qsto1_D", float64),
    ("Qsto2_D", float64),
    ("Qgut_D", float64),
    ("Qsto1_S", float64),
    ("Qsto2_S", float64),
    ("Qgut_S", float64),
    ("Qsto1_H", float64),
    ("Qsto2_H", float64),
    ("Qgut_H", float64),
    ("Isc1", float64),
    ("Isc2", float64),
    ("Ip", float64),
    ("IG", float64),
    ("SI_B", float64),
    ("SI_L", float64),
    ("SI_D", float64),
    ("SG", float64),
    ("Gb", float64),
    ("p2", float64),
    ("r1", float64),
    ("r2", float64),
    ("tau", float64),
    ("ka2", float64),
    ("kd", float64),
    ("ke", float64),
    ("kabs_B", float64),
    ("kabs_L", float64),
    ("kabs_D", float64),
    ("kabs_S", float64),
    ("kabs_H", float64),
    ("beta_B", float64),
    ("beta_L", float64),
    ("beta_D", float64),
    ("beta_S", float64),
    ("kempt", float64),
    ("f", float64),
    ("VG", float64),
    ("VI", float64),
    ("alpha", float64),
    ("Ipb", float64),
    ("u2ss", float64),
    ("theta0", theta0_type),
    ("x0", x0_type)
]


@jitclass_(JITCLASS_SPEC)
class MultiMealT1DModel:
    def __init__(self,
                 u2ss,
                 theta0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 x0=Dict.empty(key_type=types.unicode_type, value_type=float64),
                 ):
        self.u2ss = np.float64(u2ss)
        self.x0 = x0 # TODO: check if this is the right way to do it
        self.reset(theta0)

    def reset(self,
              theta0=Dict.empty(key_type=types.unicode_type, value_type=float64)
              ):
        self.f = theta0["f"] if "f" in theta0 else np.float64(0.9)
        self.VG = theta0["VG"] if "VG" in theta0 else np.float64(1.45)
        self.VI = theta0["VI"] if "VI" in theta0 else np.float64(0.135)
        self.alpha = theta0["alpha"] if "alpha" in theta0 else np.float64(7)

        self.SI_B = theta0["SI_B"] if "SI_B" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SI_L = theta0["SI_L"] if "SI_L" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SI_D = theta0["SI_D"] if "SI_D" in theta0 else np.float64(10.35e-4 / self.VG)
        self.SG = theta0["SG"] if "SG" in theta0 else np.float64(2.5e-2)
        self.Gb = theta0["Gb"] if "Gb" in theta0 else np.float64(119.13)
        self.p2 = theta0["p2"] if "p2" in theta0 else np.float64(0.012)
        self.r1 = theta0["r1"] if "r1" in theta0 else np.float64(1.4407)
        self.r2 = theta0["r2"] if "r2" in theta0 else np.float64(0.8124)

        self.ka2 = theta0["ka2"] if "ka2" in theta0 else np.float64(0.014)
        self.kd = theta0["kd"] if "kd" in theta0 else np.float64(0.026)
        self.ke = theta0["ke"] if "ke" in theta0 else np.float64(0.127)
        self.tau = theta0["tau"] if "tau" in theta0 else np.float64(8)

        self.kabs_B = theta0["kabs_B"] if "kabs_B" in theta0 else np.float64(0.012)
        self.kabs_L = theta0["kabs_L"] if "kabs_L" in theta0 else np.float64(0.012)
        self.kabs_D = theta0["kabs_D"] if "kabs_D" in theta0 else np.float64(0.012)
        self.kabs_S = theta0["kabs_S"] if "kabs_S" in theta0 else np.float64(0.012)
        self.kabs_H = theta0["kabs_H"] if "kabs_H" in theta0 else np.float64(0.012)
        self.kempt = theta0["kempt"] if "kempt" in theta0 else np.float64(0.18)

        self.beta_B = theta0["beta_B"] if "beta_B" in theta0 else np.float64(0)
        self.beta_L = theta0["beta_L"] if "beta_L" in theta0 else np.float64(0)
        self.beta_D = theta0["beta_D"] if "beta_D" in theta0 else np.float64(0)
        self.beta_S = theta0["beta_S"] if "beta_S" in theta0 else np.float64(0)

        self._G0 = self.x0["G0"] if "G0" in self.x0 else np.float64(self.Gb)
        self._X0 = self.x0["X0"] if "X0" in self.x0 else np.float64(0)
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

        ki1 = self.u2ss / self.kd
        ki2 = self.kd / self.ka2 * ki1
        self.Ipb = self.ka2 / self.ke * ki2

        self._Isc10 = self.x0["Isc10"] if "Isc10" in self.x0 else np.float64(ki1)
        self._Isc20 = self.x0["Isc20"] if "Isc20" in self.x0 else np.float64(ki2)
        self._Ip0 = self.x0["Ip0"] if "Ip0" in self.x0 else np.float64(self.Ipb)
        self._IG0 = self.x0["IG0"] if "IG0" in self.x0 else self.Gb

        self.G = self._G0
        self.X = self._X0
        self.Qsto1_B = self._Qsto1_B_0
        self.Qsto2_B = self._Qsto2_B_0
        self.Qgut_B = self._Qgut_B_0
        self.Qsto1_L = self._Qsto1_L_0
        self.Qsto2_L = self._Qsto2_L_0
        self.Qgut_L = self._Qgut_L_0
        self.Qsto1_D = self._Qsto1_D_0
        self.Qsto2_D = self._Qsto2_D_0
        self.Qgut_D = self._Qgut_D_0
        self.Qsto1_S = self._Qsto1_S_0
        self.Qsto2_S = self._Qsto2_S_0
        self.Qgut_S = self._Qgut_S_0
        self.Qsto1_H = self._Qsto1_H_0
        self.Qsto2_H = self._Qsto2_H_0
        self.Qgut_H = self._Qgut_H_0
        self.Isc1 = self._Isc10
        self.Isc2 = self._Isc20
        self.Ip = self._Ip0
        self.IG = self._IG0

    def step(self, u: float64[:], t: float64):
        """

        :param u:
        :param t: minutes since start of simulation
        :return:
        """

        u_m_b = u[0] # TODO: add if to delay meals (tau and beta params)
        u_m_l = u[1]
        u_m_d = u[2]
        u_m_s = u[3]
        u_m_h = u[4]
        u_i = u[5] + u[6]
        u_h = u[7]

        if u_h < 4 or u_h >= 17:
            SI = self.SI_D
        elif 4 <= u_h < 11:
            SI = self.SI_B
        else:
            SI = self.SI_L

        g_prev = self.G
        logGb = np.log(self.Gb)
        log60 = np.log(60.0)
        if (g_prev < self.Gb) and (g_prev >= 60.0):
            lg = np.log(g_prev)
            diff = lg ** self.r2 - logGb ** self.r2
            risk = 1.0 + 10 * self.r1 * diff * diff
        elif g_prev < 60.0:
            diff = log60 ** self.r2 - logGb ** self.r2  # constant
            risk = 1.0 + 10 * self.r1 * diff * diff
        else:
            risk = 1.0

        dg = (-(self.SG + risk * self.X) * self.G + self.SG * self.Gb + self.f * (
                self.kabs_B * self.Qgut_B +
                self.kabs_L * self.Qgut_L +
                self.kabs_D * self.Qgut_D +
                self.kabs_S * self.Qgut_S +
                self.kabs_H * self.Qgut_H)  / self.VG)
        dx = -self.p2 * (self.X - SI / self.VI * (self.Ip - self.Ipb))
        dig = - 1 / self.alpha * (self.IG - self.G)

        dqsto1_b = -self.kempt * self.Qsto1_B + u_m_b
        dqsto2_b = self.kempt * self.Qsto1_B - self.kempt * self.Qsto2_B
        dqgut_b = self.kempt * self.Qsto2_B - self.kabs_B * self.Qgut_B

        dqsto1_l = -self.kempt * self.Qsto1_L + u_m_l
        dqsto2_l = self.kempt * self.Qsto1_L - self.kempt * self.Qsto2_L
        dqgut_l = self.kempt * self.Qsto2_L - self.kabs_L * self.Qgut_L

        dqsto1_d = -self.kempt * self.Qsto1_L + u_m_d
        dqsto2_d = self.kempt * self.Qsto1_L - self.kempt * self.Qsto2_L
        dqgut_d = self.kempt * self.Qsto2_L - self.kabs_L * self.Qgut_L

        dqsto1_s = -self.kempt * self.Qsto1_S + u_m_s
        dqsto2_s = self.kempt * self.Qsto1_S - self.kempt * self.Qsto2_S
        dqgut_s = self.kempt * self.Qsto2_S - self.kabs_S * self.Qgut_S

        dqsto1_h = -self.kempt * self.Qsto1_H + u_m_h
        dqsto2_h = self.kempt * self.Qsto1_H - self.kempt * self.Qsto2_H
        dqgut_h = self.kempt * self.Qsto2_H - self.kabs_H * self.Qgut_H

        disc1 = -self.kd * self.Isc1 + u_i
        disc2 = self.kd * self.Isc1 - self.ka2 * self.Isc2
        dip = self.ka2 * self.Isc2 - self.ke * self.Ip

        self.G = self.G + dg
        self.X = self.X + dx
        self.IG = self.IG + dig

        self.Qsto1_B = self.Qsto1_B + dqsto1_b
        self.Qsto2_B = self.Qsto2_B + dqsto2_b
        self.Qgut_B = self.Qgut_B + dqgut_b

        self.Qsto1_L = self.Qsto1_L + dqsto1_l
        self.Qsto2_L = self.Qsto2_L + dqsto2_l
        self.Qgut_L = self.Qgut_L + dqgut_l

        self.Qsto1_D = self.Qsto1_D + dqsto1_d
        self.Qsto2_D = self.Qsto2_D + dqsto2_d
        self.Qgut_D = self.Qgut_D + dqgut_d

        self.Qsto1_S = self.Qsto1_S + dqsto1_s
        self.Qsto2_S = self.Qsto2_S + dqsto2_s
        self.Qgut_S = self.Qgut_S + dqgut_s

        self.Qsto1_H = self.Qsto1_H + dqsto1_h
        self.Qsto2_H = self.Qsto2_H + dqsto2_h
        self.Qgut_H = self.Qgut_H + dqgut_h

        self.Isc1 = self.Isc1 + disc1
        self.Isc2 = self.Isc2 + disc2
        self.Ip = self.Ip + dip

    def step_be(self, u: float64[:], t: float64):
        """

        :param u:
        :param t: minutes since start of simulation
        :return:
        """

        u_m_b = u[0] # TODO: add if to delay meals (tau and beta params)
        u_m_l = u[1]
        u_m_d = u[2]
        u_m_s = u[3]
        u_m_h = u[4]
        u_i = u[5] + u[6]
        u_h = u[7]

        if u_h < 4 or u_h >= 17:
            SI = self.SI_D
        elif 4 <= u_h < 11:
            SI = self.SI_B
        else:
            SI = self.SI_L

        g_prev = self.G
        logGb = np.log(self.Gb)
        log60 = np.log(60.0)
        if (g_prev < self.Gb) and (g_prev >= 60.0):
            lg = np.log(g_prev)
            diff = lg ** self.r2 - logGb ** self.r2
            risk = 1.0 + 10 * self.r1 * diff * diff
        elif g_prev < 60.0:
            diff = log60 ** self.r2 - logGb ** self.r2  # constant
            risk = 1.0 + 10 * self.r1 * diff * diff
        else:
            risk = 1.0

        k1 = 1.0 / (1.0 + self.kempt)
        k2 = 1.0 / (1.0 + self.kempt)
        kd_fac = 1.0 / (1.0 + self.kd)

        self.Qsto1_B = (self.Qsto1_B + u_m_b) * k1
        self.Qsto2_B = (self.Qsto2_B + self.kempt * self.Qsto1_B) * k2
        self.Qgut_B = (self.Qgut_B + self.kempt * self.Qsto2_B) / (1 + self.kabs_B)

        self.Qsto1_L = (self.Qsto1_L + u_m_l) * k1
        self.Qsto2_L = (self.Qsto2_L + self.kempt * self.Qsto1_L) * k2
        self.Qgut_L = (self.Qgut_L + self.kempt * self.Qsto2_L) / (1 + self.kabs_L)

        self.Qsto1_D = (self.Qsto1_D + u_m_d) * k1
        self.Qsto2_D = (self.Qsto2_D + self.kempt * self.Qsto1_D) * k2
        self.Qgut_D = (self.Qgut_D + self.kempt * self.Qsto2_D) / (1 + self.kabs_D)

        self.Qsto1_S = (self.Qsto1_S + u_m_s) * k1
        self.Qsto2_S = (self.Qsto2_S + self.kempt * self.Qsto1_S) * k2
        self.Qgut_S = (self.Qgut_S + self.kempt * self.Qsto2_S) / (1 + self.kabs_S)

        self.Qsto1_H = (self.Qsto1_H + u_m_h) * k1
        self.Qsto2_H = (self.Qsto2_H + self.kempt * self.Qsto1_H) * k2
        self.Qgut_H = (self.Qgut_H + self.kempt * self.Qsto2_H) / (1 + self.kabs_H)

        self.Isc1 = (self.Isc1 + u_i) * kd_fac
        self.Isc2 = (self.Isc2 + self.kd * self.Isc1) / (1 + self.ka2)
        self.Ip = (self.Ip + self.ka2 * self.Isc2) / (1 + self.ke)

        self.X = (self.X + self.p2 * (SI / self.VI) * (self.Ip - self.Ipb)) / (1 + self.p2)
        self.G = (self.G + self.SG * self.Gb + self.f * (
                self.kabs_B * self.Qgut_B + self.kabs_L * self.Qgut_L + self.kabs_D * self.Qgut_D +
                self.kabs_S * self.Qgut_S + self.kabs_H * self.Qgut_H) / self.VG) / (1 + self.SG + risk * self.X)
        self.IG = (self.alpha * self.IG + self.G) / (1 + self.alpha)

    def output(self):
        return self.IG
