import numpy as np
from numba import float32
from numba.typed import Dict
from numba import types

from environment.config import jitclass_

theta0_type = types.DictType(types.unicode_type, float32)
x0_type = types.DictType(types.unicode_type, float32)
JITCLASS_SPEC = [
    ("_G0", float32),
    ("_X0", float32),
    ("_Qsto10", float32),
    ("_Qsto20", float32),
    ("_Qgut0", float32),
    ("_Isc10", float32),
    ("_Isc20", float32),
    ("_Ip0", float32),
    ("_IG0", float32),
    ("G", float32),
    ("X", float32),
    ("Qsto1", float32),
    ("Qsto2", float32),
    ("Qgut", float32),
    ("Isc1", float32),
    ("Isc2", float32),
    ("Ip", float32),
    ("IG", float32),
    ("SI", float32),
    ("SG", float32),
    ("Gb", float32),
    ("p2", float32),
    ("ka2", float32),
    ("kd", float32),
    ("ke", float32),
    ("kabs", float32),
    ("kempt", float32),
    ("f", float32),
    ("VG", float32),
    ("VI", float32),
    ("alpha", float32),
    ("Ipb", float32),
    ("u2ss", float32),
    ("theta0", theta0_type),
    ("x0", x0_type)
]


@jitclass_(JITCLASS_SPEC)
class SingleMealT1DModel:
    def __init__(self,
                 u2ss,
                 theta0=Dict.empty(key_type=types.unicode_type, value_type=float32),
                 x0=Dict.empty(key_type=types.unicode_type, value_type=float32),
                 ):
        self.u2ss = np.float32(u2ss)
        self.x0 = x0 # TODO: check if this is the right way to do it
        self.reset(theta0)

    def reset(self,
              theta0=Dict.empty(key_type=types.unicode_type, value_type=float32)
              ):
        self.f = theta0["f"] if "f" in theta0 else np.float32(0.9)
        self.VG = theta0["VG"] if "VG" in theta0 else np.float32(1.45)
        self.VI = theta0["VI"] if "VI" in theta0 else np.float32(0.135)
        self.alpha = theta0["alpha"] if "alpha" in theta0 else np.float32(7)

        self.SI = theta0["SI"] if "SI" in theta0 else np.float32(10.35e-4 / self.VG)
        self.SG = theta0["SG"] if "SG" in theta0 else np.float32(2.5e-2)
        self.Gb = theta0["Gb"] if "Gb" in theta0 else np.float32(119.13)
        self.p2 = theta0["p2"] if "p2" in theta0 else np.float32(0.012)

        self.ka2 = theta0["ka2"] if "ka2" in theta0 else np.float32(0.014)
        self.kd = theta0["kd"] if "kd" in theta0 else np.float32(0.026)
        self.ke = theta0["ke"] if "ke" in theta0 else np.float32(0.127)

        self.kabs = theta0["kabs"] if "kabs" in theta0 else np.float32(0.012)
        self.kempt = theta0["kempt"] if "kempt" in theta0 else np.float32(0.18)

        self._G0 = self.x0["G0"] if "G0" in self.x0 else np.float32(self.Gb)
        self._X0 = self.x0["X0"] if "X0" in self.x0 else np.float32(0)
        self._Qsto10 = self.x0["Qsto10"] if "Qsto10" in self.x0 else np.float32(0)
        self._Qsto20 = self.x0["Qsto20"] if "Qsto20" in self.x0 else np.float32(0)
        self._Qgut0 = self.x0["Qgut0"] if "Qgut0" in self.x0 else np.float32(0)

        ki1 = self.u2ss / self.kd
        ki2 = self.kd / self.ka2 * ki1
        self.Ipb = self.ka2 / self.ke * ki2

        self._Isc10 = self.x0["Isc10"] if "Isc10" in self.x0 else np.float32(ki1)
        self._Isc20 = self.x0["Isc20"] if "Isc20" in self.x0 else np.float32(ki2)
        self._Ip0 = self.x0["Ip0"] if "Ip0" in self.x0 else np.float32(self.Ipb)
        self._IG0 = self.x0["IG0"] if "IG0" in self.x0 else self.Gb

        self.G = self._G0
        self.X = self._X0
        self.Qsto1 = self._Qsto10
        self.Qsto2 = self._Qsto20
        self.Qgut = self._Qgut0
        self.Isc1 = self._Isc10
        self.Isc2 = self._Isc20
        self.Ip = self._Ip0
        self.IG = self._IG0

    def step(self, u: float32[:], t: float32):
        """

        :param u:
        :param t: minutes since start of simulation
        :return:
        """

        u_m = u[0]
        u_i = u[1] + u[2]  # TODO: add if to delay meals (tau and beta params)

        dg = -(self.SG + self.X) * self.G + self.SG * self.Gb + self.f * self.kabs * self.Qgut / self.VG
        dx = -self.p2 * (self.X - self.SI * (self.Ip - self.Ipb))
        dig = - 1 / self.alpha * (self.IG - self.G)

        dqsto1 = -self.kempt * self.Qsto1 + u_m
        dqsto2 = self.kempt * self.Qsto1 - self.kempt * self.Qsto2
        dqgut = self.kempt * self.Qsto2 - self.kabs * self.Qgut

        disc1 = -self.kd * self.Isc1 + u_i / self.VI
        disc2 = self.kd * self.Isc1 - self.ka2 * self.Isc2
        dip = self.ka2 * self.Isc2 - self.ke * self.Ip

        self.G = self.G + dg
        self.X = self.X + dx
        self.IG = self.IG + dig

        self.Qsto1 = self.Qsto1 + dqsto1
        self.Qsto2 = self.Qsto2 + dqsto2
        self.Qgut = self.Qgut + dqgut

        self.Isc1 = self.Isc1 + disc1
        self.Isc2 = self.Isc2 + disc2
        self.Ip = self.Ip + dip

    def output(self):
        return self.IG
