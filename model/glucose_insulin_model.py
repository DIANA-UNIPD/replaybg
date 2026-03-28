import numpy as np
from numba import float32
from numba.typed import Dict
from numba import types

from environment.config import jitclass_
from model import Model

theta_type = types.DictType(types.unicode_type, float32)
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
    ("theta", theta_type)
]


@jitclass_(JITCLASS_SPEC)
class GlucoseInsulinModel:
    def __init__(self, u2ss, theta=Dict.empty(
        key_type=types.unicode_type,
        value_type=float32,
    )):

        self.f = theta["f"] if "f" in theta else np.float32(0.9)
        self.VG = theta["VG"] if "VG" in theta else np.float32(1.45)
        self.VI = theta["VI"] if "VI" in theta else np.float32(0.135)
        self.alpha = theta["alpha"] if "alpha" in theta else np.float32(7)

        self.SI = theta["SI"] if "SI" in theta else np.float32(10.35e-4 / self.VG)
        self.SG = theta["SG"] if "SG" in theta else np.float32(2.5e-2)
        self.Gb = theta["Gb"] if "Gb" in theta else np.float32(119.13)
        self.p2 = theta["p2"] if "p2" in theta else np.float32(0.012)

        self.ka2 = theta["ka2"] if "ka2" in theta else np.float32(0.014)
        self.kd = theta["kd"] if "kd" in theta else np.float32(0.026)
        self.ke = theta["ke"] if "ke" in theta else np.float32(0.127)

        self.kabs = theta["kabs"] if "kabs" in theta else np.float32(0.012)
        self.kempt = theta["kempt"] if "kempt" in theta else np.float32(0.18)

        self.u2ss = np.float32(u2ss)

        self._G0 = np.float32(self.Gb)
        self._X0 = np.float32(0)
        self._Qsto10 = np.float32(0)
        self._Qsto20 = np.float32(0)
        self._Qgut0 = np.float32(0)

        ki1 = self.u2ss / self.kd
        ki2 = self.kd / self.ka2 * ki1
        self.Ipb = self.ka2 / self.ke * ki2

        self._Isc10 = np.float32(ki1)
        self._Isc20 = np.float32(ki2)
        self._Ip0 = np.float32(self.Ipb)
        self._IG0 = self.Gb

        self.G = self._G0
        self.X = self._X0
        self.Qsto1 = self._Qsto10
        self.Qsto2 = self._Qsto20
        self.Qgut = self._Qgut0
        self.Isc1 = self._Isc10
        self.Isc2 = self._Isc20
        self.Ip = self._Ip0
        self.IG = self._IG0

    def reset(self):
        self.G = self._G0
        self.X = self._X0
        self.Qsto1 = self._Qsto10
        self.Qsto2 = self._Qsto20
        self.Qgut = self._Qgut0
        self.Isc1 = self._Isc10
        self.Isc2 = self._Isc20
        self.Ip = self._Ip0
        self.IG = self._IG0

    def step(self, u: float32[:]):
        u_m = u[0]
        u_b = u[1]

        dg = -(self.SG + self.X) * self.G + self.SG * self.Gb + self.f * self.kabs * self.Qgut / self.VG
        dx = -self.p2 * (self.X - self.SI * (self.Ip - self.Ipb))
        dig = - 1/self.alpha * (self.IG - self.G)

        dqsto1 = -self.kempt * self.Qsto1 + u_m
        dqsto2 = self.kempt * self.Qsto1 - self.kempt * self.Qsto2
        dqgut = self.kempt * self.Qsto2 - self.kabs * self.Qgut

        disc1 = -self.kd * self.Isc1 + u_b / self.VI
        disc2 = self.kd * self.Isc1 - self.ka2 * self.Isc2
        dip = self.ka2 * self.Isc2 - self.ke * self.Ip

        self.G = self.G + dg
        self.X = self.X + dx
        self.IG = self.X + dig

        self.Qsto1 = self.Qsto1 + dqsto1
        self.Qsto2 = self.Qsto2 + dqsto2
        self.Qgut = self.Qgut + dqgut

        self.Isc1 = self.Isc1 + disc1
        self.Isc2 = self.Isc2 + disc2
        self.Ip = self.Ip + dip

    def output(self):
        return self.IG
