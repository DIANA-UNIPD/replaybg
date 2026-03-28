
from multiprocessing import freeze_support

import pandas as pd

from data.data import RBGData
from model.glucose_insulin_model import GlucoseInsulinModel
from twinner import Twinner
from distributions.log_norm import LogNormal, Normal

if __name__ == '__main__':
    freeze_support()
    df = pd.read_csv("data.csv")
    df['t'] = pd.to_datetime(df['t'])

    data = RBGData(df)

    model = GlucoseInsulinModel(u2ss=data.u2ss)


    twinner = Twinner()

    theta_estimated = twinner.twin(model=model,
                                   data=data,
                                   unknown_parameters_prior={'SI': LogNormal(mu=0.01, sigma=0.1),
                                                       'Gb': Normal(mu=120, sigma=10)})
    print(theta_estimated)
    """
    x0 = [100, ]
    model = GlucoseInsulinModel( theta=theta_estimated)
    simulator = Simulator()
    results = simulator.simulate()

    print(results)"""