
from multiprocessing import freeze_support

import os
import pandas as pd

from replaybg import ReplayBG

if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data.parquet")
    df['t'] = pd.to_datetime(df['t'])

    save_folder = os.path.join(os.path.abspath(''))
    save_name = 'test'

    rbg = ReplayBG(save_folder=save_folder)
    rbg.twin(data=df, bw=100, save_name=save_name)


    """
    x0 = [100, ]
    model = GlucoseInsulinModel(theta=theta_estimated)
    simulator = Simulator()
    results = simulator.simulate()

    print(results)"""