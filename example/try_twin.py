
from multiprocessing import freeze_support

import os
import pandas as pd

from model.glucose_insulin_model import GlucoseInsulinModel
from replaybg import ReplayBG

if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data.parquet")
    df['t'] = pd.to_datetime(df['t'])

    save_folder = os.path.join(os.path.abspath(''))
    save_name = 'test'

    rbg = ReplayBG(save_folder=save_folder)
    theta_estimated = rbg.twin(data=df, bw=100, save_name=save_name)
    out = rbg.replay(data=df,theta=theta_estimated, bw=100, save_name=save_name)

    import matplotlib.pyplot as plt
    plt.plot(df.glucose)
    plt.plot(out)
    plt.show()


    """
    x0 = [100, ]


    print(results)"""