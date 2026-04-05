
from multiprocessing import freeze_support

import os
import pandas as pd

from model.single_meal_t1d import SingleMealT1DModel
from replaybg import ReplayBG

if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data_day_1.parquet")
    df['t'] = pd.to_datetime(df['t'])
    save_folder = os.path.join(os.path.abspath(''))
    save_name = 'test'

    df.loc[10, 'cho'] = 10
    df.loc[10, 'cho_label'] = 'B'
    rbg = ReplayBG(save_folder=save_folder)
    theta_estimated = rbg.twin(data=df, bw=100, save_name=save_name)
    print(theta_estimated)
    out = rbg.replay(data=df,theta=theta_estimated, bw=100, save_name=save_name)

    # TODO: add a plot utility
    import matplotlib.pyplot as plt
    plt.plot(df.glucose)
    plt.plot(out)
    plt.show()