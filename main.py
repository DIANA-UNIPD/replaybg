# This is a sample Python script.
import datetime

import numpy as np
import pandas as pd

from data.data import RBGData
from model.glucose_insulin_model import GlucoseInsulinModel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # g = GlucoseInsulinModel([100, 100, 100, 100])
    # print(g.G)


    df = pd.read_csv("example/data.csv")
    df['t'] = pd.to_datetime(df['t'])

    data = RBGData(df)
    print(len(data.t))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
