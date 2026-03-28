import numpy as np
import pandas as pd

class RBGData:
    def __init__(self, data=pd.DataFrame):
        self.glucose = data['glucose'].values
        self.cho = data['cho'].values
        self.bolus = data['bolus'].values
        self.t = data['t'].values
        self.u2ss = np.mean(data.bolus.values)