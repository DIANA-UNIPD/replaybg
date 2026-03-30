from datetime import datetime

import numpy as np
import pandas as pd

from environment import Environment


class T1DData:
    def __init__(self, data=pd.DataFrame, data_to_input=None, body_weight=100., environment: Environment = None, ):
        if data_to_input is None:
            self.data_to_input = {0: 'meal',
                                  1: 'bolus',
                                  2: 'basal'}
        else:
            self.data_to_input = data_to_input

        self.u2ss = np.mean(data.basal.values)
        self.body_weight = body_weight

        self.yts = 5

        # From the time retain only the hour since is the only thing actually needed during the simulation
        self.__time_setup(data, environment)

        # Set glucose from given data
        self.glucose = data.glucose.values.astype(float)
        self.glucose_idxs = np.where(~np.isnan(self.glucose))[0]

        # Set insulin from given data
        self.__insulin_setup(data)
        self.__meal_setup(data)
        self.__setup_u()

    def __time_setup(self,
                     data: pd.DataFrame,
                     environment: Environment = None,
                     ) -> None:
        """
        Unpacks the time data.

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe which contains the data to be used by the tool.

        Returns
        -------
        None

        Raises
        ------
        None

        See Also
        --------
        None

        Examples
        --------
        None
        """

        self.tsteps = int(
            (np.array(data.t)[-1].astype(datetime) - np.array(data.t)[0].astype(datetime)).total_seconds() / (
                60) + self.yts) * environment.ts  # number of steps in the simulation with the sampling rate of the integration step
        self.tysteps = int(
            self.tsteps / self.yts)  # number of steps of the simulation, with the sampling rate of the data

        self.t_data = data['t'].to_numpy()  # times in the data

        self.t_hour = np.zeros([self.tsteps, ])  # hours in the data for each integration step
        self.t_min = np.zeros([self.tsteps, ])  # minutes in the data for each integration step
        t_m = np.array(data.t.dt.minute.values).astype(int)
        t_h = np.array(data.t.dt.hour.values).astype(int)

        for t in range(data.shape[0]):
            self.t_hour[(t * self.yts):((t + 1) * self.yts)] = t_h[t]
            self.t_min[(t * self.yts):((t + 1) * self.yts)] = np.arange(t_m[t], t_m[t] + self.yts)

    def __insulin_setup(self,
                        data: pd.DataFrame,
                        ) -> None:
        """
        Unpacks the insulin data.

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe which contains the data to be used by the tool.
        model: T1DModelSingleMeal | T1DModelMultiMeal
            An object that represents the physiological model to be used by ReplayBG.

        Returns
        -------
        None

        Raises
        ------
        None

        See Also
        --------
        None

        Examples
        --------
        None
        """
        self.basal = np.zeros([self.tsteps, ])
        self.bolus = np.zeros([self.tsteps, ])
        self.bolus_label = np.empty([self.tsteps, ], dtype=str)

        self.bolus_data = []
        self.basal_data = []

        self.bolus_data = data.bolus.values

        # Find the boluses
        b_idx = np.where(data.bolus)[0]

        # Set the bolus vector
        for i in range(np.size(b_idx)):
            self.bolus[(b_idx[i] * self.yts): ((b_idx[i] + 1) * self.yts)] = data['bolus'][b_idx[
                i]] * (1000 / self.body_weight)  # mU/(kg*min)
            self.bolus_label[(b_idx[i] * self.yts): ((b_idx[i] + 1) * self.yts)] = data['bolus_label'][b_idx[i]]

        self.basal_data = data.basal.values
        # Set the basal vector
        for time in range(0, np.size(np.arange(0, self.tsteps, self.yts))):
            self.basal[(time * self.yts): ((time + 1) * self.yts)] = \
                data['basal'][time] * (1000 / self.body_weight)  # mU/(kg*min)

    def __meal_setup(self,
                     data: pd.DataFrame,
                     ) -> None:
        """
        Unpacks the meal data.

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe which contains the data to be used by the tool.
        environment: Environment
            An object that represents the hyperparameters to be used by ReplayBG.
        model: T1DModelSingleMeal | T1DModelMultiMeal
            An object that represents the physiological model to be used by ReplayBG.

        Returns
        -------
        None

        Raises
        ------
        None

        See Also
        --------
        None

        Examples
        --------
        None
        """

        # Initialize the meal vector
        self.meal = np.zeros([self.tsteps, ])

        # Initialize the mealAnnouncements vector
        self.meal_announcement = np.zeros([self.tsteps, ])

        # Initialize the meal type vector
        self.meal_type = np.empty([self.tsteps, ], dtype=str)

        self.meal_B = np.zeros([self.tsteps, ])
        self.meal_L = np.zeros([self.tsteps, ])
        self.meal_D = np.zeros([self.tsteps, ])
        self.meal_S = np.zeros([self.tsteps, ])
        self.meal_H = np.zeros([self.tsteps, ])

        self.meal_B2 = np.zeros([self.tsteps, ])
        self.meal_L2 = np.zeros([self.tsteps, ])
        self.meal_S2 = np.zeros([self.tsteps, ])

        self.meal_data = []

        self.meal_data = data.cho.values

        # Find the meals
        m_idx = np.where(data.cho)[0]

        # Set the main meal vector
        for i in range(np.size(m_idx)):
            self.meal[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = data['cho'][m_idx[
                i]] * (1000 / self.body_weight)  # mg/(kg*min)
            self.meal_announcement[(m_idx[i] * self.yts)] = data['cho'][m_idx[i]] * self.yts  # mg/(kg*min)

            self.meal_type[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = data['cho_label'][m_idx[i]]

            if data['cho_label'][m_idx[i]] == 'B':
                self.meal_B[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'L':
                self.meal_L[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'D':
                self.meal_D[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'S':
                self.meal_S[(m_idx[i] * self.yts):(
                        (m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'H':
                self.meal_H[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]

            if data['cho_label'][m_idx[i]] == 'B2':
                self.meal_B2[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'L2':
                self.meal_L2[(m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]
            if data['cho_label'][m_idx[i]] == 'S2':
                self.meal_S2[(m_idx[i] * self.yts):(
                        (m_idx[i] + 1) * self.yts)] = self.meal[
                    (m_idx[i] * self.yts):((m_idx[i] + 1) * self.yts)]

    def __setup_u(self):
        self.u = np.empty((self.tsteps, len(self.data_to_input.keys())))
        for i in range(len(self.data_to_input.keys())):
            self.u[:, i] = getattr(self, self.data_to_input[i])
