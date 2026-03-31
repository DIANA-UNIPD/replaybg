from typing import Callable, Dict

import numpy as np
import pandas as pd

from data.t1ddata import T1DData
from environment import Environment
from distributions import LogNormal, LogGamma, LogLogNormal, to_constrained
from model.glucose_insulin_model import GlucoseInsulinModel
from twinner.twinner import Twinner
from utils.numba_dicts import to_typed_f32_dict


class ReplayBG:
    """
    Core class of ReplayBG.

    ...
    Attributes
    ----------
    environment: Environment
        An object that represents the hyperparameters to be used by ReplayBG.

    Methods
    -------
    TODO
    """

    def __init__(self, save_folder: str,
                 ts: int = 1,
                 seed: int = 1,
                 plot_mode: bool = True, verbose: bool = True
                 ):
        """
        Constructs all the necessary attributes for the ReplayBG object.

        Parameters
        ----------
        save_folder : str
            A string defining the folder that will contain the results of the twinning procedure and the replay
            simulations.
        ys: int, optional, default : 1
            An integer that specifies the integration step (in minutes).
        seed: int, optional, default: 1
            An integer that specifies the random seed. For reproducibility.
        plot_mode : boolean, optional, default : True
            A boolean that specifies whether to show the plot of the results or not.
        verbose : boolean, optional, default : True
            A boolean that specifies the verbosity of ReplayBG.

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

        References
        --------
        Cappon et al., "ReplayBG: a methodology to identify a personalized model from type 1 diabetes data and simulate
        glucose concentrations to assess alternative therapies", IEEE Transactions on Biomedical Engineering, 2023.
        """

        # TODO: Validate input

        # Initialize the environment parameters
        self.environment = Environment(save_folder=save_folder,
                                       ts=ts,
                                       seed=seed,
                                       plot_mode=plot_mode,
                                       verbose=verbose)

    def twin(self, data: pd.DataFrame, bw: float, save_name: str,
             # twinning_method: str = 'mcmc', TODO: <-- this will become "custom_twinner : object = None" so user can pass their own Twinner
             # model: TODO: <-- this will become "model : object = None" so user can pass their own model
             u2ss: float | None = None, x0: Dict | None = None,
             # previous_data_name: str | None = None, # TODO: decide whether we still need it
             parallelize: bool = False, n_processes: int | None = None,
             ) -> Dict:
        """
        Runs ReplayBG twinning procedure.

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe which contains the data to be used by the tool.
        bw: float
            The patient's body weight.
        save_name : str
            A string used to label, thus identify, each output file and result.

        u2ss : float, optional, default : None
            The steady state of the basal insulin infusion.
        x0 : list, optional, default : None
            The initial model conditions.

        parallelize : boolean, optional, default : False
            A boolean that specifies whether to parallelize the twinning process.
        n_processes : int, optional, default : None
            The number of processes to be spawn if `parallelize` is `True`. If None, the number of CPU cores is used.

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
        # TODO: valida the inputs

        # if self.environment.verbose:
        #    print('Creating the digital twin using ' + twinning_method.upper())

        # Unpack data to optimize performance during simulation
        rbg_data = T1DData(data=data,
                           environment=self.environment)  # TODO: use also the inputs model=model, environment=self.environment to set up the data in a model-agnostic fashion

        # Initialize model TODO: change this to the model provided in input
        model = GlucoseInsulinModel(u2ss=rbg_data.u2ss)  # TODO: can we set u2ss AFTER data?

        # Initialize the twinner
        twinner = Twinner()

        unknown_parameters_prior = {
            'SI': {'prior': LogGamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': 1},
            'SG' : {'prior': LogLogNormal(mu=-3.8, sigma=0.5), 'min': 0, 'max': 1},
            'Gb': {'prior': LogNormal(mu=119.13, sigma=7.11), 'min': 70, 'max': 180},
            'p2': {'prior': LogNormal(mu=0.11, sigma=0.004), 'min': 0, 'max': 1},
            'kabs': {'prior': LogLogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': 1},
            'kempt' : {'prior' : LogLogNormal(mu=-1.9646, sigma=0.7069), 'min': 0, 'max': 1},
            'ka2' : {'prior' : LogLogNormal(mu=-4.2875, sigma=0.4274), 'min': 0, 'max': 1},
            'kd' : {'prior' : LogLogNormal(mu=-3.5090, sigma=0.6187), 'min': 0, 'max': 1},
        }

        # Run the twinning procedure
        theta_estimated = twinner.twin(model=model,
                                       data=rbg_data,
                                       unknown_parameters_prior=unknown_parameters_prior)
        print(theta_estimated['fun'], to_constrained(theta_estimated['x']))
        theta_phi = {}
        for i, k in enumerate(unknown_parameters_prior.keys()):
            theta_phi[k] = to_constrained(theta_estimated['x'][i], unknown_parameters_prior[k]['min'],
                                          unknown_parameters_prior[k]['max'])

        return dict(zip(unknown_parameters_prior.keys(), theta_phi.values()))


    def replay(self, data: pd.DataFrame, theta, bw: float, save_name: str,
               # twinning_method: str = 'mcmc', TODO: <-- this will become "custom_twinner : object = None" so user can pass their own Twinner
               # model: TODO: <-- this will become "model : object = None" so user can pass their own model
               u2ss: float | None = None, x0: Dict | None = None,
               # previous_data_name: str | None = None, # TODO: decide whether we still need it
               parallelize: bool = False, n_processes: int | None = None,
               ) -> Dict:
        """
        Runs ReplayBG twinning procedure.

        Parameters
        ----------
        data: pd.DataFrame
            Pandas dataframe which contains the data to be used by the tool.
        bw: float
            The patient's body weight.
        save_name : str
            A string used to label, thus identify, each output file and result.

        u2ss : float, optional, default : None
            The steady state of the basal insulin infusion.
        x0 : list, optional, default : None
            The initial model conditions.

        parallelize : boolean, optional, default : False
            A boolean that specifies whether to parallelize the twinning process.
        n_processes : int, optional, default : None
            The number of processes to be spawn if `parallelize` is `True`. If None, the number of CPU cores is used.

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
        # TODO: valida the inputs

        # if self.environment.verbose:
        #    print('Creating the digital twin using ' + twinning_method.upper())

        # Unpack data to optimize performance during simulation
        rbg_data = T1DData(data=data,
                           environment=self.environment)  # TODO: use also the inputs model=model, environment=self.environment to set up the data in a model-agnostic fashion

        # convert theta to numba typed dict
        theta_typed = to_typed_f32_dict(theta)

        # Initialize model TODO: change this to the model provided in input
        model = GlucoseInsulinModel(u2ss=rbg_data.u2ss, theta0=theta_typed)  # TODO: can we set u2ss AFTER data?

        out = np.zeros(rbg_data.tsteps, )
        for k in range(out.shape[0]):
            model.step(rbg_data.u[k], k)
            out[k] = model.output()

        out = out[0::rbg_data.yts]

        return out
