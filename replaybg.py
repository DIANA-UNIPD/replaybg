from typing import Callable, Dict

import numpy as np
import pandas as pd

from environment import Environment
from data.data import RBGData
from distributions import LogNormal, Normal
from model.glucose_insulin_model import GlucoseInsulinModel
from twinner import Twinner


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
                 yts: int = 5,
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
        yts: int, optional, default : 5
            An integer that specifies the data sample time (in minutes).
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
                                       yts=yts,
                                       seed=seed,
                                       plot_mode=plot_mode,
                                       verbose=verbose)

    def twin(self, data: pd.DataFrame, bw: float, save_name: str,
             # twinning_method: str = 'mcmc', TODO: <-- this will become "custom_twinner : object = None" so user can pass their own Twinner
             # model: TODO: <-- this will become "model : object = None" so user can pass their own model
             u2ss: float | None = None, x0: Dict | None = None,
             # previous_data_name: str | None = None, # TODO: decide whether we still need it
             parallelize: bool = False, n_processes: int | None = None,
             ) -> None:
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
        #TODO: valida the inputs

        #if self.environment.verbose:
        #    print('Creating the digital twin using ' + twinning_method.upper())

        # Initialize model TODO: change this to the model provided in input
        model = GlucoseInsulinModel(u2ss=u2ss)

        # Unpack data to optimize performance during simulation
        rbg_data = RBGData(data=data) # TODO: use also the inputs model=model, environment=self.environment to set up the data in a model-agnostic fashion

        # Initialize the twinner
        twinner = Twinner()

        # Run the twinning procedure
        theta_estimated = twinner.twin(model=model,
                                       data=rbg_data,
                                       unknown_parameters_prior={'SI': LogNormal(mu=0.01, sigma=0.1),
                                                                 'Gb': Normal(mu=120, sigma=10)})
        print(theta_estimated)
