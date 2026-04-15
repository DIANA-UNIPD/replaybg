from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from data.multi_meal_t1d_data import MultiMealT1DData

from environment import Environment
from model.multi_meal_t1d import MultiMealT1DModel
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

    def twin(self, data, bw: float, save_name: str,
             # twinning_method: str = 'mcmc', TODO: <-- this will become "custom_twinner : object = None" so user can pass their own Twinner
             model: object = None,
             unknown_parameters_prior: Dict = None,
             n_starts: int = 64,
             u2ss: float | None = None,
             x0_setup: Optional[Callable] = None,
             # previous_data_name: str | None = None, # TODO: decide whether we still need it
             parallelize: bool = False, n_jobs: int | None = None,
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
        x0_setup : callable, optional, default : None
            A ``(model, data) -> None`` callable that sets up initial model conditions
            and updates the data forcing inputs before twinning begins.  Typically
            created by ``MultiMealT1DModel.setup_x0(x0, previous_theta)`` when twinning
            consecutive days of data.  If ``None``, the model is used as-is (cold start).

        parallelize : boolean, optional, default : False
            A boolean that specifies whether to parallelize the twinning process.
        n_jobs : int, optional, default : None
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
        # TODO: validate the inputs

        # if self.environment.verbose:
        #    print('Creating the digital twin using ' + twinning_method.upper())

        if x0_setup is not None:
            x0_setup(model, data)

        # Initialize the twinner
        twinner = Twinner(parallelize=parallelize, n_jobs=n_jobs, n_starts=n_starts)

        # Run the twinning procedure
        theta_estimated = twinner.twin(model=model,
                                       data=data,
                                       unknown_parameters_prior=unknown_parameters_prior,
                                       environment=self.environment)

        return dict(zip(unknown_parameters_prior.keys(), theta_estimated['x']))


    def replay(self, data: pd.DataFrame, theta, bw: float, save_name: str,
               # twinning_method: str = 'mcmc', TODO: <-- this will become "custom_twinner : object = None" so user can pass their own Twinner
               # model: TODO: <-- this will become "model : object = None" so user can pass their own model
               u2ss: float | None = None, x0_setup: Optional[Callable] = None,
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
        # TODO: validate the inputs

        # if self.environment.verbose:
        #    print('Creating the digital twin using ' + twinning_method.upper())

        # Unpack data to optimize performance during simulation
        rbg_data = MultiMealT1DData(data=data,
                           environment=self.environment)  # TODO: use also the inputs model=model, environment=self.environment to set up the data in a model-agnostic fashion

        # convert theta to numba typed dict
        theta_typed = to_typed_f32_dict(theta)

        # Initialize model TODO: change this to the model provided in input
        model = MultiMealT1DModel(u2ss=rbg_data.u2ss, theta0=theta_typed)  # TODO: can we set u2ss AFTER data?

        if x0_setup is not None:
            x0_setup(model, rbg_data)

        out = np.zeros(rbg_data.tsteps, )
        out[0] = model.output(0)
        for k in np.arange(1,out.shape[0]):
            model.step(rbg_data.u[k], k)
            out[k] = model.output(k)

        out = out[0::rbg_data.yts]

        return out
