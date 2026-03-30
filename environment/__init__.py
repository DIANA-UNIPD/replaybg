import os


class Environment:
    """
    A class that represents the hyperparameters to be used by ReplayBG.

    ...
    Attributes
    ----------
    save_name : str
        A string used to label, thus identify, each output file and result.
    replay_bg_path: str
        The current absolute path of ReplayBG.

    blueprint: str, {'single-meal', 'multi-meal'}
            A string that specifies the blueprint to be used to create the digital twin.
    yts: int
        An integer that specifies the data sampling time (in minutes).

    seed : int
        An integer that specifies the random seed. For reproducibility.

    plot_mode : bool
        A boolean that specifies whether to show the plot of the results or not.
    verbose : bool
        A boolean that specifies the verbosity of ReplayBG.

    Methods
    -------
    None
    """

    def __init__(self,
                 save_name: str = '',
                 save_folder: str = '',
                 yts: int = 5,
                 seed: int = 42,
                 plot_mode: bool = True,
                 verbose: bool = True
                 ):
        """
        Constructs all the necessary attributes for the Environment object.

        Parameters
        ----------

        save_name : str, optional, default : ''
            A string used to label, thus identify, each output file and result.
        save_folder : str, optional, default : ''
            A string used to set the folder where the ReplayBG results will be saved.

        yts: int, optional, default : 5
            An integer that specifies the data sampling time (in minutes).

        seed : int, optional, default : 1
            An integer that specifies the random seed. For reproducibility.

        plot_mode : boolean, optional, default : True
            A boolean that specifies whether to show the plot of the results or not.
        verbose : boolean, optional, default : True
            A boolean that specifies the verbosity of ReplayBG.
        """

        # Set the save name and folder
        self.save_name = save_name
        self.replay_bg_path = save_folder

        # Create the results sub folders if they do not exist
        if not (os.path.exists(os.path.join(self.replay_bg_path, 'results'))):
            os.mkdir(os.path.join(self.replay_bg_path, 'results'))
        if not (os.path.exists(os.path.join(self.replay_bg_path, 'results', 'twins'))):
            os.mkdir(os.path.join(self.replay_bg_path, 'results', 'twins'))
        if not (os.path.exists(os.path.join(self.replay_bg_path, 'results', 'workspaces'))):
            os.mkdir(os.path.join(self.replay_bg_path, 'results', 'workspaces'))

        # Set sample time
        self.yts = yts


        # Set the seed
        self.seed = seed

        # Set plot mode and verbosity
        self.plot_mode = plot_mode
        self.verbose = verbose
