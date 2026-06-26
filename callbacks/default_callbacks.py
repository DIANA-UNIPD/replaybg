from callbacks.callback import ReplayCallback


class CorrectionBolus(ReplayCallback):
    """Reactive correction-bolus policy with a refractory window.

    If the most recent glucose reading is above ``threshold`` and no correction
    has been issued within the last ``lockout_min`` minutes, deliver a correction
    bolus of ``(glucose - target) / cf`` units.

    ...
    Attributes
    ----------
    threshold : float
        Glucose level (mg/dL) above which a correction is considered.
    target : float
        Target glucose level (mg/dL) used to size the correction.
    cf : float
        Correction factor (mg/dL per unit of insulin).
    lockout_min : int
        Minimum number of minutes between consecutive boluses.
    last_correction_k : int
        Integration minute of the last issued correction (memory across steps).

    Methods
    -------
    action(ctx):
        Issues a correction bolus when the policy conditions are met.
    """

    def __init__(self, threshold=180.0, target=120.0, cf=40.0, lockout_min=60):
        """Constructs all the necessary attributes for the CorrectionBolus object.

        Parameters
        ----------
        threshold : float, optional, default : 180.0
            Glucose level (mg/dL) above which a correction is considered.
        target : float, optional, default : 120.0
            Target glucose level (mg/dL) used to size the correction.
        cf : float, optional, default : 40.0
            Correction factor (mg/dL per unit of insulin).
        lockout_min : int, optional, default : 60
            Minimum number of minutes between consecutive boluses.
        """
        self.threshold = threshold
        self.target = target
        self.cf = cf
        self.lockout_min = lockout_min
        self.last_correction_k = -10 ** 9

    def action(self, ctx):
        """Issues a correction bolus when glucose is high and the lockout elapsed.

        Parameters
        ----------
        ctx : ReplayContext
            The replay context for the current step.
        """
        g = ctx.measurement
        if g > self.threshold and (ctx.k - self.last_correction_k) >= self.lockout_min:
            cb_u = (g - self.target) / self.cf
            cb_mu_kgmin = cb_u * (1000.0 / self.rbg_data.body_weight)
            bolus_prev = ctx.get_input("bolus")
            ctx.add_input("bolus", cb_mu_kgmin)
            ctx.log(u=cb_u, prev_u=bolus_prev, new_u=ctx.get_input("bolus"))
            self.last_correction_k = ctx.k


class HypoTreatment(ReplayCallback):
    """Reactive hypo-treatment policy with a refractory window.

    If the most recent glucose reading is below ``threshold`` and no treatment
    has been issued within the last ``lockout_min`` minutes, deliver a fixed
    amount of fast-acting carbohydrates (the "rule of 15": 15 g, wait 15 min).

    ...
    Attributes
    ----------
    threshold : float
        Glucose level (mg/dL) below which a treatment is considered.
    carbs : float
        Amount of rescue carbohydrates to deliver (grams).
    lockout_min : int
        Minimum number of minutes between consecutive treatments.
    last_treatment_k : int
        Integration minute of the last issued treatment (memory across steps).

    Methods
    -------
    action(ctx):
        Issues a hypo-treatment when the policy conditions are met.
    """

    def __init__(self, threshold=70.0, carbs=15.0, lockout_min=15):
        """Constructs all the necessary attributes for the HypoTreatment object.

        Parameters
        ----------
        threshold : float, optional, default : 70.0
            Glucose level (mg/dL) below which a treatment is considered.
        carbs : float, optional, default : 15.0
            Amount of rescue carbohydrates to deliver (grams).
        lockout_min : int, optional, default : 15
            Minimum number of minutes between consecutive treatments.
        """
        self.threshold = threshold
        self.carbs = carbs
        self.lockout_min = lockout_min
        self.last_treatment_k = -10 ** 9

    def action(self, ctx):
        """Issues a hypo-treatment when glucose is low and the lockout elapsed.

        Parameters
        ----------
        ctx : ReplayContext
            The replay context for the current step.
        """
        g = ctx.measurement
        if g < self.threshold and (ctx.k - self.last_treatment_k) >= self.lockout_min:
            carbs_mg_kgmin = self.carbs * (1000.0 / self.rbg_data.body_weight)
            meal_prev = ctx.get_input("meal_H")
            ctx.add_input("meal_H", carbs_mg_kgmin)
            ctx.log(u=self.carbs, prev_u=meal_prev, new_u=ctx.get_input("meal_H"))
            self.last_treatment_k = ctx.k
