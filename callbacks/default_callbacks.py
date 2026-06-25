from callbacks.callback import ReplayCallback


class CorrectionBolus(ReplayCallback):
    """Reactive correction-bolus policy with a refractory window.

    If the most recent glucose reading is above ``threshold`` and no correction
    has been issued within the last ``lockout_min`` minutes, deliver a correction
    bolus of ``(glucose - target) / cf`` units.

    Hyperparameters:
        threshold: Glucose level (mg/dL) above which a correction is considered.
        target: Target glucose level (mg/dL) used to size the correction.
        cf: Correction factor (mg/dL per unit of insulin).
        lockout_min: Minimum number of minutes between consecutive boluses.

    Memory:
        last_correction_k: Integration minute of the last issued correction.
    """

    def __init__(self, threshold=180.0, target=120.0, cf=40.0, lockout_min=60):
        self.threshold = threshold
        self.target = target
        self.cf = cf
        self.lockout_min = lockout_min
        self.last_correction_k = -10 ** 9

    def action(self, ctx):
        g = ctx.output_history[ctx.k - 1]
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

    Hyperparameters:
        threshold: Glucose level (mg/dL) below which a treatment is considered.
        carbs: Amount of rescue carbohydrates to deliver (grams).
        lockout_min: Minimum number of minutes between consecutive treatments.

    Memory:
        last_treatment_k: Integration minute of the last issued treatment.
    """

    def __init__(self, threshold=70.0, carbs=15.0, lockout_min=15):
        self.threshold = threshold
        self.carbs = carbs
        self.lockout_min = lockout_min
        self.last_treatment_k = -10 ** 9

    def action(self, ctx):
        g = ctx.output_history[ctx.k - 1]
        if g < self.threshold and (ctx.k - self.last_treatment_k) >= self.lockout_min:
            carbs_mg_kgmin = self.carbs * (1000.0 / self.rbg_data.body_weight)
            meal_prev = ctx.get_input("meal_H")
            ctx.add_input("meal_H", carbs_mg_kgmin)
            ctx.log(u=self.carbs, prev_u=meal_prev, new_u=ctx.get_input("meal_H"))
            self.last_treatment_k = ctx.k
