import numpy as np
import config


class PhysiologyController:
    @staticmethod
    def hill_emax(concentration_mg_l: float, emax: float, ec50_mg_l: float, hill: float = 1.0) -> float:
        c = max(0.0, float(concentration_mg_l))
        ec50 = max(1e-12, float(ec50_mg_l))
        n = max(1e-6, float(hill))
        cn = c ** n
        return float(emax * cn / (ec50 ** n + cn))

    @staticmethod
    def calculate_nonlinear_toxicity(fetal_conc_mg_l: float) -> float:
        return PhysiologyController.hill_emax(
            fetal_conc_mg_l,
            config.TOXICITY_EMAX,
            config.TOXICITY_EC50_MG_L,
            config.TOXICITY_HILL,
        )

    @staticmethod
    def remaining_activity(concentration_mg_l: float, halfmax_mg_l: float, imax: float = 1.0, hill: float = 1.0) -> float:
        inhibition = PhysiologyController.hill_emax(concentration_mg_l, imax, halfmax_mg_l, hill)
        return float(np.clip(1.0 - inhibition, 0.0, 1.0))

    @staticmethod
    def evaluate_ode_derivatives(
        state_arr,
        maternal_vmax_mg_hr,
        fetal_vmax_mg_hr,
        km_mg_l,
        maternal_blood_volume_l,
        maternal_liver_volume_l,
        placenta_volume_l,
        fetal_blood_volume_l,
        fetal_liver_volume_l,
        fetal_kidney_volume_l,
        q_h_l_hr,
        maternal_placenta_clearance_l_hr,
        placenta_fetal_clearance_l_hr,
        q_f_l_hr,
        q_k_l_hr,
        maternal_renal_clearance_l_hr,
        fetal_renal_clearance_l_hr,
        kp_maternal_liver,
        kp_placenta,
        kp_fetal_liver,
        kp_fetal_kidney,
        liver_halfmax_mg_l,
        kidney_halfmax_mg_l,
        cyp_baseline_expression,
        dose_rate_mg_hr,
    ):
        M, L, P, F, Lf, K, Met, E, R = state_arr

        liver_remaining = PhysiologyController.remaining_activity(Lf, liver_halfmax_mg_l, config.AI_INHIBITION_IMAX, config.AI_INHIBITION_HILL)
        kidney_remaining = PhysiologyController.remaining_activity(K, kidney_halfmax_mg_l, config.AI_INHIBITION_IMAX, config.AI_INHIBITION_HILL)

        maternal_metabolism_mg_hr = (maternal_vmax_mg_hr * L) / (km_mg_l + L + 1e-12)
        fetal_metabolism_mg_hr = (fetal_vmax_mg_hr * Lf * E * liver_remaining) / (km_mg_l + Lf + 1e-12)

        hepatic_exchange_mg_hr = q_h_l_hr * (M - L / kp_maternal_liver)
        maternal_placenta_flux_mg_hr = maternal_placenta_clearance_l_hr * (M - P / kp_placenta)
        placenta_fetal_flux_mg_hr = placenta_fetal_clearance_l_hr * (P / kp_placenta - F)
        fetal_liver_exchange_mg_hr = q_f_l_hr * (F - Lf / kp_fetal_liver)
        fetal_kidney_exchange_mg_hr = q_k_l_hr * (F - K / kp_fetal_kidney)
        maternal_renal_elimination_mg_hr = maternal_renal_clearance_l_hr * M
        fetal_renal_elimination_mg_hr = fetal_renal_clearance_l_hr * kidney_remaining * K

        dM_dt = (dose_rate_mg_hr - hepatic_exchange_mg_hr - maternal_placenta_flux_mg_hr - maternal_renal_elimination_mg_hr) / maternal_blood_volume_l
        dL_dt = (hepatic_exchange_mg_hr - maternal_metabolism_mg_hr) / maternal_liver_volume_l
        dP_dt = (maternal_placenta_flux_mg_hr - placenta_fetal_flux_mg_hr) / placenta_volume_l
        dF_dt = (placenta_fetal_flux_mg_hr - fetal_liver_exchange_mg_hr - fetal_kidney_exchange_mg_hr) / fetal_blood_volume_l
        dLf_dt = (fetal_liver_exchange_mg_hr - fetal_metabolism_mg_hr) / fetal_liver_volume_l
        dK_dt = (fetal_kidney_exchange_mg_hr - fetal_renal_elimination_mg_hr) / fetal_kidney_volume_l
        dMet_dt = maternal_metabolism_mg_hr + fetal_metabolism_mg_hr

        cyp_induction = PhysiologyController.hill_emax(Lf, config.CYP_INDUCTION_EMAX, config.CYP_INDUCTION_EC50_MG_L, config.CYP_INDUCTION_HILL)
        synthesis = config.CYP_DEGRADATION_RATE_HR * cyp_baseline_expression * (1.0 + cyp_induction)
        dE_dt = synthesis - config.CYP_DEGRADATION_RATE_HR * E

        tox = PhysiologyController.calculate_nonlinear_toxicity(F)
        resistance_stimulus = PhysiologyController.hill_emax(F, 1.0, config.RESISTANCE_EC50_MG_L, config.RESISTANCE_HILL)
        dR_dt = config.RESISTANCE_PRODUCTION_MAX_HR * resistance_stimulus - config.RESISTANCE_DECAY_HR * R

        return [dM_dt, dL_dt, dP_dt, dF_dt, dLf_dt, dK_dt, dMet_dt, dE_dt, dR_dt]
