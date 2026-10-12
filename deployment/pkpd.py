from __future__ import annotations

import numpy as np

import config
from physiology import PhysiologyController


def _flow_permeability_limited_clearance(flow_l_hr: float, permeability_clearance_l_hr: float) -> float:
    q = max(float(flow_l_hr), 1e-12)
    ps = max(float(permeability_clearance_l_hr), 1e-12)
    return q * ps / (q + ps)


def _negative_log_molar_to_mg_l(score: float, molecular_weight_g_mol: float) -> float:
    molar = 10.0 ** (-float(score))
    return max(1e-12, molar * float(molecular_weight_g_mol) * 1000.0)


def _placenta_transfer_multiplier(score: float) -> float:
    if not config.AI_PLACENTA_COUPLING_ENABLED:
        return 1.0
    delta = float(score) - float(config.AI_PLACENTA_REFERENCE_SCORE)
    multiplier = np.exp(float(config.AI_PLACENTA_COUPLING_STRENGTH) * delta)
    return float(np.clip(multiplier, config.AI_PLACENTA_MIN_MULTIPLIER, config.AI_PLACENTA_MAX_MULTIPLIER))


class PKPDNumericalEngine:
    @staticmethod
    def advance_state_adaptive(state, patient, current_dose_rate_mg_hr, ai_modifiers, dt: float):
        liver_score = float(ai_modifiers.get("liver", ai_modifiers.get("liver_risk", 5.0)))
        placenta_score = float(ai_modifiers.get("placenta", ai_modifiers.get("placenta_risk", config.AI_PLACENTA_REFERENCE_SCORE)))
        kidney_score = float(ai_modifiers.get("kidney", ai_modifiers.get("kidney_risk", 5.0)))

        liver_halfmax_mg_l = _negative_log_molar_to_mg_l(liver_score, patient.molecular_weight_g_mol)
        kidney_halfmax_mg_l = _negative_log_molar_to_mg_l(kidney_score, patient.molecular_weight_g_mol)

        maternal_vmax_mg_hr = max(1e-6, patient.vmax)
        fetal_vmax_mg_hr = max(1e-6, patient.vmax)
        km_mg_l = max(1e-9, patient.km_mg_l)

        q_h_l_hr = max(1e-6, patient.q_liver)
        q_p_l_hr = max(1e-6, patient.q_placenta)
        q_f_l_hr = max(1e-6, patient.fetal_liver_flow)
        q_k_l_hr = max(1e-6, patient.fetal_kidney_flow)

        maternal_renal_clearance_l_hr = max(1e-6, config.MATERNAL_RENAL_CLEARANCE_BASE_L_HR)
        placenta_ai_multiplier = _placenta_transfer_multiplier(placenta_score)
        maternal_ps_l_hr = max(1e-6, config.MATERNAL_PLACENTA_PS_BASE_L_HR * patient.transporter_modifier * patient.get_unbound_maternal_fraction() * placenta_ai_multiplier)
        maternal_placenta_clearance_l_hr = _flow_permeability_limited_clearance(q_p_l_hr, maternal_ps_l_hr)
        placenta_fetal_clearance_l_hr = max(1e-6, config.PLACENTA_FETAL_PS_BASE_L_HR * patient.transporter_modifier * placenta_ai_multiplier)
        fetal_renal_clearance_l_hr = max(1e-6, config.FETAL_RENAL_CLEARANCE_BASE_L_HR * patient.kidney_maturation)

        y0 = state.to_array()
        t_span = (state.current_time, state.current_time + dt)

        def ode_wrapper(t, y):
            return PhysiologyController.evaluate_ode_derivatives(
                y,
                maternal_vmax_mg_hr,
                fetal_vmax_mg_hr,
                km_mg_l,
                config.MATERNAL_BLOOD_VOLUME_L,
                patient.v_maternal_liver,
                patient.v_placenta,
                patient.v_fetal_blood,
                patient.v_liver,
                patient.v_fetal_kidney,
                q_h_l_hr,
                maternal_placenta_clearance_l_hr,
                placenta_fetal_clearance_l_hr,
                q_f_l_hr,
                q_k_l_hr,
                maternal_renal_clearance_l_hr,
                fetal_renal_clearance_l_hr,
                patient.kp_maternal_liver,
                patient.kp_placenta,
                patient.kp_fetal_liver,
                patient.kp_fetal_kidney,
                liver_halfmax_mg_l,
                kidney_halfmax_mg_l,
                patient.cyp3a7_expression,
                max(0.0, float(current_dose_rate_mg_hr)),
            )

        from scipy.integrate import solve_ivp
        sol = solve_ivp(ode_wrapper, t_span, y0, method="RK45", rtol=1e-5, atol=1e-8)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        state.update_from_array(sol.y[:, -1])
        state.current_time += dt
        state.toxicity_index = PhysiologyController.calculate_nonlinear_toxicity(state.fetal_blood_conc)
        return state
