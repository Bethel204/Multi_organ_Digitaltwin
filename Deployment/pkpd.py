import numpy as np
from scipy.integrate import solve_ivp
from physiology import PhysiologyController

class PKPDNumericalEngine:
    @staticmethod
    def advance_state_adaptive(state, patient, current_dose, ai_modifiers, dt: float):
       
       
        liver_risk = ai_modifiers.get("liver_risk", 5.0)
        placenta_risk = ai_modifiers.get("placenta_risk", 5.0)
        kidney_risk = ai_modifiers.get("kidney_risk", 5.0)

        
        v_max = 5.0 / (1.0 + (liver_risk * 0.1))
        k_m = 2.0 + (liver_risk * 0.05)
        ps_f = 1.2 / (1.0 + (placenta_risk * 0.15))
        cl_k = 0.5 / (1.0 + (kidney_risk * 0.2))

        
        q_h = 90.0
        q_p = 45.0
        cl_r = 7.2
        ps = 2.0
        q_f = patient.fetal_liver_flow
        q_k = patient.fetal_kidney_flow

        
        k_syn, k_deg, k_ind = 0.1, 0.1, 0.02
        alpha, beta, gamma = 0.4, 0.2, 0.1

        y0 = state.to_array()
        t_span = (state.current_time, state.current_time + dt)

        def ode_wrapper(t, y):
            return PhysiologyController.evaluate_ode_derivatives(
                y, v_max, k_m, q_h, q_p, cl_r, ps, ps_f, q_f, q_k, cl_k,
                k_syn, k_deg, k_ind, alpha, beta, gamma, current_dose
            )

        sol = solve_ivp(ode_wrapper, t_span, y0, method='RK45', rtol=1e-5, atol=1e-8)
        
       
        final_y = sol.y[:, -1]
        state.update_from_array(final_y)
        state.toxicity_index = PhysiologyController.calculate_nonlinear_toxicity(state.fetal_blood_conc)
        
        return state