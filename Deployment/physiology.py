import numpy as np

class PhysiologyController:
    @staticmethod
    def calculate_nonlinear_toxicity(fetal_conc: float, c50: float = 25.0, k: float = 0.2) -> float:
        return 10.0 / (1.0 + np.exp(-k * (fetal_conc - c50)))

    @staticmethod
    def evaluate_ode_derivatives(state_arr: list, v_max: float, k_m: float, 
                                 q_h: float, q_p: float, cl_r: float, 
                                 ps: float, ps_f: float, q_f: float, q_k: float, cl_k: float,
                                 k_syn: float, k_deg: float, k_ind: float,
                                 alpha: float, beta: float, gamma: float,
                                 current_dose: float) -> list:
        
        M, L, P, F, Lf, K, Met, E, R = state_arr
        
        
        v_met = (v_max * L) / (k_m + L + 1e-6)
        dM_dt = current_dose - q_h * (M - L) - q_p * (M - P) - cl_r * M
        dL_dt = q_h * (M - L) - v_met
        
        
        dP_dt = ps * (M - P) - ps_f * (P - F)
        
        
        v_cyp3a7 = (v_max * 0.4 * Lf * E) / (k_m + Lf + 1e-6)
        dF_dt = ps_f * (P - F) - q_f * (F - Lf) - q_k * (F - K)
        dLf_dt = q_f * (F - Lf) - v_cyp3a7
        dK_dt = q_k * (F - K) - cl_k * K
        
        dMet_dt = v_met + v_cyp3a7
        
        
        tox = PhysiologyController.calculate_nonlinear_toxicity(F)
        dE_dt = k_syn - k_deg * E + k_ind * Lf
        dR_dt = alpha * tox + beta * E - gamma * R
        
        return [dM_dt, dL_dt, dP_dt, dF_dt, dLf_dt, dK_dt, dMet_dt, dE_dt, dR_dt]