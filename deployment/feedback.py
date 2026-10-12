from state import DigitalTwinState

class DataAssimilationFeedback:
    @staticmethod
    def assimilate_observations(state: DigitalTwinState, obs_dict: dict, variance_weight: float = 0.3) -> DigitalTwinState:
      
        if not obs_dict:
            return state

       
        if "observed_fetal_blood_conc" in obs_dict and obs_dict["observed_fetal_blood_conc"] is not None:
            residual = obs_dict["observed_fetal_blood_conc"] - state.fetal_blood_conc
            state.fetal_blood_conc += residual * variance_weight

        if "observed_cyp3a7_expression" in obs_dict and obs_dict["observed_cyp3a7_expression"] is not None:
            residual_cyp = obs_dict["observed_cyp3a7_expression"] - state.cyp3a7_expression
            state.cyp3a7_expression += residual_cyp * variance_weight

        return state