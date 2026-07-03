import numpy as np

class PatientProfile:
    def __init__(
        self,
        gestational_age_weeks: float,
        cyp3a7_expression: float,
        liver_volume_l: float,
        placental_blood_flow_l_hr: float,
        kidney_maturation_factor: float,
        plasma_protein_binding_fraction: float,
        hepatic_blood_flow_l_hr: float,
        km_micromolar: float,
        vmax_mg_hr: float,
        transporter_activity_coefficient: float
    ):
       
        self.gestational_age = gestational_age_weeks
        self.cyp3a7_expression = cyp3a7_expression
        self.v_liver = liver_volume_l
        self.q_placenta = placental_blood_flow_l_hr
        self.kidney_maturation = kidney_maturation_factor
        self.fu_maternal = 1.0 - plasma_protein_binding_fraction
        self.q_liver = hepatic_blood_flow_l_hr
        
       
        self.km = km_micromolar
        self.vmax = vmax_mg_hr
        self.transporter_modifier = transporter_activity_coefficient
        
       
        self.v_fetal_blood = float(0.005 * (self.gestational_age ** 2.4)) 
        self.v_fetal_kidney = self.v_fetal_blood * 0.12 * self.kidney_maturation
        
        
        self.fetal_kidney_flow = 0.45 * self.v_fetal_kidney
        self.fetal_liver_flow = 0.60 * self.v_liver

    def get_unbound_maternal_fraction(self) -> float:
        return self.fu_maternal