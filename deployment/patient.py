import config


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
        transporter_activity_coefficient: float,
        molecular_weight_g_mol: float,
        maternal_liver_volume_l: float = 1.5,
        placenta_volume_l: float = 0.25,
        maternal_liver_kp: float = config.TISSUE_KP_MATERNAL_LIVER,
        placenta_kp: float = config.TISSUE_KP_PLACENTA,
        fetal_liver_kp: float = config.TISSUE_KP_FETAL_LIVER,
        fetal_kidney_kp: float = config.TISSUE_KP_FETAL_KIDNEY,
    ):
        self.gestational_age = float(gestational_age_weeks)
        self.cyp3a7_expression = float(cyp3a7_expression)
        self.v_liver = float(liver_volume_l)
        self.q_placenta = float(placental_blood_flow_l_hr)
        self.kidney_maturation = float(kidney_maturation_factor)
        self.fu_maternal = 1.0 - float(plasma_protein_binding_fraction)
        self.q_liver = float(hepatic_blood_flow_l_hr)
        self.km_micromolar = float(km_micromolar)
        self.vmax = float(vmax_mg_hr)
        self.transporter_modifier = float(transporter_activity_coefficient)
        self.molecular_weight_g_mol = float(molecular_weight_g_mol)
        self.v_maternal_liver = float(maternal_liver_volume_l)
        self.v_placenta = float(placenta_volume_l)
        self.kp_maternal_liver = float(maternal_liver_kp)
        self.kp_placenta = float(placenta_kp)
        self.kp_fetal_liver = float(fetal_liver_kp)
        self.kp_fetal_kidney = float(fetal_kidney_kp)

        if self.molecular_weight_g_mol <= 0:
            raise ValueError("Molecular weight must be positive.")
        if min(self.v_liver, self.v_maternal_liver, self.v_placenta) <= 0:
            raise ValueError("Compartment volumes must be positive.")
        if min(self.kp_maternal_liver, self.kp_placenta, self.kp_fetal_liver, self.kp_fetal_kidney) <= 0:
            raise ValueError("Tissue:plasma partition coefficients must be positive.")
        if not 0.0 <= self.fu_maternal <= 1.0:
            raise ValueError("Plasma protein binding fraction must be between 0 and 1.")

        self.km_mg_l = self.km_micromolar * self.molecular_weight_g_mol / 1000.0
        self.km = self.km_mg_l
        self.v_fetal_blood = float(0.005 * (self.gestational_age ** 2.4))
        self.v_fetal_kidney = max(1e-6, self.v_fetal_blood * 0.12 * self.kidney_maturation)
        self.fetal_kidney_flow = 0.45 * self.v_fetal_kidney
        self.fetal_liver_flow = 0.60 * self.v_liver

    def get_unbound_maternal_fraction(self) -> float:
        return self.fu_maternal
