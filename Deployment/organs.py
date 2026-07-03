from patient import PatientProfile

class OrganCompartment:
    def __init__(self, name: str, volume_l: float, vascular_flow_l_hr: float):
        self.name = name
        self.volume = volume_l
        self.flow = vascular_flow_l_hr

    def compute_passive_flux(self, concentration_matrix_a: float, concentration_matrix_b: float, permeability_surface_area: float) -> float:
        return permeability_surface_area * (concentration_matrix_a - concentration_matrix_b)


class PlacentalBarrier(OrganCompartment):
    def __init__(self, volume_l: float, vascular_flow_l_hr: float):
        super().__init__("Placenta", volume_l, vascular_flow_l_hr)

    def compute_bidirectional_transfer(self, maternal_c: float, fetal_c: float, ai_placenta_risk: float) -> tuple:
        
       
        ps_maternal_to_placenta = 2.0 / (1.0 + (ai_placenta_risk * 0.05))
        ps_placenta_to_fetal = 1.2 / (1.0 + (ai_placenta_risk * 0.15))
        
        forward_flux = ps_maternal_to_placenta * (maternal_c - fetal_c)
        reverse_flux = ps_placenta_to_fetal * (fetal_c - maternal_c)
        
        return forward_flux, reverse_flux