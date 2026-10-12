class OrganCompartment:
    def __init__(self, name: str, volume_l: float, vascular_flow_l_hr: float, tissue_plasma_partition: float = 1.0):
        self.name = name
        self.volume = float(volume_l)
        self.flow = float(vascular_flow_l_hr)
        self.kp = float(tissue_plasma_partition)

    def compute_perfusion_limited_flux(self, blood_concentration: float, tissue_concentration: float) -> float:
        return self.flow * (blood_concentration - tissue_concentration / self.kp)

    def compute_passive_flux(self, concentration_a: float, concentration_b: float, permeability_surface_area_l_hr: float) -> float:
        return float(permeability_surface_area_l_hr) * (float(concentration_a) - float(concentration_b))


class PlacentalBarrier(OrganCompartment):
    def compute_bidirectional_transfer(self, maternal_c: float, placenta_c: float, fetal_c: float, ps_maternal_l_hr: float, ps_fetal_l_hr: float):
        maternal_to_placenta = self.compute_passive_flux(maternal_c, placenta_c / self.kp, ps_maternal_l_hr)
        placenta_to_fetal = self.compute_passive_flux(placenta_c / self.kp, fetal_c, ps_fetal_l_hr)
        return maternal_to_placenta, placenta_to_fetal
