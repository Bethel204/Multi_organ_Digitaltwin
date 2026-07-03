class DosingRegimen:
    def __init__(self, initial_dose_mg: float, repeated_dose_mg: float = 0.0, interval_hrs: float = 8.0, repeated: bool = False):
        self.initial_dose_mg = initial_dose_mg
        self.repeated_dose_mg = repeated_dose_mg
        self.interval_hrs = interval_hrs
        self.repeated = repeated

    def get_dose_at_time(self, t: float, dt: float) -> float:
        
        if t == 0:
            return self.initial_dose_mg
        if self.repeated and t > 0:
            
            if abs((t % self.interval_hrs)) < (dt / 2.0):
                return self.repeated_dose_mg
        return 0.0