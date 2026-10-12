import math


class DosingRegimen:
    def __init__(self, initial_dose_mg: float, repeated_dose_mg: float = 0.0, interval_hrs: float = 8.0, repeated: bool = False):
        self.initial_dose_mg = float(initial_dose_mg)
        self.repeated_dose_mg = float(repeated_dose_mg)
        self.interval_hrs = float(interval_hrs)
        self.repeated = bool(repeated)
        if self.interval_hrs <= 0:
            raise ValueError("Dose interval must be positive.")

    def get_dose_at_time(self, t: float, dt: float) -> float:
        t = float(t)
        dt = float(dt)
        if abs(t) < 1e-12:
            return self.initial_dose_mg
        if not self.repeated or self.repeated_dose_mg <= 0:
            return 0.0
        n = max(1, int(math.ceil((t - 1e-12) / self.interval_hrs)))
        scheduled = n * self.interval_hrs
        if t - 1e-12 <= scheduled < t + dt - 1e-12:
            return self.repeated_dose_mg
        return 0.0
