from __future__ import annotations

import copy
import numpy as np
import pandas as pd

import config
from ai_prediction import AIPredictionPipeline
from dosing import DosingRegimen
from patient import PatientProfile
from pkpd import PKPDNumericalEngine
from state import DigitalTwinState, SimulationHistory


class FetalDigitalTwinEngine:
    def __init__(self, smiles: str, patient: PatientProfile, dosing: DosingRegimen, ai_pipeline=None, organ_predictions=None):
        self.smiles = smiles
        self.patient = patient
        self.dosing = dosing
        self.ai_pipeline = ai_pipeline if ai_pipeline is not None else AIPredictionPipeline(device="cpu")
        self.organ_predictions = organ_predictions if organ_predictions is not None else self.ai_pipeline.predict_organ_risks(self.smiles)
        self.state = None
        self.reset_state()

    def reset_state(self):
        self.state = DigitalTwinState()
        self.state.liver_risk = float(self.organ_predictions["liver"])
        self.state.kidney_risk = float(self.organ_predictions["kidney"])
        self.state.placenta_risk = float(self.organ_predictions["placenta"])
        self.state.cyp3a7_expression = self.patient.cyp3a7_expression
        self.state.current_time = 0.0

    def run_simulation(self, total_duration_hrs: float, dt: float = 0.10, reset: bool = True) -> pd.DataFrame:
        if reset:
            self.reset_state()
        history = SimulationHistory()
        if self.dosing.initial_dose_mg > 0.0:
            self.state.maternal_conc += self.dosing.initial_dose_mg / config.MATERNAL_BLOOD_VOLUME_L
        history.log_step(self.state.current_time, self.state)
        while self.state.current_time < total_duration_hrs - 1e-12:
            step = min(dt, total_duration_hrs - self.state.current_time)
            dose_mg = 0.0 if self.state.current_time < 1e-12 else self.dosing.get_dose_at_time(self.state.current_time, step)
            if dose_mg > 0.0:
                self.state.maternal_conc += dose_mg / config.MATERNAL_BLOOD_VOLUME_L
            self.state = PKPDNumericalEngine.advance_state_adaptive(
                self.state, self.patient, 0.0, self.organ_predictions, step
            )
            history.log_step(self.state.current_time, self.state)
        return history.to_dataframe()

    def run_monte_carlo_uncertainty(self, total_duration_hrs: float, iterations: int = 40, dt: float = 0.2) -> dict:
        rng = np.random.default_rng(42)
        curves = []
        base_liver_flow = self.patient.fetal_liver_flow
        base_kidney_flow = self.patient.fetal_kidney_flow
        time = None
        try:
            for _ in range(iterations):
                self.patient.fetal_liver_flow = max(1e-6, rng.normal(base_liver_flow, abs(base_liver_flow) * config.POPULATION_FLOW_VARIANCE_COV))
                self.patient.fetal_kidney_flow = max(1e-6, rng.normal(base_kidney_flow, abs(base_kidney_flow) * config.POPULATION_FLOW_VARIANCE_COV))
                df = self.run_simulation(total_duration_hrs, dt=dt, reset=True)
                time = df["current_time"].to_numpy()
                curves.append(df["fetal_blood_conc"].to_numpy())
        finally:
            self.patient.fetal_liver_flow = base_liver_flow
            self.patient.fetal_kidney_flow = base_kidney_flow
            self.reset_state()

        matrix = np.asarray(curves)
        return {
            "time": time,
            "mean": np.mean(matrix, axis=0),
            "lower_95ci": np.percentile(matrix, 2.5, axis=0),
            "upper_95ci": np.percentile(matrix, 97.5, axis=0),
        }
