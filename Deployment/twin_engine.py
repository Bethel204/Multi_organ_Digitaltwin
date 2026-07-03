import numpy as np
import copy
import pandas as pd
from state import DigitalTwinState, SimulationHistory
from patient import PatientProfile
from dosing import DosingRegimen
from pkpd import PKPDNumericalEngine
from ai_prediction import AIPredictionPipeline

class FetalDigitalTwinEngine:
    def __init__(self, smiles: str, patient: PatientProfile, dosing: DosingRegimen):
        self.smiles = smiles
        self.patient = patient
        self.dosing = dosing
        self.ai_pipeline = AIPredictionPipeline()
        self.state = None
        self.history = None
        self.initialize_twin_state()

    def initialize_twin_state(self):
        risks = self.ai_pipeline.predict_organ_risks(self.smiles)
        self.state = DigitalTwinState()
        self.state.liver_risk = float(risks["liver"])
        self.state.kidney_risk = float(risks["kidney"])
        self.state.placenta_risk = float(risks["placenta"])
        self.state.cyp3a7_expression = self.patient.cyp3a7_expression
        self.state.current_time = 0.0
        self.history = SimulationHistory()

    def run_simulation(self, total_duration_hrs: float, dt: float = 0.05) -> pd.DataFrame:
        
        from state import SimulationHistory
        
        history = SimulationHistory()
        current_time = 0.0
        
        
        history.log_step(current_time, self.state)
        
       
        while current_time < total_duration_hrs:
           
            dose = self.dosing.get_dose_at_time(current_time, dt)
            
            
            ai_modifiers = self.ai_pipeline.predict_organ_risks(self.smiles)
            
           
            self.state = PKPDNumericalEngine.advance_state_adaptive(
                self.state, self.patient, dose, ai_modifiers, dt
            )
            
            current_time += dt
           
            history.log_step(current_time, self.state)
            
        return history.to_dataframe()

    def run_monte_carlo_uncertainty(self, total_duration_hrs: float, iterations: int = 100, dt: float = 0.5) -> dict:
       
        rng = np.random.default_rng(42)
        all_curves = []

        base_liver_flow = self.patient.fetal_liver_flow
        base_kidney_flow = self.patient.fetal_kidney_flow

        for _ in range(iterations):
            
            self.patient.fetal_liver_flow = rng.normal(base_liver_flow, base_liver_flow * 0.15)
            self.patient.fetal_kidney_flow = rng.normal(base_kidney_flow, base_kidney_flow * 0.15)
            
            df = self.run_simulation(total_duration_hrs, dt=dt)
            all_curves.append(df["fetal_blood_conc"].values)

       
        self.patient.fetal_liver_flow = base_liver_flow
        self.patient.fetal_kidney_flow = base_kidney_flow

        curves_matrix = np.array(all_curves)
        return {
            "time": df["current_time"].values,
            "mean": np.mean(curves_matrix, axis=0),
            "lower_95ci": np.percentile(curves_matrix, 2.5, axis=0),
            "upper_95ci": np.percentile(curves_matrix, 97.5, axis=0)
        }