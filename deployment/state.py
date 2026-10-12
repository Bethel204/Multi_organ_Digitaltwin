import dataclasses
from typing import Dict, List, Any
import pandas as pd

@dataclasses.dataclass
class DigitalTwinState:
    current_time: float = 0.0
    
   
    maternal_conc: float = 0.0
    maternal_liver_conc: float = 0.0
    placenta_conc: float = 0.0
    fetal_blood_conc: float = 0.0
    fetal_liver_conc: float = 0.0
    fetal_kidney_conc: float = 0.0
    metabolite_amount_mg: float = 0.0
    
   
    cyp3a7_expression: float = 1.0
    resistance_score: float = 0.0
    toxicity_index: float = 0.0
    
    
    liver_risk: float = 0.0
    placenta_risk: float = 0.0
    kidney_risk: float = 0.0

    def to_array(self) -> list:
        return [
            self.maternal_conc, self.maternal_liver_conc, self.placenta_conc,
            self.fetal_blood_conc, self.fetal_liver_conc, self.fetal_kidney_conc,
            self.metabolite_amount_mg, self.cyp3a7_expression, self.resistance_score
        ]

    def update_from_array(self, arr: list):
        self.maternal_conc = max(0.0, arr[0])
        self.maternal_liver_conc = max(0.0, arr[1])
        self.placenta_conc = max(0.0, arr[2])
        self.fetal_blood_conc = max(0.0, arr[3])
        self.fetal_liver_conc = max(0.0, arr[4])
        self.fetal_kidney_conc = max(0.0, arr[5])
        self.metabolite_amount_mg = max(0.0, arr[6])
        self.cyp3a7_expression = max(0.1, arr[7])
        self.resistance_score = max(0.0, arr[8])

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)




class SimulationHistory:
    def __init__(self):
        self.records = []

    def log_step(self, time: float, state: DigitalTwinState):
        """Captures every single organ compartment concentration and tracker at the current timestamp."""
        self.records.append({
            "current_time": time,
            "maternal_conc": state.maternal_conc,
            "maternal_liver_conc": state.maternal_liver_conc,
            "placenta_conc": state.placenta_conc,
            "fetal_blood_conc": state.fetal_blood_conc,
            "fetal_liver_conc": state.fetal_liver_conc,
            "fetal_kidney_conc": state.fetal_kidney_conc,
            "metabolite_amount_mg": state.metabolite_amount_mg,
            "cyp3a7_expression": state.cyp3a7_expression,
            "resistance_score": state.resistance_score, 
            "toxicity_index": getattr(state, "toxicity_index", 0.0)
        })

    def to_dataframe(self) -> pd.DataFrame:
        
        return pd.DataFrame(self.records)