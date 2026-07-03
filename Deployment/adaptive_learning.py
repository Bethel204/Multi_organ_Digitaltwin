import pandas as pd
from ai_prediction import AIPredictionPipeline

class AdaptiveLearningManager:
    def __init__(self, pipeline: AIPredictionPipeline):
        self.pipeline = pipeline
        self.buffered_observations = []

    def queue_new_observation(self, smiles: str, observed_liver_clearance: float, observed_kidney_clearance: float):       
        self.buffered_observations.append({
            "SMILES": smiles,
            "Liver_Prediction_Risk": observed_liver_clearance,
            "Kidney_Prediction_Risk": observed_kidney_clearance,
            "Placenta_Prediction_Risk": 1.0
        })

    def trigger_model_alignment(self) -> bool:      
        if len(self.buffered_observations) < 1:
            return False
        
        if self.pipeline.db_predictions is not None:
            new_df = pd.DataFrame(self.buffered_observations)
            self.pipeline.db_predictions = pd.concat([self.pipeline.db_predictions, new_df], ignore_index=True)
            self.buffered_observations.clear()
            return True
        return False