import numpy as np
import pandas as pd

class ClinicalReportGenerator:
    @staticmethod
    def calculate_pk_metrics(df: pd.DataFrame) -> dict:
        
        time_hours = df["current_time"].values
        fetal_conc = df["fetal_blood_conc"].values
        
        
        auc_fetal = np.trapz(fetal_conc, x=time_hours)
        c_max = float(np.max(fetal_conc))
        t_max = float(time_hours[np.argmax(fetal_conc)])
        
        avg_tox = float(np.mean(df["toxicity_index"].values))
        peak_resistance = float(np.max(df["resistance_score"].values))
        
        
        safety_status = "PASSED (Safe)"
        if c_max > 45.0 or avg_tox > 7.0:
            safety_status = "CRITICAL WARNING: HIGH FETAL EXPOSURE RISK"
            
        return {
            "AUC_Fetal_Circulation_mg_hr_L": round(auc_fetal, 2),
            "Fetal_Cmax_mg_L": round(c_max, 2),
            "Fetal_Tmax_Hours": round(t_max, 2),
            "Mean_Toxicity_Index": round(avg_tox, 2),
            "Peak_Resistance_Score": round(peak_resistance, 2),
            "Clinical_Safety_Evaluation": safety_status
        }

    @staticmethod
    def export_text_summary(metrics: dict, compound_name: str) -> str:
        """Format parameters cleanly into a structured medical validation report summary."""
        return f"""
======================================================================
         FETAL DIGITAL TWIN PHARMACOKINETIC VALIDATION REPORT
======================================================================
Target Analyte Structure : {compound_name}
Safety Evaluation Outcome: {metrics['Clinical_Safety_Evaluation']}

PRIMARY METRIC EXPLORATION:
----------------------------------------------------------------------
• Area Under Curve (AUC Fetal) : {metrics['AUC_Fetal_Circulation_mg_hr_L']} mg*hr/L
• Peak Concentration (Cmax)    : {metrics['Fetal_Cmax_mg_L']} mg/L
• Time to Peak Level (Tmax)     : {metrics['Fetal_Tmax_Hours']} Hours
• Quantified Toxicity Index    : {metrics['Mean_Toxicity_Index']} / 10.0
• Peak Cellular Resistance     : {metrics['Peak_Resistance_Score']} / 10.0
======================================================================
"""