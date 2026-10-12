from __future__ import annotations

from io import BytesIO
from typing import Dict, Optional

import numpy as np
import pandas as pd


class ClinicalReportGenerator:
    @staticmethod
    def calculate_pk_metrics(df: pd.DataFrame) -> dict:
        time_hours = df["current_time"].to_numpy()
        fetal_conc = df["fetal_blood_conc"].to_numpy()
        auc_fetal = float(np.trapz(fetal_conc, x=time_hours))
        c_max = float(np.max(fetal_conc))
        t_max = float(time_hours[np.argmax(fetal_conc)])
        avg_tox = float(np.mean(df["toxicity_index"].to_numpy()))
        peak_tox = float(np.max(df["toxicity_index"].to_numpy()))
        peak_resistance = float(np.max(df["resistance_score"].to_numpy()))
        terminal = float(fetal_conc[-1])
        return {
            "AUC_Fetal_Circulation_mg_hr_L": round(auc_fetal, 3),
            "Fetal_Cmax_mg_L": round(c_max, 3),
            "Fetal_Tmax_Hours": round(t_max, 3),
            "Fetal_Terminal_Concentration_mg_L": round(terminal, 3),
            "Mean_Toxicity_Index": round(avg_tox, 3),
            "Peak_Toxicity_Index": round(peak_tox, 3),
            "Peak_Resistance_Score": round(peak_resistance, 3),
        }

    @staticmethod
    def _format_shap_section(shap_results: Optional[Dict[str, dict]]) -> str:
        if not shap_results:
            return "SHAP EXPLAINABILITY\n----------------------------------------------------------------------\nNot calculated for this run.\n"
        lines = ["SHAP EXPLAINABILITY", "----------------------------------------------------------------------"]
        for organ in ("liver", "placenta", "kidney"):
            res = shap_results.get(organ, {})
            lines.append(f"\n{organ.upper()}")
            if not res.get("available"):
                lines.append(f"  Unavailable: {res.get('reason', 'unknown reason')}")
                continue
            df = res.get("top_features")
            if df is None or len(df) == 0:
                lines.append("  No local SHAP features returned.")
                continue
            for _, row in df.head(8).iterrows():
                arrow = "+" if row["shap_value"] >= 0 else "-"
                lines.append(f"  {row['feature']:<34} {arrow}{abs(float(row['shap_value'])):.4f}")
        lines.append("\nInterpretation: SHAP values explain the Stage-2 model output for this molecule; they are not causal effects.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_sensitivity_section(sensitivity_df: Optional[pd.DataFrame]) -> str:
        if sensitivity_df is None or sensitivity_df.empty:
            return "SENSITIVITY ANALYSIS\n----------------------------------------------------------------------\nNot calculated for this run.\n"
        ranked = sensitivity_df.sort_values("auc_effect_percent_abs", ascending=False).head(8)
        lines = ["SENSITIVITY ANALYSIS", "----------------------------------------------------------------------"]
        for _, row in ranked.iterrows():
            lines.append(
                f"{row['parameter']:<30} AUC effect={row['auc_effect_percent']:+.2f}%  "
                f"Cmax effect={row['cmax_effect_percent']:+.2f}%"
            )
        lines.append("Interpretation: one-at-a-time perturbation of simulator inputs; this is separate from SHAP.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def export_text_summary(metrics: dict, compound_name: str, organ_predictions=None,
                            patient=None, dosing=None, shap_results=None,
                            sensitivity_df: Optional[pd.DataFrame] = None) -> str:
        pred = organ_predictions or {}
        s1 = pred.get("stage1", {})
        s1int = pred.get("stage1_intervals_90", {})
        patient_lines = ""
        if patient is not None:
            patient_lines = f"""
PERSONALIZED / DEVELOPMENTAL INPUTS
----------------------------------------------------------------------
Gestational age      : {patient.gestational_age:.1f} weeks
CYP3A7 expression    : {patient.cyp3a7_expression:.3f}
Fetal liver volume   : {patient.v_liver:.3f} L
Maternal liver volume: {patient.v_maternal_liver:.3f} L
Placenta volume      : {patient.v_placenta:.3f} L
Hepatic blood flow   : {patient.q_liver:.3f} L/hr
Placental blood flow : {patient.q_placenta:.3f} L/hr
Kidney maturation    : {patient.kidney_maturation:.3f}
Unbound fraction     : {patient.fu_maternal:.3f}
Vmax                 : {patient.vmax:.3f} mg/hr
Km (entered)          : {patient.km_micromolar:.3f} uM
Km (ODE converted)    : {patient.km_mg_l:.6f} mg/L
Molecular weight      : {patient.molecular_weight_g_mol:.3f} g/mol
Transporter modifier : {patient.transporter_modifier:.3f}
"""
        dosing_lines = ""
        if dosing is not None:
            dosing_lines = f"""
DOSING / SIMULATION SETUP
----------------------------------------------------------------------
Initial dose          : {dosing.initial_dose_mg:.3f} mg
Repeated dose         : {dosing.repeated_dose_mg:.3f} mg
Dose interval         : {dosing.interval_hrs:.3f} hr
Repeated dosing       : {bool(dosing.repeated)}
"""
        interval_lines = []
        for organ in ("placenta", "kidney"):
            if organ in s1int:
                lo, hi = s1int[organ]
                interval_lines.append(f"{organ.capitalize()} Stage-1 90% interval: [{lo:.4f}, {hi:.4f}]")
        interval_text = "\n".join(interval_lines) if interval_lines else "No calibrated Stage-2 prediction interval is available."

        return f"""
======================================================================
         FETAL DIGITAL TWIN RESEARCH SIMULATION REPORT
======================================================================
Target compound : {compound_name}
Model source    : {pred.get('source', 'not recorded')}

STAGE-2 ORGAN PREDICTIONS
----------------------------------------------------------------------
Liver CYP3A7        : {pred.get('liver', float('nan')):.4f}
Placental transfer  : {pred.get('placenta', float('nan')):.4f}
Kidney OAT1         : {pred.get('kidney', float('nan')):.4f}

STAGE-1 REFERENCE / UNCERTAINTY
----------------------------------------------------------------------
Liver Stage-1       : {s1.get('liver', float('nan')):.4f}
Placenta Stage-1    : {s1.get('placenta', float('nan')):.4f}
Kidney Stage-1      : {s1.get('kidney', float('nan')):.4f}
{interval_text}

{patient_lines}{dosing_lines}
SIMULATED PK/PD SUMMARY
----------------------------------------------------------------------
Fetal AUC            : {metrics['AUC_Fetal_Circulation_mg_hr_L']} mg*hr/L
Fetal Cmax           : {metrics['Fetal_Cmax_mg_L']} mg/L
Fetal Tmax           : {metrics['Fetal_Tmax_Hours']} hr
Terminal fetal conc. : {metrics['Fetal_Terminal_Concentration_mg_L']} mg/L
Mean toxicity index  : {metrics['Mean_Toxicity_Index']}
Peak toxicity index  : {metrics['Peak_Toxicity_Index']}
Peak resistance      : {metrics['Peak_Resistance_Score']}

{ClinicalReportGenerator._format_shap_section(shap_results)}
{ClinicalReportGenerator._format_sensitivity_section(sensitivity_df)}
PLS THIS USE FOR RESEARCH ONLY: Our Modeling has
not been clinically calibrated, externally validated for dosing, or validated.
======================================================================
"""

    @staticmethod
    def export_pdf_bytes(report_text: str) -> bytes:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        x, y = 42, height - 42
        c.setFont("Courier", 8.5)
        for line in report_text.splitlines():
            if y < 42:
                c.showPage()
                c.setFont("Courier", 8.5)
                y = height - 42
            
            chunks = [line[i:i+92] for i in range(0, max(len(line), 1), 92)] or [""]
            for chunk in chunks:
                c.drawString(x, y, chunk)
                y -= 10
        c.save()
        return buf.getvalue()
