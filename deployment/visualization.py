from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class PublicationVisualizer:
    @staticmethod
    def generate_concentration_curves(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        traces = [
            ("maternal_conc", "Maternal plasma", "solid"),
            ("placenta_conc", "Placenta", "dot"),
            ("fetal_blood_conc", "Fetal circulation", "solid"),
            ("fetal_liver_conc", "Fetal liver", "dash"),
            ("fetal_kidney_conc", "Fetal kidney", "dashdot"),
        ]
        for col, name, dash in traces:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df["current_time"], y=df[col], name=name, line=dict(width=2.6, dash=dash)))
        fig.update_layout(
            title="Concentration–time profiles across simulated compartments",
            xaxis_title="Time (hours)", yaxis_title="Concentration (mg/L)",
            template="plotly_white", legend=dict(orientation="h", y=1.12), height=500,
        )
        return fig

    @staticmethod
    def generate_fetal_organ_curves(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        for col, name in [("fetal_blood_conc", "Fetal circulation"), ("fetal_liver_conc", "Fetal liver"), ("fetal_kidney_conc", "Fetal kidney")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df["current_time"], y=df[col], name=name, line=dict(width=3)))
        fig.update_layout(title="Fetal compartment exposure", xaxis_title="Time (hours)", yaxis_title="Concentration (mg/L)", template="plotly_white", height=430)
        return fig

    @staticmethod
    def generate_toxicity_resistance(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if "toxicity_index" in df.columns:
            fig.add_trace(go.Scatter(x=df["current_time"], y=df["toxicity_index"], name="Toxicity index", line=dict(width=3)))
        if "resistance_score" in df.columns:
            fig.add_trace(go.Scatter(x=df["current_time"], y=df["resistance_score"], name="Resistance score", line=dict(width=3, dash="dash")))
        fig.update_layout(title="Modeled toxicity and resistance over time", xaxis_title="Time (hours)", yaxis_title="Model index", template="plotly_white", height=420)
        return fig

    @staticmethod
    def generate_uncertainty_bounds(mc_results: dict) -> go.Figure:
        fig = go.Figure()
        time = mc_results["time"]
        fig.add_trace(go.Scatter(x=time, y=mc_results["upper_95ci"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=time, y=mc_results["lower_95ci"], fill="tonexty", line=dict(width=0), name="95% Monte Carlo interval"))
        fig.add_trace(go.Scatter(x=time, y=mc_results["mean"], line=dict(width=3), name="Monte Carlo mean"))
        fig.update_layout(
            title="Fetal exposure uncertainty from physiological variability",
            xaxis_title="Time (hours)", yaxis_title="Fetal concentration (mg/L)",
            template="plotly_white", height=430,
        )
        return fig

    @staticmethod
    def generate_sensitivity_chart(sensitivity_df: pd.DataFrame) -> go.Figure:
        if sensitivity_df is None or sensitivity_df.empty:
            return go.Figure()
        df = sensitivity_df.sort_values("auc_effect_percent_abs", ascending=True)
        fig = go.Figure(go.Bar(x=df["auc_effect_percent"], y=df["parameter"], orientation="h"))
        fig.update_layout(title="One-at-a-time sensitivity of fetal AUC", xaxis_title="Average absolute AUC change (%)", yaxis_title="", template="plotly_white", height=430)
        return fig

    @staticmethod
    def generate_scenario_chart(scenario_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if scenario_df is None or scenario_df.empty:
            return fig
        fig.add_trace(go.Bar(x=scenario_df["scenario"], y=scenario_df["AUC"], name="Fetal AUC"))
        fig.add_trace(go.Bar(x=scenario_df["scenario"], y=scenario_df["Cmax"], name="Fetal Cmax"))
        fig.update_layout(title="Scenario comparison", barmode="group", xaxis_title="Scenario", yaxis_title="Relative metric scale", template="plotly_white", height=430)
        return fig
