import plotly.graph_objects as go
import pandas as pd
import numpy as np

class PublicationVisualizer:
    @staticmethod
    def generate_concentration_curves(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        
        fig.add_trace(go.Scatter(x=df["current_time"], y=df["maternal_conc"], name="Maternal Plasma", line=dict(width=3, color='#1f77b4')))
        fig.add_trace(go.Scatter(x=df["current_time"], y=df["fetal_blood_conc"], name="Fetal Circulation", line=dict(width=3, color='#ff7f0e')))
        fig.add_trace(go.Scatter(x=df["current_time"], y=df["fetal_liver_conc"], name="Fetal Liver Compartment", line=dict(dash='dash', color='#2ca02c')))
        fig.add_trace(go.Scatter(x=df["current_time"], y=df["fetal_kidney_conc"], name="Fetal Kidney Compartment", line=dict(dash='dot', color='#9467bd')))
        
        fig.update_layout(
            title="Multi-Organ Pharmacokinetic Trajectory Simulation (Adaptive Solver)",
            xaxis_title="Time Frame (Hours)", yaxis_title="Compartment Analyte Level (mg/L)",
            template="plotly_white", legend=dict(x=0.01, y=0.99)
        )
        return fig

    @staticmethod
    def generate_uncertainty_bounds(mc_results: dict) -> go.Figure:
        """Renders the 95% Confidence Interval cloud generated during Monte Carlo iterations."""
        fig = go.Figure()
        time = mc_results["time"]
        
        
        fig.add_trace(go.Scatter(x=time, y=mc_results["upper_95ci"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=time, y=mc_results["lower_95ci"], fill='tonexty', fillcolor='rgba(255,127,14,0.15)', line=dict(width=0), name="95% Confidence Interval"))
        
       
        fig.add_trace(go.Scatter(x=time, y=mc_results["mean"], line=dict(width=3, color='#ff7f0e'), name="Population Mean Target"))
        
        fig.update_layout(
            title="Monte Carlo Population Uncertainty Quantification",
            xaxis_title="Time Frame (Hours)", yaxis_title="Fetal Concentration Range (mg/L)",
            template="plotly_white"
        )
        return fig