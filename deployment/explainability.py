from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DESCRIPTOR_NAMES = [
    "Molecular weight", "MolLogP", "TPSA", "H-bond donors", "H-bond acceptors",
    "Rotatable bonds", "Ring count", "Aromatic rings", "FractionCSP3", "Heavy atom count",
]


def stage2_feature_names() -> List[str]:
    return DESCRIPTOR_NAMES + [f"Morgan bit {i}" for i in range(2048)] + ["Stage-1 organ prediction"]


class ExplainableAIModule:
    @staticmethod
    def compute_stage2_shap(ai_pipeline, smiles: str, organ: str, prediction_payload: Optional[dict] = None, top_n: int = 12) -> Dict[str, object]:
        """Compute local SHAP values for the selected compound and Stage-2 LightGBM model.

        This explains the Stage-2 meta-refiner only. It does not claim causal or biological
        importance and it does not explain the downstream ODE simulator.
        """
        try:
            import shap
        except Exception as exc:
            return {"available": False, "reason": f"SHAP package is unavailable: {exc}"}

        if ai_pipeline.stage2_bundle is None:
            return {"available": False, "reason": "Stage-2 refiner bundle is not loaded."}

        try:
            smi = ai_pipeline.canonicalize(smiles)
            stage1 = ai_pipeline.get_stage1_predictions(smi)
            xchem = ai_pipeline.shared_features([smi])
            fusion = np.hstack([xchem, np.asarray([[stage1[organ]]], dtype=np.float32)])
            wrapper = ai_pipeline.stage2_bundle[organ]
            model = getattr(wrapper, "model", wrapper)

            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(fusion)
            if isinstance(raw, list):
                raw = raw[0]
            vals = np.asarray(raw, dtype=float).reshape(-1)
            names = stage2_feature_names()
            if len(vals) != len(names):
                names = [f"Feature {i}" for i in range(len(vals))]

            frame = pd.DataFrame({"feature": names, "shap_value": vals, "abs_shap": np.abs(vals)})
            frame["direction"] = np.where(frame["shap_value"] >= 0, "Increases prediction", "Decreases prediction")
            frame = frame.sort_values("abs_shap", ascending=False).head(int(top_n)).reset_index(drop=True)

            expected = explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = float(np.asarray(expected).reshape(-1)[0])
            else:
                expected = float(expected)

            pred = prediction_payload or ai_pipeline.predict_organ_risks(smi)
            return {
                "available": True,
                "organ": organ,
                "prediction": float(pred[organ]),
                "stage1_prediction": float(stage1[organ]),
                "expected_value": expected,
                "top_features": frame,
                "note": "SHAP values explain the Stage-2 meta-refiner for this molecule; they are not causal effects.",
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    @staticmethod
    def render_local_shap_bar(shap_result: Dict[str, object], title: Optional[str] = None) -> go.Figure:
        if not shap_result.get("available"):
            fig = go.Figure()
            fig.add_annotation(text=shap_result.get("reason", "SHAP explanation unavailable"), x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title=title or "SHAP explanation unavailable", template="plotly_white", height=390)
            return fig

        df = shap_result["top_features"].copy().sort_values("shap_value")
        colors = ["#b23a48" if v < 0 else "#2a9d8f" for v in df["shap_value"]]
        fig = go.Figure(go.Bar(
            x=df["shap_value"], y=df["feature"], orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_width=1, line_color="#666")
        fig.update_layout(
            title=title or f"Local SHAP drivers — {str(shap_result['organ']).capitalize()}",
            xaxis_title="SHAP contribution to Stage-2 prediction",
            yaxis_title="",
            template="plotly_white",
            height=max(380, 27 * len(df) + 120),
            margin=dict(l=180, r=30, t=70, b=50),
        )
        return fig

    @staticmethod
    def render_feature_importance_chart(stage2_bundle, organ="liver") -> go.Figure:
        labels = ["Stage-1 organ prediction", "Morgan fingerprint", "Physicochemical descriptors"]
        values = [0.0, 0.0, 0.0]
        if stage2_bundle is not None and organ in stage2_bundle:
            obj = stage2_bundle[organ]
            model = getattr(obj, "model", obj)
            imp = getattr(model, "feature_importances_", None)
            if imp is not None and len(imp) >= 2059:
                imp = np.asarray(imp, dtype=float)
                values = [float(imp[2058]), float(imp[10:2058].sum()), float(imp[:10].sum())]
        if not any(values):
            values = [1.0, 0.0, 0.0]
        total = max(sum(values), 1e-12)
        values = [v / total for v in values]
        df = pd.DataFrame({"Feature group": labels, "Relative importance": values}).sort_values("Relative importance")
        fig = go.Figure(go.Bar(x=df["Relative importance"], y=df["Feature group"], orientation="h"))
        fig.update_layout(
            title=f"Global Stage-2 {organ.capitalize()} feature-group importance",
            xaxis_title="Normalized split importance",
            yaxis_title="",
            template="plotly_white",
            height=330,
        )
        return fig
