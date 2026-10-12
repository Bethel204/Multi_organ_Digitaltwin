from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

import config
from ai_prediction import AIPredictionPipeline
from dosing import DosingRegimen
from explainability import ExplainableAIModule
from patient import PatientProfile
from reports import ClinicalReportGenerator
from twin_engine import FetalDigitalTwinEngine
from utils import fetch_pubchem_title, validate_and_canonicalize_smiles
from visualization import PublicationVisualizer

st.set_page_config(page_title="Hybrid AI Multi-Organ Fetal Digital Twin", layout="wide")
st.title("Hybrid AI Multi-Organ Fetal Digital Twin")
st.caption("Stage-2 multi-organ prediction • personalized developmental PK/PD simulation • uncertainty • explainability")
st.warning("Please NOTE that this work is a research framework only, it is not a clinical decision support or dosing tool.")


@st.cache_resource(show_spinner="Loading Stage-1 and Stage-2 model resources...")
def get_ai_pipeline():
    return AIPredictionPipeline(device="cpu")


ai_pipeline = get_ai_pipeline()


def _coupling_summary(pred, patient):
    liver_score = float(pred["liver"])
    placenta_score = float(pred["placenta"])
    kidney_score = float(pred["kidney"])
    mw = float(patient.molecular_weight_g_mol)
    liver_halfmax = max(1e-12, (10.0 ** (-liver_score)) * mw * 1000.0)
    kidney_halfmax = max(1e-12, (10.0 ** (-kidney_score)) * mw * 1000.0)
    delta = placenta_score - float(config.AI_PLACENTA_REFERENCE_SCORE)
    placenta_multiplier = float(np.clip(
        np.exp(float(config.AI_PLACENTA_COUPLING_STRENGTH) * delta),
        config.AI_PLACENTA_MIN_MULTIPLIER,
        config.AI_PLACENTA_MAX_MULTIPLIER,
    )) if config.AI_PLACENTA_COUPLING_ENABLED else 1.0
    return liver_halfmax, placenta_multiplier, kidney_halfmax


def render_deployment_status(status: dict, show_paths: bool = False):
    
    stage2_ready = bool(status.get("stage2_ready", False))
    cache_ready = bool(status.get("prediction_cache_loaded", False))
    errors = status.get("errors") or []

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;background:#ffffff;">
            <div style="font-size:0.82rem;color:#6b7280;">Stage-2 multi-organ model</div>
            <div style="font-size:1.05rem;font-weight:700;color:{'#15803d' if stage2_ready else '#b91c1c'};">{'Connected' if stage2_ready else 'Not ready'}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;background:#ffffff;">
            <div style="font-size:0.82rem;color:#6b7280;">Prediction cache</div>
            <div style="font-size:1.05rem;font-weight:700;color:{'#15803d' if cache_ready else '#b45309'};">{'Loaded' if cache_ready else 'Unavailable'}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;background:#ffffff;">
            <div style="font-size:0.82rem;color:#6b7280;">Loading errors</div>
            <div style="font-size:1.05rem;font-weight:700;color:{'#15803d' if not errors else '#b91c1c'};">{'None' if not errors else str(len(errors)) + ' detected'}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if stage2_ready:
        st.markdown("**Organ model stack:** Liver CYP3A7 — Ready · Placenta — Ready · Kidney OAT1 — Ready")
    else:
        st.markdown("**Organ model stack:** not fully initialized")

    refiners_path = status.get("stage2_refiners_path")
    predictions_path = status.get("stage2_predictions_path")
    if refiners_path:
        st.caption(f"Model source: {Path(refiners_path).name}")
    if predictions_path:
        st.caption(f"Prediction source: {Path(predictions_path).name}")

    if errors:
        st.error("Deployment loading reported one or more errors. Review the deployment status above.")

    if show_paths:
        st.code(json.dumps(status, indent=2, default=str), language="json")


st.sidebar.header("1. Target compound")
smiles_str = st.sidebar.text_input("SMILES", "CC1=CC2=C(C=CC=N2)C3=C1N(C(=N3)N)C")

st.sidebar.header("2. Developmental phenotype")
gestational_age = st.sidebar.slider("Gestational age (weeks)", 4.0, 40.0, 24.0, 1.0)
cyp3a7 = st.sidebar.slider("CYP3A7 relative expression", 0.1, 5.0, 1.0, 0.1)
f_binding = st.sidebar.slider("Plasma protein binding (fraction bound)", 0.0, 0.99, 0.85, 0.01)

st.sidebar.header("3. Organ physiology")
v_liver = st.sidebar.number_input("Fetal liver volume (L)", 0.01, 2.5, 0.35)
v_maternal_liver = st.sidebar.number_input("Maternal liver volume (L)", 0.1, 5.0, 1.5)
v_placenta = st.sidebar.number_input("Placenta volume (L)", 0.01, 2.0, 0.25)
q_liver = st.sidebar.number_input("Hepatic blood flow (L/hr)", 1.0, 120.0, 75.0)
q_placenta = st.sidebar.number_input("Placental blood flow (L/hr)", 0.1, 50.0, 12.5)
k_maturation = st.sidebar.slider("Kidney maturation factor", 0.05, 1.0, 0.65, 0.05)

st.sidebar.header("4. Kinetics / transport")
vmax_val = st.sidebar.number_input("Vmax (mg/hr)", 0.1, 500.0, 45.0)
km_val = st.sidebar.number_input("Km (µM)", 0.1, 200.0, 25.0)
transporter_act = st.sidebar.slider("Transporter activity coefficient", 0.1, 3.0, 1.0, 0.1)

st.sidebar.header("5. Dosing / simulation")
dose_load = st.sidebar.number_input("Initial dose (mg)", 0.0, 2000.0, 100.0)
dose_maint = st.sidebar.number_input("Repeated dose (mg)", 0.0, 2000.0, 0.0)
dose_int = st.sidebar.number_input("Dose interval (hr)", 1.0, 48.0, 8.0)
sim_duration = st.sidebar.number_input("Simulation duration (hr)", 1.0, 240.0, 72.0)
mc_iterations = st.sidebar.slider("Monte Carlo iterations", 20, 150, max(40, config.MONTE_CARLO_MIN_ITERATIONS), 10)

try:
    canonical = validate_and_canonicalize_smiles(smiles_str)
    valid_mol = Chem.MolFromSmiles(canonical)
    compound_identity = fetch_pubchem_title(canonical)
except Exception:
    canonical = None
    valid_mol = None
    compound_identity = "Invalid structure"

head1, head2 = st.columns([2, 1])
with head1:
    st.subheader(compound_identity)
    if canonical:
        st.code(canonical, language=None)
        try:
            descriptors = ai_pipeline.molecular_descriptors(canonical)
            st.dataframe(pd.DataFrame([descriptors]), use_container_width=True, hide_index=True)
        except Exception:
            pass
    else:
        st.error("Enter a valid SMILES string.")
with head2:
    if valid_mol is not None:
        st.image(Draw.MolToImage(valid_mol, size=(360, 230)), caption="RDKit 2D structure")


def make_patient(**overrides):
    vals = dict(
        gestational_age_weeks=gestational_age,
        cyp3a7_expression=cyp3a7,
        liver_volume_l=v_liver,
        placental_blood_flow_l_hr=q_placenta,
        kidney_maturation_factor=k_maturation,
        plasma_protein_binding_fraction=f_binding,
        hepatic_blood_flow_l_hr=q_liver,
        km_micromolar=km_val,
        vmax_mg_hr=vmax_val,
        transporter_activity_coefficient=transporter_act,
        molecular_weight_g_mol=Descriptors.MolWt(valid_mol),
        maternal_liver_volume_l=v_maternal_liver,
        placenta_volume_l=v_placenta,
    )
    vals.update(overrides)
    return PatientProfile(**vals)


def make_dosing():
    return DosingRegimen(dose_load, dose_maint, float(dose_int), repeated=dose_maint > 0)


if st.sidebar.button("Generate personalized twin simulation", type="primary"):
    if valid_mol is None:
        st.error("Simulation cannot run with an invalid SMILES structure.")
        st.stop()
    patient = make_patient()
    dosing = make_dosing()
    try:
        with st.spinner("Running Stage-2 inference, PK/PD simulation, Monte Carlo uncertainty, and local SHAP..."):
            engine = FetalDigitalTwinEngine(canonical, patient, dosing, ai_pipeline=ai_pipeline)
            output_df = engine.run_simulation(float(sim_duration), dt=config.TIME_STEP_HRS)
            mc_bounds = engine.run_monte_carlo_uncertainty(float(sim_duration), iterations=int(mc_iterations), dt=0.2)
            metrics = ClinicalReportGenerator.calculate_pk_metrics(output_df)
            shap_results = {
                organ: ExplainableAIModule.compute_stage2_shap(ai_pipeline, canonical, organ, engine.organ_predictions, top_n=12)
                for organ in ("liver", "placenta", "kidney")
            }
        st.session_state["digital_twin_run"] = {
            "canonical": canonical,
            "compound_identity": compound_identity,
            "patient": patient,
            "dosing": dosing,
            "organ_predictions": engine.organ_predictions,
            "output_df": output_df,
            "mc_bounds": mc_bounds,
            "metrics": metrics,
            "shap_results": shap_results,
            "sensitivity_df": None,
            "scenario_df": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        st.error(str(exc))
        st.info("Keep this deployment folder inside compareNew beside multiorgan_results_latent, or set MULTIORGAN_RESULTS_DIR to the Stage-2 results folder.")
        st.stop()

run = st.session_state.get("digital_twin_run")
if run is None:
    st.info("Enter the compound and individualized parameters in the sidebar, then run the personalized digital twin simulation.")
else:
    pred = run["organ_predictions"]
    output_df = run["output_df"]
    metrics = run["metrics"]
    patient = run["patient"]
    dosing = run["dosing"]
    shap_results = run["shap_results"]

    tabs = st.tabs([
        "Overview",
        "Stage-2 Predictions",
        "SHAP Explainability",
        "Concentration Profiles",
        "PK / Toxicity / Resistance",
        "Developmental Scenarios",
        "Sensitivity & Uncertainty",
        "Report & Downloads",
    ])

    
    with tabs[0]:
        st.subheader("Integrated multiorgan digital twin overview")
        #a, b, c, d = st.columns(4)
       # a.metric("Fetal liver CYP3A7", f"{pred['liver']:.4f}")
        #b.metric("Placental transfer", f"{pred['placenta']:.4f}")
        #c.metric("Fetal kidney OAT1", f"{pred['kidney']:.4f}")
        #d.metric("Model source", pred.get("source", "unknown"))

        st.markdown(
            """
            <div style='padding:18px;border:1px solid #ddd;border-radius:12px;text-align:center;'>
            <b>Maternal exposure</b> &nbsp; → &nbsp; <b>Placenta transfer</b> &nbsp; → &nbsp;
            <b>Fetal circulation</b> &nbsp; → &nbsp; <b>Fetal liver (CYP3A7)</b> + <b>Fetal kidney (OAT1)</b><br>
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        liver_halfmax, placenta_multiplier, kidney_halfmax = _coupling_summary(pred, patient)
        st.markdown("### ML → PBPK coupling")
        st.caption("These values are ML outputs before PBPK coupling. The quantities below show how each output enters the PBPK/ODE simulation.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Liver CYP3A7 ", f"{pred['liver']:.4f}")
            st.markdown(f"**→ CYP3A7 activity coupling**  \nConcentration-scale half-max parameter: **{liver_halfmax:.6g} mg/L**  \nAffects: **fetal hepatic metabolism**")
        with c2:
            st.metric("Placenta ", f"{pred['placenta']:.4f}")
            st.markdown(f"**→ Placental transfer coupling**  \nTransfer multiplier: **{placenta_multiplier:.3f}×**  \nAffects: **maternal→placenta and placenta→fetal transfer**")
        with c3:
            st.metric("Kidney OAT1 ", f"{pred['kidney']:.4f}")
            st.markdown(f"**→ OAT1/renal activity coupling**  \nConcentration-scale half-max parameter: **{kidney_halfmax:.6g} mg/L**  \nAffects: **fetal renal elimination**")

       
        st.markdown("#### Outputs after Finetuning the PBPK with Our ML IC50")
        o1, o2, o3 = st.columns(3)
        o1.metric("Fetal AUC", f"{metrics['AUC_Fetal_Circulation_mg_hr_L']:.3f} mg·hr/L")
        o2.metric("Fetal Cmax", f"{metrics['Fetal_Cmax_mg_L']:.3f} mg/L")
        o3.metric("Fetal Tmax", f"{metrics['Fetal_Tmax_Hours']:.2f} hr")
        st.plotly_chart(PublicationVisualizer.generate_concentration_curves(output_df), use_container_width=True, key="overview_concentration_profile")

    
    with tabs[1]:
        st.subheader("Stage-2 multi-organ predictions")
        p1, p2, p3 = st.columns(3)
        p1.metric("Liver CYP3A7", f"{pred['liver']:.4f}", help=f"Run-reported Stage-2 OOF R²: {ai_pipeline.metrics['liver']:.4f}")
        p2.metric("Placenta transfer", f"{pred['placenta']:.4f}", help=f"Run-reported Stage-2 OOF R²: {ai_pipeline.metrics['placenta']:.4f}")
        p3.metric("Kidney OAT1", f"{pred['kidney']:.4f}", help=f"Run-reported Stage-2 OOF R²: {ai_pipeline.metrics['kidney']:.4f}")

        st.caption(
            f"Stage-2 performance metadata: liver R²={ai_pipeline.metrics['liver']:.4f}; "
            f"placenta R²={ai_pipeline.metrics['placenta']:.4f}; kidney R²={ai_pipeline.metrics['kidney']:.4f}."
        )

        if "stage1" in pred:
            s1df = pd.DataFrame({
                "Organ": ["Liver CYP3A7", "Placenta", "Kidney OAT1"],
                "Stage-1 prediction": [pred["stage1"]["liver"], pred["stage1"]["placenta"], pred["stage1"]["kidney"]],
                "Stage-2 prediction": [pred["liver"], pred["placenta"], pred["kidney"]],
            })
            st.dataframe(s1df, use_container_width=True, hide_index=True)

        intervals = pred.get("stage1_intervals_90", {})
        if intervals:
            rows = []
            for organ in ("placenta", "kidney"):
                if organ in intervals:
                    lo, hi = intervals[organ]
                    rows.append({"Organ": organ.capitalize(), "90% lower": lo, "90% upper": hi, "Applies to": "Stage-1 conformal model"})
            st.markdown("#### Prediction uncertainty")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.info("These are our calibrated Stage1 conformal intervals.")

    
    with tabs[2]:
        st.subheader("Per-compound Stage-2 SHAP explainability")
        st.caption("SHAP explains how the Stage-2 meta-refiner features shift this molecule's prediction. It is not a causal biological interpretation.")
        for organ, label in [("liver", "Liver CYP3A7"), ("placenta", "Placental transfer"), ("kidney", "Kidney OAT1")]:
            res = shap_results.get(organ, {})
            st.markdown(f"### {label}")
            if res.get("available"):
                st.plotly_chart(ExplainableAIModule.render_local_shap_bar(res, f"{label}: top local SHAP drivers"), use_container_width=True, key=f"shap_local_{organ}")
                show = res["top_features"][["feature", "shap_value", "direction"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.warning(res.get("reason", "SHAP explanation unavailable."))
        with st.expander("Global Stage-2 feature-group importance"):
            for organ in ("liver", "placenta", "kidney"):
                st.plotly_chart(ExplainableAIModule.render_feature_importance_chart(ai_pipeline.stage2_bundle, organ), use_container_width=True, key=f"shap_global_{organ}")

    
    with tabs[3]:
        st.subheader("Concentration time profiles (outputs after ML–PBPK integration)")
        st.plotly_chart(PublicationVisualizer.generate_concentration_curves(output_df), use_container_width=True, key="concentration_tab_main_profile")
        st.plotly_chart(PublicationVisualizer.generate_fetal_organ_curves(output_df), use_container_width=True, key="concentration_fetal_organs")
        st.caption("Only compartments explicitly represented by the implemented ODE system are plotted.")

    
    with tabs[4]:
        st.subheader("PK/PD summary — outputs after ML–PBPK coupling")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fetal AUC", f"{metrics['AUC_Fetal_Circulation_mg_hr_L']:.3f} mg·hr/L")
        m2.metric("Fetal Cmax", f"{metrics['Fetal_Cmax_mg_L']:.3f} mg/L")
        m3.metric("Fetal Tmax", f"{metrics['Fetal_Tmax_Hours']:.2f} hr")
        m4.metric("Terminal fetal concentration", f"{metrics['Fetal_Terminal_Concentration_mg_L']:.3f} mg/L")
        t1, t2, t3 = st.columns(3)
        t1.metric("Mean toxicity index", f"{metrics['Mean_Toxicity_Index']:.3f}")
        t2.metric("Peak toxicity index", f"{metrics['Peak_Toxicity_Index']:.3f}")
        t3.metric("Peak resistance score", f"{metrics['Peak_Resistance_Score']:.3f}")
        st.plotly_chart(PublicationVisualizer.generate_toxicity_resistance(output_df), use_container_width=True, key="pkpd_toxicity_resistance")
        

   
    with tabs[5]:
        st.subheader("Developmental and physiological scenario comparison")
        st.write("Compare the same molecule and dosing regimen while changing developmental or physiological assumptions. Stage-2 molecular predictions are held constant so this section isolates simulator-level changes.")
        if st.button("Run developmental scenario comparison"):
            scenarios = [
                ("Baseline", {}),
                (f"GA {max(4.0, patient.gestational_age-6):.0f} wk", {"gestational_age_weeks": max(4.0, patient.gestational_age-6)}),
                (f"GA {min(40.0, patient.gestational_age+6):.0f} wk", {"gestational_age_weeks": min(40.0, patient.gestational_age+6)}),
                ("CYP3A7 ×0.5", {"cyp3a7_expression": max(0.1, patient.cyp3a7_expression*0.5)}),
                ("CYP3A7 ×1.5", {"cyp3a7_expression": min(5.0, patient.cyp3a7_expression*1.5)}),
                ("Transporter ×0.5", {"transporter_activity_coefficient": max(0.1, patient.transporter_modifier*0.5)}),
                ("Transporter ×1.5", {"transporter_activity_coefficient": min(3.0, patient.transporter_modifier*1.5)}),
            ]
            rows = []
            with st.spinner("Running scenarios..."):
                for name, ov in scenarios:
                    p = make_patient(**ov)
                    e = FetalDigitalTwinEngine(run["canonical"], p, make_dosing(), ai_pipeline=ai_pipeline, organ_predictions=pred)
                    df = e.run_simulation(float(sim_duration), dt=max(config.TIME_STEP_HRS, 0.2))
                    met = ClinicalReportGenerator.calculate_pk_metrics(df)
                    rows.append({"scenario": name, "AUC": met["AUC_Fetal_Circulation_mg_hr_L"], "Cmax": met["Fetal_Cmax_mg_L"], "Tmax": met["Fetal_Tmax_Hours"]})
            run["scenario_df"] = pd.DataFrame(rows)
            st.session_state["digital_twin_run"] = run
        if run.get("scenario_df") is not None:
            st.dataframe(run["scenario_df"], use_container_width=True, hide_index=True)
            st.plotly_chart(PublicationVisualizer.generate_scenario_chart(run["scenario_df"]), use_container_width=True, key="developmental_scenario_chart")

    
    with tabs[6]:
        st.subheader("Uncertainty and simulator sensitivity")
        st.plotly_chart(PublicationVisualizer.generate_uncertainty_bounds(run["mc_bounds"]), use_container_width=True, key="monte_carlo_uncertainty")
        st.caption("The shaded interval reflects Monte Carlo variation in selected physiological flows.")

        if st.button("Run one-at-a-time sensitivity analysis"):
            base_auc = max(metrics["AUC_Fetal_Circulation_mg_hr_L"], 1e-9)
            base_cmax = max(metrics["Fetal_Cmax_mg_L"], 1e-9)
            tests = [
                ("CYP3A7 expression", "cyp3a7_expression", patient.cyp3a7_expression),
                ("Placental blood flow", "q_placenta", patient.q_placenta),
                ("Hepatic blood flow", "q_liver", patient.q_liver),
                ("Kidney maturation", "kidney_maturation", patient.kidney_maturation),
                ("Transporter activity", "transporter_modifier", patient.transporter_modifier),
                ("Vmax", "vmax", patient.vmax),
                ("Km (µM)", "km_micromolar", patient.km_micromolar),
                ("Unbound fraction", "fu_maternal", patient.fu_maternal),
            ]
            rows = []
            with st.spinner("Perturbing simulator parameters ±20%..."):
                for label, attr, base in tests:
                    auc_changes, cmax_changes = [], []
                    for factor in (0.8, 1.2):
                        p = copy.deepcopy(patient)
                        setattr(p, attr, max(1e-6, base * factor))
                        if attr == "km_micromolar":
                            p.km_mg_l = p.km_micromolar * p.molecular_weight_g_mol / 1000.0
                            p.km = p.km_mg_l
                        # Recalculate dependent fetal flows when maturation-related parameters change.
                        if attr == "kidney_maturation":
                            p.v_fetal_kidney = p.v_fetal_blood * 0.12 * p.kidney_maturation
                            p.fetal_kidney_flow = 0.45 * p.v_fetal_kidney
                        e = FetalDigitalTwinEngine(run["canonical"], p, make_dosing(), ai_pipeline=ai_pipeline, organ_predictions=pred)
                        df = e.run_simulation(float(sim_duration), dt=max(config.TIME_STEP_HRS, 0.25))
                        met = ClinicalReportGenerator.calculate_pk_metrics(df)
                        auc_changes.append(abs((met["AUC_Fetal_Circulation_mg_hr_L"] - base_auc) / base_auc * 100.0))
                        cmax_changes.append(abs((met["Fetal_Cmax_mg_L"] - base_cmax) / base_cmax * 100.0))
                    rows.append({
                        "parameter": label,
                        "auc_effect_percent": float(np.mean(auc_changes)),
                        "cmax_effect_percent": float(np.mean(cmax_changes)),
                        "auc_effect_percent_abs": float(np.mean(auc_changes)),
                    })
            run["sensitivity_df"] = pd.DataFrame(rows)
            st.session_state["digital_twin_run"] = run
        if run.get("sensitivity_df") is not None:
            st.dataframe(run["sensitivity_df"], use_container_width=True, hide_index=True)
            st.plotly_chart(PublicationVisualizer.generate_sensitivity_chart(run["sensitivity_df"]), use_container_width=True, key="simulator_sensitivity")
            st.info("Sensitivity analysis explains which ODE/simulator parameters most affect fetal exposure. This is distinct from SHAP, which explains the Stage-2 ML prediction.")

    
    with tabs[7]:
        st.subheader("Our Research report")
        report_text = ClinicalReportGenerator.export_text_summary(
            metrics,
            f"{run['compound_identity']} (GA {patient.gestational_age:.1f} weeks)",
            pred,
            patient=patient,
            dosing=dosing,
            shap_results=shap_results,
            sensitivity_df=run.get("sensitivity_df"),
        )
        st.text(report_text)
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button("Download report (TXT)", report_text.encode("utf-8"), file_name="fetal_digital_twin_report.txt", mime="text/plain")
        try:
            pdf_bytes = ClinicalReportGenerator.export_pdf_bytes(report_text)
            dl2.download_button("Download report (PDF)", pdf_bytes, file_name="fetal_digital_twin_report.pdf", mime="application/pdf")
        except Exception as exc:
            dl2.warning(f"PDF unavailable: {exc}")

        export_cols = [c for c in [
            "current_time", "maternal_conc", "maternal_liver_conc", "placenta_conc",
            "fetal_blood_conc", "fetal_liver_conc", "fetal_kidney_conc", "metabolite_amount_mg",
            "cyp3a7_expression", "resistance_score", "toxicity_index"
        ] if c in output_df.columns]
        dl3.download_button(
            "Download trajectories (CSV)",
            output_df[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="fetal_digital_twin_trajectories.csv",
            mime="text/csv",
        )

        shap_rows = []
        for organ, res in shap_results.items():
            if res.get("available"):
                tmp = res["top_features"].copy()
                tmp.insert(0, "organ", organ)
                shap_rows.append(tmp)
        if shap_rows:
            shap_df = pd.concat(shap_rows, ignore_index=True)
            st.download_button("Download SHAP drivers (CSV)", shap_df.to_csv(index=False).encode("utf-8"), file_name="stage2_local_shap.csv", mime="text/csv")
        if run.get("sensitivity_df") is not None:
            st.download_button("Download sensitivity analysis (CSV)", run["sensitivity_df"].to_csv(index=False).encode("utf-8"), file_name="simulator_sensitivity.csv", mime="text/csv")
        if run.get("scenario_df") is not None:
            st.download_button("Download scenario comparison (CSV)", run["scenario_df"].to_csv(index=False).encode("utf-8"), file_name="developmental_scenarios.csv", mime="text/csv")

        with st.expander("Model provenance / deployment status"):
            status = ai_pipeline.status()
            status.update({
                "simulation_timestamp": run["timestamp"],
                "canonical_smiles": run["canonical"],
                "reported_stage2_r2": ai_pipeline.metrics,
                "monte_carlo_iterations": mc_iterations,
                "research_use_only": True,
            })
            render_deployment_status(status)
            st.markdown("**Advanced run details**")
            st.code(json.dumps(status, indent=2, default=str), language="json")

with st.expander("Deployment status (always available)"):
    render_deployment_status(ai_pipeline.status())

