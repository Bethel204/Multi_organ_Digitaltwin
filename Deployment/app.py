import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from patient import PatientProfile
from dosing import DosingRegimen
from twin_engine import FetalDigitalTwinEngine
from visualization import PublicationVisualizer
from reports import ClinicalReportGenerator
from explainability import ExplainableAIModule
from utils import fetch_pubchem_title, validate_and_canonicalize_smiles

st.set_page_config(page_title="Personalized Fetal Digital Twin", layout="wide")
st.title(" A Hybrid AI Multi-Organ Digital Twin Framework for Developmental Pharmacokinetics and Drug Resistance Prediction")


st.sidebar.header("1. Target Compound Input")
smiles_str = st.sidebar.text_input(
    "Target Compound SMILES Structure", 
    "CC1=CC2=C(C=CC=N2)C3=C1N(C(=N3)N)C"
)

st.sidebar.header("2. Personalized Patient Phenotype")
gestational_age = st.sidebar.slider("Gestational Age (Weeks)", 4.0, 40.0, 24.0, step=1.0)
cyp3a7 = st.sidebar.slider("CYP3A7 Relative Enzyme Expression", 0.1, 5.0, 1.0, step=0.1)
f_binding = st.sidebar.slider("Plasma Protein Binding (Fraction Bound)", 0.0, 0.99, 0.85, step=0.01)

st.sidebar.header("3. Organ Volume & Flow Perfusions")
v_liver = st.sidebar.number_input("Liver Compartment Volume (L)", min_value=0.01, max_value=2.5, value=0.35)
q_liver = st.sidebar.number_input("Hepatic Blood Flow (L/hr)", min_value=1.0, max_value=120.0, value=75.0)
q_placenta = st.sidebar.number_input("Placental Blood Flow (L/hr)", min_value=0.1, max_value=50.0, value=12.5)
k_maturation = st.sidebar.slider("Kidney Maturation Clearance Factor", 0.05, 1.0, 0.65, step=0.05)

st.sidebar.header("4. Enzyme Saturation Kinetics")
vmax_val = st.sidebar.number_input("Inherent Clearance Vmax (mg/hr)", min_value=0.1, max_value=500.0, value=45.0)
km_val = st.sidebar.number_input("Affinity Constant Km (uM)", min_value=0.1, max_value=200.0, value=25.0)
transporter_act = st.sidebar.slider("Transporter Activity Coefficient", 0.1, 3.0, 1.0, step=0.1)

st.sidebar.header("5. Therapeutic Intake Protocol")
dose_load = st.sidebar.number_input("Initial Intake Loading Dose (mg)", value=500.0)
dose_maint = st.sidebar.number_input("Repeated Maintenance Dose Amount (mg)", value=250.0)
dose_int = st.sidebar.slider("Dosing Interval Frequency Window (Hours)", 4, 24, 8)
sim_duration = st.sidebar.slider("Total Active Simulation Track Timeline (Hours)", 12, 168, 72)



st.markdown("---")
col_meta, col_img = st.columns([2, 1])

with col_meta:
    st.subheader("Analyte Identification Context")
    if smiles_str.strip():
        try:
            canonical_smi = validate_and_canonicalize_smiles(smiles_str)
            compound_identity = fetch_pubchem_title(canonical_smi)
            
            st.success(f"**Identified Compound Name:** {compound_identity}")
            st.info(f"**Canonical SMILES:** `{canonical_smi}`")
            valid_mol = Chem.MolFromSmiles(canonical_smi)
        except Exception as e:
            st.error(f"⚠️ Structural Validation Error: {e}")
            valid_mol = None
    else:
        st.warning("Please enter a valid SMILES sequence in the sidebar panel.")
        valid_mol = None

with col_img:
    if valid_mol is not None:
        st.markdown("**2D Structural Diagram Visualization**")
        img = Draw.MolToImage(valid_mol, size=(300, 200))
        st.image(img, use_container_width=False, caption="RDKit Vector Geometry Map")



st.markdown("---")
if st.sidebar.button("Generate Personalized Twin Simulation Run"):
    if valid_mol is None:
        st.error("Cannot execute simulation loop with an invalid or unsanitized SMILES chemical input.")
    else:
       
        personalized_patient = PatientProfile(
            gestational_age_weeks=gestational_age,
            cyp3a7_expression=cyp3a7,
            liver_volume_l=v_liver,
            placental_blood_flow_l_hr=q_placenta,
            kidney_maturation_factor=k_maturation,
            plasma_protein_binding_fraction=f_binding,
            hepatic_blood_flow_l_hr=q_liver,
            km_micromolar=km_val,
            vmax_mg_hr=vmax_val,
            transporter_activity_coefficient=transporter_act
        )
        
        
        custom_dosing = DosingRegimen(
            initial_dose_mg=dose_load,
            repeated_dose_mg=dose_maint,
            interval_hrs=float(dose_int),
            repeated=(dose_maint > 0)
        )
        
        with st.spinner("Assembling unique twin phenotype profile and running adaptive ODE integration solver..."):
            engine = FetalDigitalTwinEngine(smiles=smiles_str, patient=personalized_patient, dosing=custom_dosing)
            
            
            output_df = engine.run_simulation(total_duration_hrs=sim_duration, dt=0.1)
            mc_bounds = engine.run_monte_carlo_uncertainty(total_duration_hrs=sim_duration, iterations=40, dt=0.2)
            
            
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Patient-Specific Compartmental Clearance Curves")
                st.plotly_chart(PublicationVisualizer.generate_concentration_curves(output_df), use_container_width=True)
            with chart_col2:
                st.markdown("#### Custom Cohort Uncertainty Boundary Variance Clouds")
                st.plotly_chart(PublicationVisualizer.generate_uncertainty_bounds(mc_bounds), use_container_width=True)
                
            
            st.markdown("---")
            export_col, contrib_col = st.columns(2)
            
            with export_col:
                st.markdown("####  Export Dynamic Organ Concentrations")
                st.write("Download the simulated concentration trajectories over time for all target organs.")
                
               
                target_columns = [
                    "current_time", 
                    "maternal_conc", 
                    "maternal_liver_conc", 
                    "placenta_conc",
                    "fetal_blood_conc", 
                    "fetal_liver_conc",
                    "fetal_kidney_conc",
                    "metabolite_conc",
                    "cyp3a7_expression",
                    "toxicity_index"
                ]

                export_columns = [col for col in target_columns if col in output_df.columns]
                filtered_df = output_df[export_columns]

                
                filtered_df = filtered_df.rename(columns={
                    "current_time": "Time (Hours)",
                    "maternal_conc": "Maternal Plasma Concentration (mg/L)",
                    "maternal_liver_conc": "Maternal Liver Concentration (mg/L)",
                    "placenta_conc": "Placenta Concentration (mg/L)",
                    "fetal_blood_conc": "Fetal Blood Concentration (mg/L)",
                    "fetal_liver_conc": "Fetal Liver Concentration (mg/L)",
                    "fetal_kidney_conc": "Fetal Kidney Concentration (mg/L)",
                    "metabolite_conc": "Total Metabolite Concentration (mg/L)"
                })
                                
               
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label=" Download Comprehensive Multi-Organ Trajectories (CSV)",
                    data=csv_data,
                    file_name=f"fetal_twin_{compound_identity.replace(' ', '_')}_full_trajectories.csv",
                    mime="text/csv"
                )
                
                st.markdown("#### Automated Digital Twin Screening Evaluation Report")
                metrics = ClinicalReportGenerator.calculate_pk_metrics(output_df)
                st.text(ClinicalReportGenerator.export_text_summary(metrics, f"{compound_identity} (GA: {gestational_age} Wks)"))

            with contrib_col:
                st.markdown("####  Major Structural & Physiological Contributors")
                st.write("Relative feature drivers shaping active clearance modifications and organ risk evaluations.")
                
               
                fig_contributors = ExplainableAIModule.render_feature_importance_chart(engine.ai_pipeline.model_bundle)
                st.plotly_chart(fig_contributors, use_container_width=True)