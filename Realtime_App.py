import streamlit as st
import numpy as np
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit import DataStructs
import io

def get_drug_name(smi):
    try:
        res = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smi}/description/JSON", timeout=2)
        return res.json()['InformationList']['Information'][0]['Title']
    except: 
        return "Experimental Compound"

def generate_tabular_features(smi):
    """Generates the uniform 2058 feature row required by your tree models"""
    mol = Chem.MolFromSmiles(smi)
    if not mol: 
        return None
        
    desc = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), 
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
        Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol), 
        Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol), 
        mol.GetNumHeavyAtoms()
    ]
    
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    
    features = np.array([desc + fp.tolist()])
    return features

def simulate_pk_profile(res, duration_hrs):
    t = np.linspace(0, duration_hrs, 100)
   
    ke = 0.1 + (res['Liver Risk'] + res['Kidney Risk']) * 0.02
    ka = 0.6 
    conc = (100 * ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
    return t, np.maximum(conc, 0)


st.set_page_config(page_title="Fetal Digital Twin Dashboard", layout="wide")
st.title("Hybrid AI Multiorgan Digital Twin for Developmental Pharmacokinetics and Fetal Drug Resistance")

@st.cache_resource
def load_assets():
    model_bundle = joblib.load('Multiorganmodel.pt')
    return model_bundle

@st.cache_data
def load_prediction_database():
    """Loads and caches the spreadsheet containing pre-calculated risks"""
    try:
        return pd.read_csv("PRETRAINED_MULTIORGAN_PREDICTIONS_COMPLETED.csv")
    except Exception:
        return None

try:
    models = load_assets()
    db_predictions = load_prediction_database()
    st.success("Models loaded successfully from Multiorganmodel.pt!")
except Exception as e:
    st.error("Missing model dictionary. Ensure 'Multiorganmodel.pt' is generated and placed in this directory.")
    st.stop()


st.sidebar.header(" Configuration")
trimester = st.sidebar.select_slider("Select Trimester", options=["1st Trimester", "2nd Trimester", "3rd Trimester"])
sim_hrs = st.sidebar.slider("Simulation Duration (Hours)", 6, 72, 24)
dual_mode = st.sidebar.checkbox("Enable Dual Comparison Mode")


ontogeny_map = {
    "1st Trimester": {"liver": 1.4, "kidney": 1.6, "placenta": 1.2},
    "2nd Trimester": {"liver": 1.2, "kidney": 1.3, "placenta": 1.0},
    "3rd Trimester": {"liver": 1.0, "kidney": 1.0, "placenta": 0.8}
}


col_in1, col_in2 = st.columns(2)
with col_in1:
    smi_a = st.text_input("SMILES for Molecule A", "CC(=O)NC1=CC=C(O)C=C1")
    name_a = get_drug_name(smi_a)
    st.caption(f"**Name:** {name_a}")
    mol_a = Chem.MolFromSmiles(smi_a)
    if mol_a: 
        st.image(Draw.MolToImage(mol_a, size=(300, 300)))

smi_b = None
if dual_mode:
    with col_in2:
        smi_b = st.text_input("SMILES for Molecule B", "CC1=C(C(C(=C(N1)C)C(=O)OC)C2=CC=CC=C2[N+](=O)[O-])C(=O)OC")
        name_b = get_drug_name(smi_b)
        st.caption(f"**Name:** {name_b}")
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_b: 
            st.image(Draw.MolToImage(mol_b, size=(300, 300)))

def run_prediction(smi, name):
    is_precomputed = False
    if db_predictions is not None and not db_predictions.empty:
        matched_row = db_predictions[db_predictions["SMILES"] == smi]
        if not matched_row.empty:
            raw_liver = float(matched_row["Liver_Prediction_Risk"].values[0])
            raw_kidney = float(matched_row["Kidney_Prediction_Risk"].values[0])
            raw_placenta = float(matched_row["Placenta_Prediction_Risk"].values[0])
            is_precomputed = True

    if not is_precomputed:
        features = generate_tabular_features(smi)
        if features is None: 
            return None
        raw_liver = float(models["liver"].predict(features)[0])
        raw_kidney = float(models["kidney"].predict(features)[0])
        raw_placenta = float(models["placenta"].predict(features)[0])
    
    f = ontogeny_map[trimester]
    binding_score = np.clip(np.mean([raw_liver, raw_kidney, raw_placenta]) * 1.5, 0, 10) 
    
    res = {
        "Name": name,
        "Liver Risk": raw_liver * f['liver'],
        "Placenta Risk": raw_placenta * f['placenta'],
        "Kidney Risk": raw_kidney * f['kidney'],
        "Target Binding (pKd)": binding_score,
        "Is_Precomputed": is_precomputed,
        "SMILES_String": smi
    }
    
    risk_values = {k: res[k] for k in ["Liver Risk", "Placenta Risk", "Kidney Risk"]}
    sorted_risks = sorted(risk_values.items(), key=lambda x: x[1], reverse=True)
    res["Top_2_Risks"] = sorted_risks[:2]
    return res


if st.button(" Analyze & Compare Compounds"):
    res_a = run_prediction(smi_a, name_a)
    res_b = run_prediction(smi_b, name_b) if dual_mode and smi_b else None
    
    if res_a:
       
        st.subheader(" Structural Toxicity Predictions")
        results_list = [res_a]
        if res_b: 
            results_list.append(res_b)
        
        df_comp = pd.DataFrame(results_list).drop(columns=["Top_2_Risks", "Is_Precomputed", "SMILES_String"])
        
        st.table(df_comp.style.highlight_max(axis=0, color='#ffcccc', subset=["Liver Risk", "Placenta Risk", "Kidney Risk"]))

        st.subheader(" Medicinal Chemistry Property Evaluation")
        rule_cols = st.columns(2) if res_b else [st.container()]
        for idx, res in enumerate(results_list):
            with rule_cols[idx]:
                m_obj = Chem.MolFromSmiles(res["SMILES_String"])
                if m_obj:
                    w = Descriptors.MolWt(m_obj)
                    lp = Descriptors.MolLogP(m_obj)
                    hd = Descriptors.NumHDonors(m_obj)
                    ha = Descriptors.NumHAcceptors(m_obj)
                    
                    infractions = 0
                    if w > 500: infractions += 1
                    if lp > 5: infractions += 1
                    if hd > 5: infractions += 1
                    if ha > 10: infractions += 1
                    
                    st.markdown(f"**Lipinski Profile for {res['Name']}:**")
                    st.text(f"• MolWt: {w:.1f} | • LogP: {lp:.2f} | • H-Donors: {hd} | • H-Acceptors: {ha}")
                    if infractions > 1:
                        st.warning(f"⚠️ High Absorption Infraction Alert: {infractions} violations registered.")
                    else:
                        st.success(" Fulfills Standard Bioavailability Matrices.")

        st.subheader(" Multi-Organ Risk Profiling Report")
        exp_cols = st.columns(2) if res_b else [st.container()]
        
        for idx, res in enumerate(results_list):
            with exp_cols[idx]:
                st.markdown(f"### Analysis Summary: {res['Name']}")
                if res["Is_Precomputed"]:
                    st.caption(" Metrics fetched instantly from internal database records.")
                r1_name, r1_val = res["Top_2_Risks"][0]
                r2_name, r2_val = res["Top_2_Risks"][1]
                
                st.error(f"**Primary Risk Vulnerability:** {r1_name} (Value: {r1_val:.3f})")
                st.write(f"The model flags this as the principal clearance risk. The generated molecular attributes point toward structural characteristics matching high-affinity accumulation profiles in {r1_name} tissues during the {trimester}.")
                
                st.warning(f"**Secondary Risk Vulnerability:** {r2_name} (Value: {r2_val:.3f})")
                st.write(f"The {r2_name} exhibits elevated systemic vulnerability. Fetal cellular clearance thresholds in the {trimester} run at a fraction of standard maternal metabolic baselines, indicating limited processing capacity.")
                
                st.info(f"**Calculated Overall Binding Intensity:** Estimated at {res['Target Binding (pKd)']:.2f} pKd.")

        st.subheader(" Feature Driver Explanations")
        imp_cols = st.columns(2) if res_b else [st.container()]
        desc_labels = ["MolWt", "MolLogP", "TPSA", "H-Donors", "H-Acceptors", "RotatableBonds", "Rings", "AromaticRings", "FractionCSP3", "HeavyAtoms"]
        for idx, res in enumerate(results_list):
            with imp_cols[idx]:
                st.markdown(f"**Descriptor Weights Driving Prediction ({res['Name']}):**")
                raw_importances = models["liver"].feature_importances_[:10]
                feat_df = pd.DataFrame({"Descriptor": desc_labels, "Weight Intensity": raw_importances}).sort_values("Weight Intensity", ascending=True)
                fig_bar = go.Figure(go.Bar(x=feat_df["Weight Intensity"], y=feat_df["Descriptor"], orientation='h', marker_color='#4A90E2'))
                fig_bar.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
               
                st.plotly_chart(fig_bar, use_container_width=True, key=f"feat_driver_chart_{idx}")

        st.subheader(" Visual Fingerprint Profiles")
        fig = go.Figure()
        metrics = ["Liver Risk", "Placenta Risk", "Kidney Risk"]
        for res in results_list:
            fig.add_trace(go.Scatterpolar(r=[res[m] for m in metrics], theta=metrics, fill='toself', name=res['Name']))
        fig.update_layout(title="Relative Risk Geometry Mapping", polar=dict(radialaxis=dict(visible=True, range=[0, 10])))
        st.plotly_chart(fig, use_container_width=True)

        
        st.subheader(" Estimated Plasma Availability Profile")
        fig_pk = go.Figure()
        pk_export_data = {}
        
        for res in results_list:
            t, c = simulate_pk_profile(res, sim_hrs)
            fig_pk.add_trace(go.Scatter(x=t, y=c, name=res['Name'], line=dict(width=3)))
            pk_export_data["Time (hrs)"] = t
            pk_export_data[f"{res['Name']}_Conc"] = c

        fig_pk.add_hrect(y0=60, y1=100, fillcolor="red", opacity=0.1, line_width=0)
        fig_pk.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Fetal Toxicity Ceiling Threshold")
        
        fig_pk.update_layout(xaxis_title="Time Elapsed (Hours)", yaxis_title="Calculated Value Concentration (mg/L)", template="plotly_white")
        st.plotly_chart(fig_pk, use_container_width=True)

        if db_predictions is not None and not db_predictions.empty:
            st.subheader("🔍 Structural Homology Search (Top 3 Chemical Matches in Library)")
            sim_cols = st.columns(2) if res_b else [st.container()]
            for idx, res in enumerate(results_list):
                with sim_cols[idx]:
                    st.markdown(f"**Homologues matching {res['Name']}:**")
                    current_mol = Chem.MolFromSmiles(res["SMILES_String"])
                    if current_mol:
                        current_fp = AllChem.GetMorganFingerprintAsBitVect(current_mol, radius=2, nBits=1024)
                        similarities = []
                        for db_smi in db_predictions["SMILES"].head(300).values:
                            try:
                                db_mol = Chem.MolFromSmiles(db_smi)
                                if db_mol:
                                    db_fp = AllChem.GetMorganFingerprintAsBitVect(db_mol, radius=2, nBits=1024)
                                    sim = DataStructs.TanimotoSimilarity(current_fp, db_fp)
                                    similarities.append((db_smi, sim))
                            except:
                                pass
                        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
                        for rank, (smi_match, coef) in enumerate(similarities):
                            st.caption(f"**Rank {rank+1} Match** (Tanimoto Similarity Index: {coef:.3f})")
                           
                            st.code(smi_match, language="text", key=f"homology_code_{idx}_{rank}")

        st.subheader(" Export Reports")
        csv_pred = df_comp.to_csv(index=False).encode('utf-8')
        csv_pk = pd.DataFrame(pk_export_data).to_csv(index=False).encode('utf-8')
        
        d1, d2 = st.columns(2)
        d1.download_button("Download Predictions Report", data=csv_pred, file_name="organ_predictions.csv", mime="text/csv")
        d2.download_button("Download PK Curve Data", data=csv_pk, file_name="pk_simulation_matrix.csv", mime="text/csv")