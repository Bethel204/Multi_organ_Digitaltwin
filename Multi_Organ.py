import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb


def get_features_for_smiles(smiles_list):
    X = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc = [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), 
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol), 
                Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol), 
                mol.GetNumHeavyAtoms()
            ]
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            X.append(desc + fp.tolist())
        else:
            X.append([0] * 2058)  
    return np.array(X)

def evaluate_model(y_true, y_pred, organ_label):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    
    print(f"\n{organ_label} Performance Metrics:")
    print("=" * 40)
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}")
    print(f"Pearson  : {pearson:.4f}")
    print(f"Spearman : {spearman:.4f}")


print("Loading datasets...")
liver_raw = pd.read_csv("FINAL_TRAINED_CYP3A7_PREDICTIONS.csv")

kidney_raw = pd.read_csv("FINAL_TRAINED_KIDNEY_SLC22A6_PREDICTIONS.csv").groupby("SMILES").agg({"Target_pIC50":"mean"}).reset_index()
placenta_raw = pd.read_csv("placenta_pretrained_datamatrix.csv")

print("Generating features...")
X_liver = get_features_for_smiles(liver_raw["canonical_smiles"])
y_liver = liver_raw["y"]

X_kidney = get_features_for_smiles(kidney_raw["SMILES"])
y_kidney = kidney_raw["Target_pIC50"]

X_placenta = get_features_for_smiles(placenta_raw["SMILES"])
y_placenta = placenta_raw["Target_pIC50_y"]


Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)
Xk_train, Xk_test, yk_train, yk_test = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_placenta, y_placenta, test_size=0.2, random_state=42)


print("\nTraining Liver Model (LightGBM)...")
liver_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.02, max_depth=10, subsample=0.8, colsample_bytree=0.8)
liver_model.fit(Xl_train, yl_train)
evaluate_model(yl_test, liver_model.predict(Xl_test), "Liver")

print("\nTraining Kidney Model (CatBoost)...")
kidney_model = CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.03, verbose=False)
kidney_model.fit(Xk_train, yk_train)
evaluate_model(yk_test, kidney_model.predict(Xk_test), "Kidney")

print("\nTraining Placenta Model (XGBoost)...")
placenta_model = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, tree_method="hist")
placenta_model.fit(Xp_train, yp_train)
evaluate_model(yp_test, placenta_model.predict(Xp_test), "Placenta")


print("\nGenerating comprehensive spreadsheet calculations...")
all_unique_smiles = pd.Series(pd.concat([
    liver_raw["canonical_smiles"], kidney_raw["SMILES"], placenta_raw["SMILES"]
]).unique()).dropna().reset_index(drop=True)

X_full = get_features_for_smiles(all_unique_smiles)


full_df = pd.DataFrame({
    "SMILES": all_unique_smiles,
    "Liver_Prediction_Risk": liver_model.predict(X_full),
    "Placenta_Prediction_Risk": placenta_model.predict(X_full),
    "Kidney_Prediction_Risk": kidney_model.predict(X_full)
})


full_df["Integrated_Weighted_Risk_Score"] = (0.4 * full_df["Liver_Prediction_Risk"]) + \
                                            (0.3 * full_df["Placenta_Prediction_Risk"]) + \
                                            (0.3 * full_df["Kidney_Prediction_Risk"])

full_df.to_csv("PRETRAINED_MULTIORGAN_PREDICTIONS_COMPLETED.csv", index=False)


print("\nSaving app deployment components...")
# Save individual models
joblib.dump(liver_model, "Liver_Model.pkl")
joblib.dump(kidney_model, "Kidney_Model.pkl")
joblib.dump(placenta_model, "Placenta_Model.pkl")

model_bundle = {
    "liver": liver_model,
    "kidney": kidney_model,
    "placenta": placenta_model
}
joblib.dump(model_bundle, "Multiorganmodel.pt")

print("Done! All models evaluated, CSV created, and deployment models exported.")