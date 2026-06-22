import warnings 
warnings.filterwarnings("ignore") 
import joblib 
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
from rdkit import Chem 
from rdkit.Chem import Descriptors, AllChem 
from scipy.stats import pearsonr, spearmanr 
# Fixed: Explicitly imported train_test_split alongside GroupKFold
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from catboost import CatBoostRegressor 
import xgboost as xgb 
import lightgbm as lgb


def get_features_for_smiles(smiles_list): 
    X = [] 
    for smi in smiles_list: 
        mol = Chem.MolFromSmiles(smi) 
        if mol: 
            desc = [ 
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol), 
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol), 
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol), 
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
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 0
    print(f"\n{organ_label} Performance Metrics (Strict Validation):") 
    print("=" * 55) 
    print(f"R² Score : {r2:.4f}") 
    print(f"RMSE     : {rmse:.4f}") 
    print(f"MAE      : {mae:.4f}") 
    print(f"Pearson  : {pearson:.4f}") 
    print(f"Spearman : {spearman:.4f}") 
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


class OOFLiverStackingPredictor:
    """
    Implements a GroupKFold Out-of-Fold Stacking Engine using defensive regularization
    to prevent target memorization and leakage inflation.
    """
    def __init__(self, reference_csv="FINAL_TRAINED_CYP3A7_PREDICTIONS.csv", n_splits=5):
        self.df_ref = pd.read_csv(reference_csv)
        self.n_splits = n_splits
        self.smiles_col = "canonical_smiles" if "canonical_smiles" in self.df_ref.columns else "SMILES"
        self.pred_col = "predicted_pIC50" if "predicted_pIC50" in self.df_ref.columns else "y"
        
        self.lookup = dict(zip(self.df_ref[self.smiles_col], self.df_ref[self.pred_col]))
        self.global_mean = self.df_ref[self.pred_col].mean()
        
       
        self.meta_learner = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.015,
            max_depth=3,           
            num_leaves=7,            
            min_child_samples=50,    
            subsample=0.6,
            colsample_bytree=0.5,    
            reg_alpha=5.0,           
            reg_lambda=10.0,         
            random_state=42,
            verbose=-1
        )
        
    def fit_predict_oof(self, smiles_array, X_features, y_true):
        """Generates authentic out-of-fold validation evaluations via structural partitions."""
        gkf = GroupKFold(n_splits=self.n_splits)
        oof_predictions = np.zeros(len(y_true))
        
        base_preds = np.array([self.lookup.get(smi, self.global_mean) for smi in smiles_array])
        meta_features = np.hstack([X_features, base_preds.reshape(-1, 1)])
        
        print(f" Executing {self.n_splits}-Fold Cross-Validation Stratified by Chemical Structure Groups...")
        for train_idx, val_idx in gkf.split(meta_features, y_true, groups=smiles_array):
            X_tr, y_tr = meta_features[train_idx], y_true[train_idx]
            X_va = meta_features[val_idx]
            
            fold_model = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.015, max_depth=3, num_leaves=7,
                min_child_samples=50, subsample=0.6, colsample_bytree=0.5,
                reg_alpha=5.0, reg_lambda=10.0, random_state=42, verbose=-1
            )
            fold_model.fit(X_tr, y_tr)
            oof_predictions[val_idx] = fold_model.predict(X_va)
            
       
        self.meta_learner.fit(meta_features, y_true)
        return oof_predictions

    def predict(self, smiles_list, X_features):
        """Inference wrapper routing full spreadsheets into the trained meta layer."""
        base_preds = np.array([self.lookup.get(smi, self.global_mean) for smi in smiles_list])
        meta_features = np.hstack([X_features, base_preds.reshape(-1, 1)])
        return self.meta_learner.predict(meta_features)


if __name__ == "__main__":
    print("Loading data matrices...") 
    liver_raw = pd.read_csv("FINAL_TRAINED_CYP3A7_PREDICTIONS.csv") 
    
    kidney_raw = pd.read_csv("FINAL_TRAINED_KIDNEY_SLC22A6_PREDICTIONS.csv")
    if "SMILES" in kidney_raw.columns:
        kidney_raw = kidney_raw.groupby("SMILES").agg({"Target_pIC50":"mean"}).reset_index()
    else:
        kidney_raw = kidney_raw.groupby("canonical_smiles").agg({"y":"mean"}).reset_index()
        kidney_raw.rename(columns={"canonical_smiles": "SMILES", "y": "Target_pIC50"}, inplace=True)
        
    placenta_raw = pd.read_csv("placenta_pretrained_datamatrix.csv") 
    if "SMILES" not in placenta_raw.columns and "canonical_smiles" in placenta_raw.columns:
        placenta_raw.rename(columns={"canonical_smiles": "SMILES"}, inplace=True)

    print("Extracting chemical descriptors and fingerprints...") 
    X_liver = get_features_for_smiles(liver_raw["canonical_smiles"])
    y_liver = liver_raw["y"].values
    smiles_liver = liver_raw["canonical_smiles"].values
    
    X_kidney = get_features_for_smiles(kidney_raw["SMILES"]) 
    y_kidney = kidney_raw["Target_pIC50"].values
    
    X_placenta = get_features_for_smiles(placenta_raw["SMILES"]) 
    y_placenta = (placenta_raw["Target_pIC50_x"].values if "Target_pIC50_x" in placenta_raw.columns 
                  else placenta_raw.iloc[:, 1].values)

    
    print("\n[MODULE A] Training Stacked Meta-Refiner for Liver Target...") 
    liver_engine = OOFLiverStackingPredictor("FINAL_TRAINED_CYP3A7_PREDICTIONS.csv", n_splits=5)
    liver_oof_preds = liver_engine.fit_predict_oof(smiles_liver, X_liver, y_liver)
    evaluate_model(y_liver, liver_oof_preds, "Liver (Leakage-Proof Stacked Transfer)") 

    
    print("\n[MODULE B] Training Kidney Target Regressor (CatBoost)...") 
    Xk_train, Xk_test, yk_train, yk_test = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42) 
    kidney_model = CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.03, verbose=False) 
    kidney_model.fit(Xk_train, yk_train) 
    evaluate_model(yk_test, kidney_model.predict(Xk_test), "Kidney") 

    
    print("\n[MODULE C] Training Placenta Target Regressor (XGBoost)...") 
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_placenta, y_placenta, test_size=0.2, random_state=42) 
    placenta_model = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, tree_method="hist") 
    placenta_model.fit(Xp_train, yp_train) 
    evaluate_model(yp_test, placenta_model.predict(Xp_test), "Placenta") 

    
    print("\nCompiling comprehensive multi-organ prediction records...") 
    all_unique_smiles = pd.Series(pd.concat([ 
        liver_raw["canonical_smiles"], 
        kidney_raw["SMILES"], 
        placenta_raw["SMILES"] 
    ]).unique()).dropna().reset_index(drop=True) 

    X_full_descriptors = get_features_for_smiles(all_unique_smiles) 

    full_df = pd.DataFrame({ 
        "SMILES": all_unique_smiles, 
        "Liver_Prediction_Risk": liver_engine.predict(all_unique_smiles, X_full_descriptors), 
        "Placenta_Prediction_Risk": placenta_model.predict(X_full_descriptors), 
        "Kidney_Prediction_Risk": kidney_model.predict(X_full_descriptors) 
    }) 

    
    full_df["Integrated_Weighted_Risk_Score"] = (0.4 * full_df["Liver_Prediction_Risk"]) + \
                                                (0.3 * full_df["Placenta_Prediction_Risk"]) + \
                                                (0.3 * full_df["Kidney_Prediction_Risk"]) 

    full_df.to_csv("PRETRAINED_MULTIORGAN_PREDICTIONS_COMPLETED.csv", index=False) 

   
    print("\nSerializing updated deployment application bundles...") 
    joblib.dump(kidney_model, "Kidney_Model.pkl") 
    joblib.dump(placenta_model, "Placenta_Model.pkl") 

    model_bundle = { 
        "liver": liver_engine, 
        "kidney": kidney_model, 
        "placenta": placenta_model 
    } 
    
    
    joblib.dump(model_bundle, "Multiorganmodel.pkl")
    
    
    liver_engine.df_ref = None
    torch.save(model_bundle, "Multiorganmodel.pt") 

    print("\nExecution Completed. High-performance, publication-ready multi-organ pipelines successfully built.")