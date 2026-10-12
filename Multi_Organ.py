from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


class PlacentaStackedConformalRegressor:
    def __init__(self, lgb_opts, cat_opts, rf_opts):
        self.lgb_opts = lgb_opts
        self.cat_opts = cat_opts
        self.rf_opts = rf_opts
        self.meta_learner = Ridge(alpha=1.0, positive=True)
        self.calibration_errors = None
        self.bit_cols = None
        self.non_zero_bit_mask = None
        self.fold_lgb_models = []
        self.fold_cat_models = []
        self.fold_rf_models = []

    def _prepare_features(self, X, is_training=False):
        bits = X.values
        if is_training:
            self.bit_cols = list(X.columns)
            bit_variances = np.var(bits, axis=0)
            self.non_zero_bit_mask = bit_variances > 0.005
        return bits[:, self.non_zero_bit_mask]

    def predict_with_confidence(self, X_target_raw, confidence=0.90):
        X_target = self._prepare_features(X_target_raw, is_training=False)
        lgb_preds = np.mean(
            [model.predict(X_target) for model in self.fold_lgb_models], axis=0
        )
        cat_preds = np.mean(
            [model.predict(X_target) for model in self.fold_cat_models], axis=0
        )
        rf_preds = np.mean(
            [model.predict(X_target) for model in self.fold_rf_models], axis=0
        )
        target_matrix = np.column_stack([lgb_preds, cat_preds, rf_preds])
        preds = self.meta_learner.predict(target_matrix)

        significance_level = 1.0 - confidence
        q_index = int(
            np.ceil((1.0 - significance_level) * (len(self.calibration_errors) + 1))
        ) - 1
        q_index = min(max(q_index, 0), len(self.calibration_errors) - 1)
        error_bound = np.sort(self.calibration_errors)[q_index]
        return preds, float(error_bound)


class PublicationStackedConformalRegressor:
    def __init__(self, lgb_opts, cat_opts, rf_opts):
        self.lgb_opts = lgb_opts
        self.cat_opts = cat_opts
        self.rf_opts = rf_opts
        self.scaler_desc = StandardScaler()
        self.meta_learner = Ridge(alpha=1.5, positive=True)
        self.calibration_errors = None
        self.desc_cols = None
        self.bit_cols = None
        self.non_zero_bit_mask = None
        self.fold_lgb_models = []
        self.fold_cat_models = []
        self.fold_rf_models = []

    def _prepare_features(self, X, is_training=False):
        if is_training:
            self.desc_cols = [c for c in X.columns if not c.startswith("Bit_")]
            self.bit_cols = [c for c in X.columns if c.startswith("Bit_")]
            scaled_desc = self.scaler_desc.fit_transform(X[self.desc_cols])
            bits = X[self.bit_cols].values
            bit_variances = np.var(bits, axis=0)
            self.non_zero_bit_mask = bit_variances > 0.005
        else:
            scaled_desc = self.scaler_desc.transform(X[self.desc_cols])
            bits = X[self.bit_cols].values
        filtered_bits = bits[:, self.non_zero_bit_mask]
        return np.hstack([scaled_desc, filtered_bits])

    def predict_with_confidence(self, X_target_raw, confidence=0.90):
        X_target = self._prepare_features(X_target_raw, is_training=False)
        lgb_preds = np.mean(
            [model.predict(X_target) for model in self.fold_lgb_models], axis=0
        )
        cat_preds = np.mean(
            [model.predict(X_target) for model in self.fold_cat_models], axis=0
        )
        rf_preds = np.mean(
            [model.predict(X_target) for model in self.fold_rf_models], axis=0
        )
        target_matrix = np.column_stack([lgb_preds, cat_preds, rf_preds])
        preds = self.meta_learner.predict(target_matrix)

        significance_level = 1.0 - confidence
        q_index = int(
            np.ceil((1.0 - significance_level) * (len(self.calibration_errors) + 1))
        ) - 1
        q_index = min(max(q_index, 0), len(self.calibration_errors) - 1)
        error_bound = np.sort(self.calibration_errors)[q_index]
        return preds, float(error_bound)


class PublicationTransferNetwork(nn.Module):
    def __init__(self, input_dim: int = 2215):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.05),
        )
        self.res_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.05),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )
        self.activation = nn.Mish()
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Dropout(0.05),
        )
        self.regressor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation(out + self.res_block(out))
        features = self.feature_extractor(out)
        return self.regressor_head(features)


SHARED_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtoms",
]


def canonicalize_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _morgan_array(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def shared_features_for_smiles(smiles_list: Sequence[str]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        desc = np.asarray(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                float(mol.GetNumHeavyAtoms()),
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([desc, _morgan_array(mol, 2048)]))
    return np.asarray(rows, dtype=np.float32)


def liver_hybrid_features(smiles_list: Sequence[str]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid liver SMILES: {smi}")
        morgan = _morgan_array(mol, 2048)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.asarray(
            [int(b) for b in maccs_fp.ToBitString()], dtype=np.float32
        )
        if maccs.shape[0] != 167:
            raise RuntimeError(f"Expected 167 MACCS bits, got {maccs.shape[0]}")
        rows.append(np.concatenate([morgan, maccs]))
    return np.asarray(rows, dtype=np.float32)


def placenta_features_for_smiles(
    smiles_list: Sequence[str], model: PlacentaStackedConformalRegressor
) -> pd.DataFrame:
    bit_cols = list(model.bit_cols)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid placenta SMILES: {smi}")
        fp = _morgan_array(mol, 2048)
        row = {col: fp[int(col.split("_")[-1])] for col in bit_cols}
        rows.append(row)
    return pd.DataFrame(rows, columns=bit_cols)


def kidney_features_for_smiles(
    smiles_list: Sequence[str], model: PublicationStackedConformalRegressor
) -> pd.DataFrame:
    descriptor_functions = {
        "MW": Descriptors.MolWt,
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "MolLogP": Descriptors.MolLogP,
        "HBD": Descriptors.NumHDonors,
        "NumHDonors": Descriptors.NumHDonors,
        "HBA": Descriptors.NumHAcceptors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "TPSA": Descriptors.TPSA,
        "RotB": Descriptors.NumRotatableBonds,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
    }

    bit_cols = list(model.bit_cols)
    desc_cols = list(model.desc_cols)
    max_bit = max(int(c.split("_")[-1]) for c in bit_cols) + 1

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid kidney SMILES: {smi}")
        fp = _morgan_array(mol, max_bit)
        row = {}
        for col in desc_cols:
            if col not in descriptor_functions:
                raise KeyError(
                    f"Kidney model expects descriptor '{col}', but no generator is defined."
                )
            row[col] = float(descriptor_functions[col](mol))
        for col in bit_cols:
            row[col] = fp[int(col.split("_")[-1])]
        rows.append(row)

    return pd.DataFrame(rows, columns=desc_cols + bit_cols)


def _prepare_raw_cyp3a7(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    df["numeric_val"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["numeric_val", "canonical_smiles"])
    df = df[df["numeric_val"] > 0].copy()
    df["y"] = -np.log10(df["numeric_val"] * 1e-9)
    df = df[(df["y"] >= 2) & (df["y"] <= 12)]
    return df[["canonical_smiles", "y"]].reset_index(drop=True)


def reconstruct_liver_scalers(
    cyp3a4_csv: Path,
    cyp3a7_csv: Path,
    raw_cyp3a7_csv: Path,
    cache_file: Path | None = None,
) -> Tuple[StandardScaler, StandardScaler]:
    if cache_file is not None and cache_file.exists():
        cached = joblib.load(cache_file)
        return cached["x_scaler"], cached["y_scaler"]

    print("Reconstructing liver feature/target scalers from original source datasets...")
    df_3a4 = pd.read_csv(cyp3a4_csv)
    valid_3a4 = []
    valid_smiles_3a4 = []
    for smi in df_3a4["canonical_smiles"].astype(str):
        if Chem.MolFromSmiles(smi) is not None:
            valid_smiles_3a4.append(smi)
    X_3a4 = liver_hybrid_features(valid_smiles_3a4)
    x_scaler = StandardScaler().fit(X_3a4)

    df_3a7_base = pd.read_csv(cyp3a7_csv)
    df_raw_3a7 = _prepare_raw_cyp3a7(raw_cyp3a7_csv)

    df_base_train, _ = train_test_split(
        df_3a7_base, test_size=0.2, random_state=RANDOM_STATE
    )
    df_raw_train, _ = train_test_split(
        df_raw_3a7, test_size=0.2, random_state=RANDOM_STATE
    )
    df_train_final = pd.concat([df_base_train, df_raw_train], ignore_index=True)

    valid_y = []
    for smi, y in zip(df_train_final["canonical_smiles"], df_train_final["y"]):
        if Chem.MolFromSmiles(str(smi)) is not None:
            valid_y.append(float(y))
    y_scaler = StandardScaler().fit(np.asarray(valid_y).reshape(-1, 1))

    if cache_file is not None:
        joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, cache_file)
        print(f"Saved reconstructed liver scalers: {cache_file}")

    return x_scaler, y_scaler


def load_liver_model(checkpoint: Path, device: torch.device) -> PublicationTransferNetwork:
    model = PublicationTransferNetwork(input_dim=2215).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_liver_independent(
    smiles: Sequence[str],
    model: PublicationTransferNetwork,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    out = []
    for start in range(0, len(smiles), batch_size):
        batch_smiles = smiles[start : start + batch_size]
        X = liver_hybrid_features(batch_smiles)
        X = x_scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            pred_scaled = (
                model(torch.tensor(X, dtype=torch.float32, device=device))
                .cpu()
                .numpy()
                .reshape(-1, 1)
            )
        out.append(y_scaler.inverse_transform(pred_scaled).ravel())
    return np.concatenate(out)


class OrganMetaRefiner:

    def __init__(self, organ_name: str, n_splits: int = 5):
        self.organ_name = organ_name
        self.n_splits = n_splits
        self.model = self._new_model()

    @staticmethod
    def _new_model():
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.55,
            reg_alpha=2.0,
            reg_lambda=5.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    @staticmethod
    def _fusion_matrix(X_chem: np.ndarray, base_predictions: np.ndarray) -> np.ndarray:
        base = np.asarray(base_predictions, dtype=np.float32).reshape(-1, 1)
        return np.hstack([X_chem.astype(np.float32), base])

    def fit_oof(
        self,
        smiles: Sequence[str],
        X_chem: np.ndarray,
        base_predictions: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        X_fusion = self._fusion_matrix(X_chem, base_predictions)
        groups = np.asarray(smiles, dtype=object)
        y = np.asarray(y_true, dtype=np.float64)
        oof = np.zeros(len(y), dtype=np.float64)

        splitter = GroupKFold(n_splits=self.n_splits)
        print(f"  {self.organ_name}: {self.n_splits}-fold GroupKFold stage-2 refinement")
        for fold, (tr, va) in enumerate(splitter.split(X_fusion, y, groups=groups), 1):
            m = self._new_model()
            m.fit(X_fusion[tr], y[tr])
            oof[va] = m.predict(X_fusion[va])
            print(f"    fold {fold}/{self.n_splits} complete")

        self.model = self._new_model()
        self.model.fit(X_fusion, y)
        return oof

    def predict(self, X_chem: np.ndarray, base_predictions: np.ndarray) -> np.ndarray:
        X_fusion = self._fusion_matrix(X_chem, base_predictions)
        return self.model.predict(X_fusion)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan,
        "Spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else np.nan,
    }
    print(f"\n{label}")
    print("-" * len(label))
    for k, v in metrics.items():
        print(f"{k:9s}: {v:.4f}")
    return metrics


def save_observed_predicted(y_true, y_pred, organ, out_dir):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.55, s=22)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    r2 = r2_score(y_true, y_pred)
    pr = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    sr = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    ax.set_xlabel("Observed pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title(f"{organ}: Observed vs Predicted")
    ax.text(0.04, 0.96, f"R² = {r2:.4f}\nPearson r = {pr:.4f}\nSpearman ρ = {sr:.4f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    path = out_dir / f"{organ.lower()}_observed_vs_predicted.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_residual_plot(y_true, y_pred, organ, out_dir):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.55, s=22)
    ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted pIC50")
    ax.set_ylabel("Residual (Observed - Predicted)")
    ax.set_title(f"{organ}: Residual Plot")
    fig.tight_layout()
    path = out_dir / f"{organ.lower()}_residual_plot.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_r2_comparison(metrics, out_dir):
    organs = ["Liver", "Placenta", "Kidney"]
    stage2 = [metrics["liver_stage2_oof"]["R2"], metrics["placenta_stage2_oof"]["R2"], metrics["kidney_stage2_oof"]["R2"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(organs, stage2)
    ax.set_ylabel("R²")
    ax.set_title("Integrated Multi-Organ R² Comparison")
    ax.set_ylim(min(0.0, min(stage2) - 0.1), max(1.0, max(stage2) + 0.1))
    for bar, value in zip(bars, stage2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    path = out_dir / "multi_organ_r2_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_stage_comparison(metrics, out_dir):
    organs = ["Liver", "Placenta", "Kidney"]
    stage1 = [metrics["liver_stage1"]["R2"], metrics["placenta_stage1"]["R2"], metrics["kidney_stage1"]["R2"]]
    stage2 = [metrics["liver_stage2_oof"]["R2"], metrics["placenta_stage2_oof"]["R2"], metrics["kidney_stage2_oof"]["R2"]]
    x = np.arange(len(organs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, stage1, width, label="Stage 1")
    b2 = ax.bar(x + width / 2, stage2, width, label="Stage 2")
    ax.set_xticks(x)
    ax.set_xticklabels(organs)
    ax.set_ylabel("R²")
    ax.set_title("Independent vs Integrated Model Performance")
    ax.legend()
    low = min(stage1 + stage2)
    high = max(stage1 + stage2)
    ax.set_ylim(min(0.0, low - 0.1), max(1.0, high + 0.1))
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = out_dir / "stage1_vs_stage2_r2.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def prepare_liver_stage2(liver_predictions_csv: Path):
    df = pd.read_csv(liver_predictions_csv)
    df = df.dropna(subset=["canonical_smiles", "y", "predicted_pIC50"]).copy()
    df["SMILES"] = df["canonical_smiles"].map(canonicalize_smiles)
    df = df.dropna(subset=["SMILES"])


    return (
        df["SMILES"].tolist(),
        df["y"].to_numpy(float),
        df["predicted_pIC50"].to_numpy(float),
    )


def prepare_kidney_stage2(kidney_predictions_csv: Path):
    df = pd.read_csv(kidney_predictions_csv)
    df = df.dropna(subset=["SMILES", "Target_pIC50", "predicted_pIC50"]).copy()
    df["SMILES"] = df["SMILES"].map(canonicalize_smiles)
    df = df.dropna(subset=["SMILES"])

    df = (
        df.groupby("SMILES", as_index=False)
        .agg({"Target_pIC50": "mean", "predicted_pIC50": "mean"})
    )
    return (
        df["SMILES"].tolist(),
        df["Target_pIC50"].to_numpy(float),
        df["predicted_pIC50"].to_numpy(float),
    )


def prepare_placenta_stage2(
    placenta_matrix_csv: Path,
    placenta_model: PlacentaStackedConformalRegressor,
):
    df = pd.read_csv(placenta_matrix_csv)
    target_col = "Target_pIC50_x" if "Target_pIC50_x" in df.columns else "Target_pIC50"
    df = df.dropna(subset=["SMILES", target_col]).copy()
    df["SMILES"] = df["SMILES"].map(canonicalize_smiles)
    df = df.dropna(subset=["SMILES"])


    X_base = df[list(placenta_model.bit_cols)].copy()
    base_pred, _ = placenta_model.predict_with_confidence(X_base, confidence=0.90)
    return (
        df["SMILES"].tolist(),
        df[target_col].to_numpy(float),
        np.asarray(base_pred, dtype=float),
    )



def evaluate_liver_independent_holdout(liver_predictions_csv: Path):
    df = pd.read_csv(liver_predictions_csv)
    required = ["y", "predicted_pIC50", "dataset_assignment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Liver prediction file is missing columns: {missing}")
    holdout = df[df["dataset_assignment"].astype(str).str.strip().eq("Validation_Holdout")].copy()
    holdout = holdout.dropna(subset=["y", "predicted_pIC50"])
    if holdout.empty:
        raise ValueError("No liver Validation_Holdout rows were found.")
    return evaluate(
        holdout["y"].to_numpy(float),
        holdout["predicted_pIC50"].to_numpy(float),
        f"Liver independent model - original holdout (n={len(holdout):,})",
    )


def evaluate_kidney_independent_holdout(kidney_predictions_csv: Path):
    df = pd.read_csv(kidney_predictions_csv)
    required = ["Target_pIC50", "predicted_pIC50", "dataset_assignment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kidney prediction file is missing columns: {missing}")
    holdout = df[df["dataset_assignment"].astype(str).str.strip().eq("Validation_Holdout")].copy()
    holdout = holdout.dropna(subset=["Target_pIC50", "predicted_pIC50"])
    if holdout.empty:
        raise ValueError("No kidney Validation_Holdout rows were found.")
    return evaluate(
        holdout["Target_pIC50"].to_numpy(float),
        holdout["predicted_pIC50"].to_numpy(float),
        f"Kidney independent model - original holdout (n={len(holdout):,})",
    )


def evaluate_placenta_independent_holdout(
    placenta_matrix_csv: Path,
    placenta_model: PlacentaStackedConformalRegressor,
):
    df = pd.read_csv(placenta_matrix_csv)
    target_col = "Target_pIC50_x" if "Target_pIC50_x" in df.columns else "Target_pIC50"
    if target_col not in df.columns:
        raise ValueError("Placenta matrix does not contain a Target_pIC50 target column.")
    feature_cols = [c for c in df.columns if c.startswith("Bit_")]
    if not feature_cols:
        raise ValueError("Placenta matrix does not contain Bit_* fingerprint columns.")
    clean = df.dropna(subset=[target_col]).copy()
    y = clean[target_col].to_numpy(float)
    quantiles = np.percentile(y, [20, 40, 60, 80])
    y_binned = np.digitize(y, bins=quantiles)
    indices = np.arange(len(clean))
    _, test_pos = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y_binned,
    )
    X_test = clean.iloc[test_pos][feature_cols].copy()
    y_test = clean.iloc[test_pos][target_col].to_numpy(float)
    preds, _ = placenta_model.predict_with_confidence(X_test, confidence=0.90)
    return evaluate(
        y_test,
        np.asarray(preds, dtype=float),
        f"Placenta independent model - reconstructed original holdout (n={len(y_test):,})",
    )

def build_union_smiles(*smiles_sets: Iterable[str]) -> List[str]:
    seen = set()
    union = []
    for seq in smiles_sets:
        for smi in seq:
            can = canonicalize_smiles(smi)
            if can is not None and can not in seen:
                seen.add(can)
                union.append(can)
    return union


def main(args):
    root = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def p(name: str) -> Path:
        return root / name

    required = [
        args.liver_checkpoint,
        args.cyp3a4_csv,
        args.cyp3a7_csv,
        args.raw_cyp3a7_csv,
        args.liver_predictions_csv,
        args.placenta_model,
        args.placenta_matrix_csv,
        args.kidney_model,
        args.kidney_predictions_csv,
    ]
    missing = [name for name in required if not p(name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files in data directory:\n  " + "\n  ".join(missing)
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 78)
    print("RECONSTRUCTED TWO-STAGE HYBRID MULTI-ORGAN PIPELINE")
    print("=" * 78)
    print(f"Data directory : {root}")
    print(f"Output directory: {out_dir}")
    print(f"Torch device    : {device}")


    print("\n[STAGE 1] Loading independently trained organ models...")
    placenta_model: PlacentaStackedConformalRegressor = joblib.load(p(args.placenta_model))
    kidney_model: PublicationStackedConformalRegressor = joblib.load(p(args.kidney_model))
    liver_model = load_liver_model(p(args.liver_checkpoint), device)

    scaler_cache = out_dir / "liver_reconstructed_scalers.pkl"
    x_scaler, y_scaler = reconstruct_liver_scalers(
        p(args.cyp3a4_csv),
        p(args.cyp3a7_csv),
        p(args.raw_cyp3a7_csv),
        cache_file=scaler_cache,
    )


    print("\n[STAGE 2] Preparing independent organ outputs for meta-refinement...")
    liver_smiles, liver_y, liver_base = prepare_liver_stage2(p(args.liver_predictions_csv))
    kidney_smiles, kidney_y, kidney_base = prepare_kidney_stage2(p(args.kidney_predictions_csv))
    placenta_smiles, placenta_y, placenta_base = prepare_placenta_stage2(
        p(args.placenta_matrix_csv), placenta_model
    )

    print(f"  Liver stage-2 compounds   : {len(liver_smiles):,}")
    print(f"  Placenta stage-2 compounds: {len(placenta_smiles):,}")
    print(f"  Kidney stage-2 compounds  : {len(kidney_smiles):,}")

    print("\nGenerating common 2058-dimensional chemical representation...")
    X_liver_chem = shared_features_for_smiles(liver_smiles)
    X_placenta_chem = shared_features_for_smiles(placenta_smiles)
    X_kidney_chem = shared_features_for_smiles(kidney_smiles)


    print("\nTraining leakage-resistant second-stage LightGBM refiners...")
    liver_refiner = OrganMetaRefiner("Liver", n_splits=args.folds)
    placenta_refiner = OrganMetaRefiner("Placenta", n_splits=args.folds)
    kidney_refiner = OrganMetaRefiner("Kidney", n_splits=args.folds)

    liver_oof = liver_refiner.fit_oof(liver_smiles, X_liver_chem, liver_base, liver_y)
    placenta_oof = placenta_refiner.fit_oof(
        placenta_smiles, X_placenta_chem, placenta_base, placenta_y
    )
    kidney_oof = kidney_refiner.fit_oof(kidney_smiles, X_kidney_chem, kidney_base, kidney_y)

    metrics = {
        "liver_stage1": evaluate_liver_independent_holdout(p(args.liver_predictions_csv)),
        "liver_stage2_oof": evaluate(liver_y, liver_oof, "Liver stage-2 OOF meta-refiner"),
        "placenta_stage1": evaluate_placenta_independent_holdout(
            p(args.placenta_matrix_csv), placenta_model
        ),
        "placenta_stage2_oof": evaluate(
            placenta_y, placenta_oof, "Placenta stage-2 OOF meta-refiner"
        ),
        "kidney_stage1": evaluate_kidney_independent_holdout(p(args.kidney_predictions_csv)),
        "kidney_stage2_oof": evaluate(kidney_y, kidney_oof, "Kidney stage-2 OOF meta-refiner"),
    }

    with open(out_dir / "MULTIORGAN_STAGE2_METRICS.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    graph_files = [
        save_observed_predicted(liver_y, liver_oof, "Liver", out_dir),
        save_residual_plot(liver_y, liver_oof, "Liver", out_dir),
        save_observed_predicted(placenta_y, placenta_oof, "Placenta", out_dir),
        save_residual_plot(placenta_y, placenta_oof, "Placenta", out_dir),
        save_observed_predicted(kidney_y, kidney_oof, "Kidney", out_dir),
        save_residual_plot(kidney_y, kidney_oof, "Kidney", out_dir),
        save_r2_comparison(metrics, out_dir),
        save_stage_comparison(metrics, out_dir),
    ]


    joblib.dump(
        {
            "liver": liver_refiner,
            "placenta": placenta_refiner,
            "kidney": kidney_refiner,
            "shared_feature_dim": 2058,
            "fusion_feature_dim": 2059,
        },
        out_dir / "MULTIORGAN_STAGE2_REFINERS.pkl",
    )


    print("\nGenerating integrated predictions across the union chemical space...")
    union_smiles = build_union_smiles(liver_smiles, placenta_smiles, kidney_smiles)
    print(f"  Unique union molecules: {len(union_smiles):,}")

    X_union_chem = shared_features_for_smiles(union_smiles)


    liver_union_base = predict_liver_independent(
        union_smiles, liver_model, x_scaler, y_scaler, device, batch_size=args.batch_size
    )

    placenta_union_X = placenta_features_for_smiles(union_smiles, placenta_model)
    placenta_union_base, placenta_margin = placenta_model.predict_with_confidence(
        placenta_union_X, confidence=0.90
    )

    kidney_union_X = kidney_features_for_smiles(union_smiles, kidney_model)
    kidney_union_base, kidney_margin = kidney_model.predict_with_confidence(
        kidney_union_X, confidence=0.90
    )


    liver_union_final = liver_refiner.predict(X_union_chem, liver_union_base)
    placenta_union_final = placenta_refiner.predict(X_union_chem, placenta_union_base)
    kidney_union_final = kidney_refiner.predict(X_union_chem, kidney_union_base)

    results = pd.DataFrame(
        {
            "SMILES": union_smiles,
            "Liver_Stage1_Prediction": liver_union_base,
            "Placenta_Stage1_Prediction": placenta_union_base,
            "Kidney_Stage1_Prediction": kidney_union_base,
            "Placenta_Stage1_90_Lower": np.asarray(placenta_union_base) - placenta_margin,
            "Placenta_Stage1_90_Upper": np.asarray(placenta_union_base) + placenta_margin,
            "Kidney_Stage1_90_Lower": np.asarray(kidney_union_base) - kidney_margin,
            "Kidney_Stage1_90_Upper": np.asarray(kidney_union_base) + kidney_margin,
            "Liver_Integrated_Prediction": liver_union_final,
            "Placenta_Integrated_Prediction": placenta_union_final,
            "Kidney_Integrated_Prediction": kidney_union_final,
        }
    )


    weights = np.asarray([args.liver_weight, args.placenta_weight, args.kidney_weight], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("liver_weight + placenta_weight + kidney_weight must equal 1.0")

    results["Integrated_Weighted_Risk_Score"] = (
        args.liver_weight * results["Liver_Integrated_Prediction"]
        + args.placenta_weight * results["Placenta_Integrated_Prediction"]
        + args.kidney_weight * results["Kidney_Integrated_Prediction"]
    )

    output_csv = out_dir / "PRETRAINED_MULTIORGAN_PREDICTIONS_RECONSTRUCTED.csv"
    results.to_csv(output_csv, index=False)


    manifest = {
        "method": "two_stage_independent_models_plus_groupkfold_lightgbm_meta_refinement",
        "liver_checkpoint": str(p(args.liver_checkpoint)),
        "placenta_model": str(p(args.placenta_model)),
        "kidney_model": str(p(args.kidney_model)),
        "liver_scaler_cache": str(scaler_cache),
        "stage2_refiners": str(out_dir / "MULTIORGAN_STAGE2_REFINERS.pkl"),
        "prediction_csv": str(output_csv),
        "graphs": [str(x) for x in graph_files],
        "weights": {
            "liver": args.liver_weight,
            "placenta": args.placenta_weight,
            "kidney": args.kidney_weight,
        },
        "note": (
            "The weighted score is a downstream summary index. Organ-level stage-2 "
            "predictions are produced by target-specific OOF LightGBM meta-refiners."
        ),
    }
    with open(out_dir / "MULTIORGAN_MODEL_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    print(f"Predictions : {output_csv}")
    print(f"Metrics     : {out_dir / 'MULTIORGAN_STAGE2_METRICS.json'}")
    print(f"Refiners    : {out_dir / 'MULTIORGAN_STAGE2_REFINERS.pkl'}")
    print(f"Manifest    : {out_dir / 'MULTIORGAN_MODEL_MANIFEST.json'}")
    print(f"Graphs      : {len(graph_files)} PNG files saved in {out_dir}")
    print("Independent liver, placenta, and kidney models were loaded, not retrained.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstructed two-stage hybrid multi-organ integration pipeline"
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--cpu", action="store_true", help="Force liver inference to CPU")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)

    parser.add_argument("--liver-checkpoint", default="best_transfer_model.pt")
    parser.add_argument("--cyp3a4-csv", default="MASTER_CYP3A4_ENHANCED.csv")
    parser.add_argument("--cyp3a7-csv", default="MASTER_CYP3A7_ENHANCED.csv")
    parser.add_argument("--raw-cyp3a7-csv", default="CHEMBL_CYP3A7_RAW.csv")
    parser.add_argument(
        "--liver-predictions-csv", default="FINAL_TRAINED_CYP3A7_PREDICTIONS.csv"
    )

    parser.add_argument(
        "--placenta-model", default="pretrained_placenta_stacked_conformal_model.pkl"
    )
    parser.add_argument(
        "--placenta-matrix-csv", default="placenta_pretrained_datamatrix.csv"
    )

    parser.add_argument(
        "--kidney-model", default="pretrained_kidney_stacked_conformal_model.pkl"
    )
    parser.add_argument(
        "--kidney-predictions-csv",
        default="FINAL_TRAINED_KIDNEY_SLC22A6_PREDICTIONS.csv",
    )

    parser.add_argument("--liver-weight", type=float, default=0.40)
    parser.add_argument("--placenta-weight", type=float, default=0.30)
    parser.add_argument("--kidney-weight", type=float, default=0.30)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
