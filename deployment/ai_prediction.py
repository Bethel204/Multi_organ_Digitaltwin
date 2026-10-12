from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

import config
from model_components import (
    OrganMetaRefiner,
    PlacentaStackedConformalRegressor,
    PublicationStackedConformalRegressor,
    PublicationTransferNetwork,
)

_main = sys.modules.get("__main__")
for _cls in (OrganMetaRefiner, PlacentaStackedConformalRegressor, PublicationStackedConformalRegressor):
    if _main is not None:
        setattr(_main, _cls.__name__, _cls)


def _morgan_array(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def shared_features_for_smiles(smiles_list: Sequence[str]) -> np.ndarray:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        desc = np.asarray([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
            float(mol.GetNumHeavyAtoms()),
        ], dtype=np.float32)
        rows.append(np.concatenate([desc, _morgan_array(mol, 2048)]))
    return np.asarray(rows, dtype=np.float32)


def liver_hybrid_features(smiles_list: Sequence[str]) -> np.ndarray:
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        morgan = _morgan_array(mol, 2048)
        maccs = np.asarray([int(x) for x in MACCSkeys.GenMACCSKeys(mol).ToBitString()], dtype=np.float32)
        rows.append(np.concatenate([morgan, maccs]))
    return np.asarray(rows, dtype=np.float32)



def _patch_legacy_sklearn_forest(obj):
    
    forests = []
    for attr in ("fold_rf_models",):
        forests.extend(getattr(obj, attr, []) or [])
    for forest in forests:
        if not hasattr(forest, "monotonic_cst"):
            forest.monotonic_cst = None
        for tree in getattr(forest, "estimators_", []) or []:
            if not hasattr(tree, "monotonic_cst"):
                tree.monotonic_cst = None
    return obj

class AIPredictionPipeline:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.stage2_bundle = None
        self.db_predictions = None
        self.liver_model = None
        self.liver_x_scaler = None
        self.liver_y_scaler = None
        self.placenta_model = None
        self.kidney_model = None
        self.metrics = dict(config.REPORTED_STAGE2_R2)
        self.load_errors = []
        self.load_resources()

    @staticmethod
    def canonicalize(smiles: str) -> str:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES structure: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def _safe_load(self, label, path):
        if path is None:
            self.load_errors.append(f"{label}: file not found")
            return None
        try:
            obj = joblib.load(path)
            return _patch_legacy_sklearn_forest(obj)
        except Exception as exc:
            self.load_errors.append(f"{label}: {exc}")
            return None

    def load_resources(self):
        self.stage2_bundle = self._safe_load("Stage-2 refiners", config.STAGE2_REFINERS_PATH)
        self.placenta_model = self._safe_load("Placenta Stage-1 model", config.PLACENTA_MODEL_PATH)
        self.kidney_model = self._safe_load("Kidney Stage-1 model", config.KIDNEY_MODEL_PATH)

        scalers = self._safe_load("Liver scalers", config.LIVER_SCALERS_PATH)
        if scalers:
            self.liver_x_scaler = scalers.get("x_scaler")
            self.liver_y_scaler = scalers.get("y_scaler")

        if config.LIVER_CHECKPOINT_PATH is not None:
            try:
                self.liver_model = PublicationTransferNetwork(2215).to(self.device)
                state = torch.load(config.LIVER_CHECKPOINT_PATH, map_location=self.device, weights_only=True)
                self.liver_model.load_state_dict(state, strict=True)
                self.liver_model.eval()
            except Exception as exc:
                self.load_errors.append(f"Liver Stage-1 model: {exc}")
                self.liver_model = None
        else:
            self.load_errors.append("Liver Stage-1 model: file not found")

        if config.STAGE2_PREDICTIONS_PATH is not None:
            try:
                df = pd.read_csv(config.STAGE2_PREDICTIONS_PATH)
                if "SMILES" in df.columns:
                    df = df.copy()
                    df["SMILES_Clean"] = df["SMILES"].map(self.canonicalize)
                    self.db_predictions = df.drop_duplicates("SMILES_Clean", keep="first")
            except Exception as exc:
                self.load_errors.append(f"Stage-2 prediction cache: {exc}")

        if config.STAGE2_METRICS_PATH is not None:
            try:
                payload = json.loads(Path(config.STAGE2_METRICS_PATH).read_text(encoding="utf-8"))
                for organ in ("liver", "placenta", "kidney"):
                    key = f"{organ}_stage2_oof"
                    if key in payload and "R2" in payload[key]:
                        self.metrics[organ] = float(payload[key]["R2"])
            except Exception:
                pass

    @property
    def stage2_ready(self) -> bool:
        return self.stage2_bundle is not None and all(
            x is not None for x in (
                self.liver_model, self.liver_x_scaler, self.liver_y_scaler,
                self.placenta_model, self.kidney_model,
            )
        )

    def status(self) -> Dict[str, object]:
        return {
            "stage2_ready": self.stage2_ready,
            "prediction_cache_loaded": self.db_predictions is not None,
            "stage2_refiners_path": str(config.STAGE2_REFINERS_PATH) if config.STAGE2_REFINERS_PATH else None,
            "stage2_predictions_path": str(config.STAGE2_PREDICTIONS_PATH) if config.STAGE2_PREDICTIONS_PATH else None,
            "errors": list(self.load_errors),
        }

    def shared_features(self, smiles_list: Sequence[str]) -> np.ndarray:
        return shared_features_for_smiles(smiles_list)

    def molecular_descriptors(self, smiles: str) -> Dict[str, float]:
        smi = self.canonicalize(smiles)
        mol = Chem.MolFromSmiles(smi)
        return {
            "Molecular weight": float(Descriptors.MolWt(mol)),
            "MolLogP": float(Descriptors.MolLogP(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "H-bond donors": float(Descriptors.NumHDonors(mol)),
            "H-bond acceptors": float(Descriptors.NumHAcceptors(mol)),
            "Rotatable bonds": float(Descriptors.NumRotatableBonds(mol)),
            "Ring count": float(Descriptors.RingCount(mol)),
            "Aromatic rings": float(Descriptors.NumAromaticRings(mol)),
            "FractionCSP3": float(Descriptors.FractionCSP3(mol)),
            "Heavy atom count": float(mol.GetNumHeavyAtoms()),
        }

    def _predict_liver_stage1(self, smi: str) -> float:
        X = liver_hybrid_features([smi])
        X = self.liver_x_scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            scaled = self.liver_model(torch.tensor(X, dtype=torch.float32, device=self.device)).cpu().numpy().reshape(-1, 1)
        return float(self.liver_y_scaler.inverse_transform(scaled).ravel()[0])

    def _placenta_frame(self, smi: str) -> pd.DataFrame:
        mol = Chem.MolFromSmiles(smi)
        fp = _morgan_array(mol, 2048)
        cols = list(self.placenta_model.bit_cols)
        return pd.DataFrame([{c: fp[int(c.split("_")[-1])] for c in cols}], columns=cols)

    def _kidney_frame(self, smi: str) -> pd.DataFrame:
        mol = Chem.MolFromSmiles(smi)
        funcs = {
            "MW": Descriptors.MolWt, "MolWt": Descriptors.MolWt,
            "LogP": Descriptors.MolLogP, "MolLogP": Descriptors.MolLogP,
            "HBD": Descriptors.NumHDonors, "NumHDonors": Descriptors.NumHDonors,
            "HBA": Descriptors.NumHAcceptors, "NumHAcceptors": Descriptors.NumHAcceptors,
            "TPSA": Descriptors.TPSA, "RotB": Descriptors.NumRotatableBonds,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
        }
        bit_cols = list(self.kidney_model.bit_cols)
        desc_cols = list(self.kidney_model.desc_cols)
        nbits = max(int(c.split("_")[-1]) for c in bit_cols) + 1
        fp = _morgan_array(mol, nbits)
        row = {c: float(funcs[c](mol)) for c in desc_cols}
        row.update({c: fp[int(c.split("_")[-1])] for c in bit_cols})
        return pd.DataFrame([row], columns=desc_cols + bit_cols)

    def get_stage1_predictions(self, smiles: str) -> Dict[str, float]:
        smi = self.canonicalize(smiles)
        if not all(x is not None for x in (self.liver_model, self.liver_x_scaler, self.liver_y_scaler, self.placenta_model, self.kidney_model)):
            raise RuntimeError("Stage-1 resources are not fully loaded.")
        liver = self._predict_liver_stage1(smi)
        p_pred, p_margin = self.placenta_model.predict_with_confidence(self._placenta_frame(smi), confidence=0.90)
        k_pred, k_margin = self.kidney_model.predict_with_confidence(self._kidney_frame(smi), confidence=0.90)
        return {
            "liver": float(liver),
            "placenta": float(p_pred[0]),
            "kidney": float(k_pred[0]),
            "placenta_margin_90": float(p_margin),
            "kidney_margin_90": float(k_margin),
        }

    def _stage2_model(self, organ: str):
        obj = self.stage2_bundle[organ]
       
        return obj

    def _apply_stage2(self, organ: str, Xchem: np.ndarray, stage1: float) -> float:
        obj = self._stage2_model(organ)
        if hasattr(obj, "predict") and hasattr(obj, "model"):
            return float(obj.predict(Xchem, np.asarray([stage1]))[0])
        fusion = np.hstack([Xchem.astype(np.float32), np.asarray([[stage1]], dtype=np.float32)])
        return float(obj.predict(fusion)[0])

    def _from_cache(self, smi: str):
        if self.db_predictions is None:
            return None
        row = self.db_predictions[self.db_predictions["SMILES_Clean"] == smi]
        if row.empty:
            return None
        row = row.iloc[0]
        preferred = {
            "liver": "Liver_Integrated_Prediction",
            "placenta": "Placenta_Integrated_Prediction",
            "kidney": "Kidney_Integrated_Prediction",
        }
        if not all(c in row.index for c in preferred.values()):
            return None
        return {k: float(row[c]) for k, c in preferred.items()}

    def predict_organ_risks(self, smiles: str) -> dict:
        smi = self.canonicalize(smiles)
        cached = self._from_cache(smi)
        if cached is not None:
            payload = {
                **cached,
                "liver_risk": cached["liver"], "placenta_risk": cached["placenta"], "kidney_risk": cached["kidney"],
                "source": "stage2_prediction_cache",
            }
            try:
                s1 = self.get_stage1_predictions(smi)
                payload["stage1"] = {k: s1[k] for k in ("liver", "placenta", "kidney")}
                payload["stage1_intervals_90"] = {
                    "placenta": (s1["placenta"] - s1["placenta_margin_90"], s1["placenta"] + s1["placenta_margin_90"]),
                    "kidney": (s1["kidney"] - s1["kidney_margin_90"], s1["kidney"] + s1["kidney_margin_90"]),
                }
            except Exception:
                pass
            return payload

        if not self.stage2_ready:
            raise RuntimeError(
                "Stage-2 live inference is not ready. Place MULTIORGAN_STAGE2_REFINERS.pkl and "
                "PRETRAINED_MULTIORGAN_PREDICTIONS_RECONSTRUCTED.csv in the app folder, models/data folder, "
                "or a sibling multiorgan_results_latent folder. You can also set MULTIORGAN_RESULTS_DIR."
            )

        s1 = self.get_stage1_predictions(smi)
        liver_stage1, placenta_stage1, kidney_stage1 = s1["liver"], s1["placenta"], s1["kidney"]
        Xchem = shared_features_for_smiles([smi])

        final = {
            "liver": self._apply_stage2("liver", Xchem, liver_stage1),
            "placenta": self._apply_stage2("placenta", Xchem, placenta_stage1),
            "kidney": self._apply_stage2("kidney", Xchem, kidney_stage1),
        }
        return {
            **final,
            "liver_risk": final["liver"], "placenta_risk": final["placenta"], "kidney_risk": final["kidney"],
            "stage1": {"liver": liver_stage1, "placenta": placenta_stage1, "kidney": kidney_stage1},
            "stage1_intervals_90": {
                "placenta": (placenta_stage1 - s1["placenta_margin_90"], placenta_stage1 + s1["placenta_margin_90"]),
                "kidney": (kidney_stage1 - s1["kidney_margin_90"], kidney_stage1 + s1["kidney_margin_90"]),
            },
            "source": "stage2_live_inference",
        }
