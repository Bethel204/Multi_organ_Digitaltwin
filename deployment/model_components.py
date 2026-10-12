from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PlacentaStackedConformalRegressor:
    def __init__(self, lgb_opts=None, cat_opts=None, rf_opts=None):
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
            self.non_zero_bit_mask = np.var(bits, axis=0) > 0.005
        return bits[:, self.non_zero_bit_mask]

    def predict_with_confidence(self, X_target_raw, confidence=0.90):
        X_target = self._prepare_features(X_target_raw, is_training=False)
        lgb_preds = np.mean([m.predict(X_target) for m in self.fold_lgb_models], axis=0)
        cat_preds = np.mean([m.predict(X_target) for m in self.fold_cat_models], axis=0)
        rf_preds = np.mean([m.predict(X_target) for m in self.fold_rf_models], axis=0)
        preds = self.meta_learner.predict(np.column_stack([lgb_preds, cat_preds, rf_preds]))
        errors = np.asarray(self.calibration_errors)
        q = min(max(int(np.ceil(confidence * (len(errors) + 1))) - 1, 0), len(errors) - 1)
        return preds, float(np.sort(errors)[q])


class PublicationStackedConformalRegressor:
    def __init__(self, lgb_opts=None, cat_opts=None, rf_opts=None):
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
            self.non_zero_bit_mask = np.var(bits, axis=0) > 0.005
        else:
            scaled_desc = self.scaler_desc.transform(X[self.desc_cols])
            bits = X[self.bit_cols].values
        return np.hstack([scaled_desc, bits[:, self.non_zero_bit_mask]])

    def predict_with_confidence(self, X_target_raw, confidence=0.90):
        X_target = self._prepare_features(X_target_raw, is_training=False)
        lgb_preds = np.mean([m.predict(X_target) for m in self.fold_lgb_models], axis=0)
        cat_preds = np.mean([m.predict(X_target) for m in self.fold_cat_models], axis=0)
        rf_preds = np.mean([m.predict(X_target) for m in self.fold_rf_models], axis=0)
        preds = self.meta_learner.predict(np.column_stack([lgb_preds, cat_preds, rf_preds]))
        errors = np.asarray(self.calibration_errors)
        q = min(max(int(np.ceil(confidence * (len(errors) + 1))) - 1, 0), len(errors) - 1)
        return preds, float(np.sort(errors)[q])


class PublicationTransferNetwork(nn.Module):
    def __init__(self, input_dim=2215):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.Mish(), nn.Dropout(0.05)
        )
        self.res_block = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.Mish(), nn.Dropout(0.05),
            nn.Linear(512, 512), nn.BatchNorm1d(512),
        )
        self.activation = nn.Mish()
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(0.05)
        )
        self.regressor_head = nn.Sequential(
            nn.Linear(256, 128), nn.Mish(), nn.Dropout(0.05),
            nn.Linear(128, 64), nn.Mish(), nn.Linear(64, 1),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation(out + self.res_block(out))
        return self.regressor_head(self.feature_extractor(out))


class OrganMetaRefiner:
    """Compatibility class for loading the Stage-2 wrapper saved by Multi_Organ.py."""
    def __init__(self, organ_name="Organ", n_splits=5):
        self.organ_name = organ_name
        self.n_splits = n_splits
        self.model = None

    @staticmethod
    def _fusion_matrix(X_chem, base_predictions):
        base = np.asarray(base_predictions, dtype=np.float32).reshape(-1, 1)
        return np.hstack([np.asarray(X_chem, dtype=np.float32), base])

    def predict(self, X_chem, base_predictions):
        return self.model.predict(self._fusion_matrix(X_chem, base_predictions))
