import os
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from sklearn.manifold import MDS
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib  

optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
    "text.usetex": False,
    "axes.edgecolor": "#111111",
    "axes.linewidth": 0.8,
    "figure.titlesize": 12
})
sns.set_theme(style="ticks", context="paper")

DATA_FILE = "kidney_training_matrix.csv"
OUTPUT_DIR = "."

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
        """Separates, scales continuous descriptors, and filters low-variance bits specifically."""
        if is_training:
            self.desc_cols = [c for c in X.columns if not c.startswith('Bit_')]
            self.bit_cols = [c for c in X.columns if c.startswith('Bit_')]
            
            scaled_desc = self.scaler_desc.fit_transform(X[self.desc_cols])
            bits = X[self.bit_cols].values
            
            
            bit_variances = np.var(bits, axis=0)
            self.non_zero_bit_mask = bit_variances > 0.005
            filtered_bits = bits[:, self.non_zero_bit_mask]
            
            return np.hstack([scaled_desc, filtered_bits])
        else:
            scaled_desc = self.scaler_desc.transform(X[self.desc_cols])
            bits = X[self.bit_cols].values
            filtered_bits = bits[:, self.non_zero_bit_mask]
            return np.hstack([scaled_desc, filtered_bits])

    def fit_and_calibrate(self, X_train_full, y_train_full):
        """Executes a 5-Fold Cross-Conformal Pipeline with heterogeneous estimators."""
        X_tr = self._prepare_features(X_train_full, is_training=True)
        y_tr = y_train_full
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(X_tr), 3))
        
        self.fold_lgb_models = []
        self.fold_cat_models = []
        self.fold_rf_models = []
        
        for train_idx, val_idx in kf.split(X_tr):
            fold_lgb = lgb.LGBMRegressor(**self.lgb_opts)
            fold_cat = CatBoostRegressor(**self.cat_opts)
            fold_rf = RandomForestRegressor(**self.rf_opts)
            
            
            fold_lgb.fit(
                X_tr[train_idx], y_tr[train_idx],
                eval_set=[(X_tr[val_idx], y_tr[val_idx])],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            fold_cat.fit(
                X_tr[train_idx], y_tr[train_idx],
                eval_set=(X_tr[val_idx], y_tr[val_idx]),
                early_stopping_rounds=50,
                verbose=False
            )
            
            fold_rf.fit(X_tr[train_idx], y_tr[train_idx])
            
            oof_preds[val_idx, 0] = fold_lgb.predict(X_tr[val_idx])
            oof_preds[val_idx, 1] = fold_cat.predict(X_tr[val_idx])
            oof_preds[val_idx, 2] = fold_rf.predict(X_tr[val_idx])
            
            self.fold_lgb_models.append(fold_lgb)
            self.fold_cat_models.append(fold_cat)
            self.fold_rf_models.append(fold_rf)
            
        self.meta_learner.fit(oof_preds, y_tr)
        
        cal_preds = self.meta_learner.predict(oof_preds)
        self.calibration_errors = np.abs(y_tr - cal_preds)

    def predict_with_confidence(self, X_target_raw, confidence=0.90):
        X_target = self._prepare_features(X_target_raw, is_training=False)
        
        lgb_preds = np.mean([model.predict(X_target) for model in self.fold_lgb_models], axis=0)
        cat_preds = np.mean([model.predict(X_target) for model in self.fold_cat_models], axis=0)
        rf_preds = np.mean([model.predict(X_target) for model in self.fold_rf_models], axis=0)
        
        target_matrix = np.column_stack([lgb_preds, cat_preds, rf_preds])
        preds = self.meta_learner.predict(target_matrix)
        
        significance_level = 1.0 - confidence
        q_index = int(np.ceil((1.0 - significance_level) * (len(self.calibration_errors) + 1))) - 1
        q_index = min(max(q_index, 0), len(self.calibration_errors) - 1)
        error_bound = np.sort(self.calibration_errors)[q_index]
        return preds, error_bound

def generate_publication_figures(y_true, y_pred, error_bound, train_df, test_df, name):
    print(f"  🎨 Generating Publication Figures for {name}...")
    
    # FIGURE 1: Scatter Fit
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    sort_idx = np.argsort(y_true)
    y_true_s, y_pred_s = y_true[sort_idx], y_pred[sort_idx]
    
    ax.fill_between(y_true_s, y_pred_s - error_bound, y_pred_s + error_bound, 
                    color='#2b5c8f', alpha=0.12, label=f'90% Conformal Range ($\pm${error_bound:.2f})')
    ax.scatter(y_true, y_pred, color='#1f3a60', alpha=0.55, edgecolors='none', s=25, label='Test Scaffolds')
    
    limits = [min(min(y_true), min(y_pred)) - 0.5, max(max(y_true), max(y_pred)) + 0.5]
    ax.plot(limits, limits, color='#b22222', linestyle='--', linewidth=1.2, label='Ideal Fit ($y=x$)')
    
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f'{name}\nOptimized Cross-Conformal Ensemble $R^2$: {r2:.1f}', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Experimental $\mathrm{pIC}_{50}$', fontsize=9)
    ax.set_ylabel('Stacked Ensemble Predicted $\mathrm{pIC}_{50}$', fontsize=9)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', fontsize=8)
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig1_prediction_scatter_{name.lower()}.png"), dpi=300)
    plt.close()

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
    residuals = y_true - y_pred
    
    
    ax1.scatter(y_pred, residuals, color='#348abd', alpha=0.55, edgecolors='none', s=20)
    ax1.axhline(y=0, color='#111111', linestyle='-', linewidth=0.8)
    ax1.axhline(y=error_bound, color='#b22222', linestyle=':', linewidth=1.0, label=f'Upper Boundary (+{error_bound:.2f})')
    ax1.axhline(y=-error_bound, color='#b22222', linestyle=':', linewidth=1.0, label=f'Lower Boundary (-{error_bound:.2f})')
    ax1.set_xlabel('Predicted $\mathrm{pIC}_{50}$ Values', fontsize=9)
    ax1.set_ylabel('Residual ($\mathrm{Exp} - \mathrm{Pred}$)', fontsize=9)
    ax1.set_xlim([min(y_pred)-0.5, max(y_pred)+0.5])
    ax1.set_ylim([min(residuals)-0.5, max(residuals)+0.5])
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=7)
    sns.despine(ax=ax1, trim=True)
    
    
    sns.histplot(y=residuals, ax=ax2, color='#348abd', kde=True, bins=15, edgecolor='#ffffff', linewidth=0.5, alpha=0.6)
    ax2.axhline(y=0, color='#111111', linestyle='-', linewidth=0.8)
    ax2.axhline(y=error_bound, color='#b22222', linestyle=':', linewidth=1.0)
    ax2.axhline(y=-error_bound, color='#b22222', linestyle=':', linewidth=1.0)
    ax2.set_xlabel('Density Frequency', fontsize=9)
    ax2.set_ylabel('')
    ax2.set_yticklabels([])  
    ax2.set_ylim([min(residuals)-0.5, max(residuals)+0.5])
    sns.despine(ax=ax2, trim=True)
    
    fig.suptitle(f'{name} - Residual Error Profile & Distribution Histogram', fontsize=10, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig2_residuals_{name.lower()}.png"), dpi=300)
    plt.close()

    
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    train_sample = train_df.sample(min(300, len(train_df)), random_state=42)
    test_sample = test_df.sample(min(75, len(test_df)), random_state=42)
    combined_df = pd.concat([train_sample, test_sample])
    
    bit_cols = [c for c in combined_df.columns if c.startswith('Bit_')]
    fp_array = combined_df[bit_cols].values.astype(np.float32)
    
    intersection = fp_array @ fp_array.T
    row_sums = fp_array.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - intersection
    dist_matrix = 1.0 - (intersection / np.maximum(union, 1.0))
            
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=-1, normalized_stress='auto').fit_transform(dist_matrix)
    n_train = len(train_sample)
    
    ax.scatter(coords[:n_train, 0], coords[:n_train, 1], color='#cccccc', alpha=0.4, s=20, label='Training Pool Scaffolds')
    ax.scatter(coords[n_train:, 0], coords[n_train:, 1], color='#e24a33', alpha=0.85, edgecolors='w', linewidths=0.4, s=35, label='Novel Test Scaffolds')
    
    ax.set_title(f'{name}\nChemical Space Structural Coverage Map', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('MDS Coordinates Dimension 1', fontsize=9)
    ax.set_ylabel('MDS Coordinates Dimension 2', fontsize=9)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=8)
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig3_chemical_space_{name.lower()}.png"), dpi=300)
    plt.close()
    print("  ✓ Visual assets compiled successfully.")


def generate_descriptor_importance(engine, name):
    """FIGURE 4: Feature Importance Profile isolating properties and fingerprint bits."""
    print(f"  🎨 Generating Feature Importance Visualization for {name}...")
    
    desc_features = engine.desc_cols
    lgb_importances = np.mean([model.feature_importances_ for model in engine.fold_lgb_models], axis=0)
    lgb_importances = 100.0 * (lgb_importances / np.sum(lgb_importances))
    
    num_desc = len(desc_features)
    filtered_bit_names = np.array(engine.bit_cols)[engine.non_zero_bit_mask]
    
    feat_imp_df = pd.DataFrame({
        'Feature': list(desc_features) + list(filtered_bit_names),
        'Importance': lgb_importances,
        'Type': ['Physicochemical'] * num_desc + ['Structural Bit'] * len(filtered_bit_names)
    })
    
    top_features = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    colors = ['#1f3a60' if t == 'Physicochemical' else '#348abd' for t in top_features['Type']]
    
    sns.barplot(
        x='Importance', y='Feature', data=top_features, 
        palette=colors, ax=ax, hue='Type', legend=False, edgecolor='none', alpha=0.9
    )
    
   
    legend_elements = [
        Patch(facecolor='#1f3a60', label='Physicochemical Properties'),
        Patch(facecolor='#348abd', label='Structural Fingerprint Bits')
    ]
    
    ax.set_title(f'{name}\nBase-Ensemble Feature Attribution Profile', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Relative Importance Contribution (%)', fontsize=9)
    ax.set_ylabel('Top Contributing Predictive Coordinates', fontsize=9)
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=7)
    
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig4_feature_importance_{name.lower()}.png"), dpi=300)
    plt.close()
    print("  ✓ Figure 4 visual asset compiled successfully.")


if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        print(f"[!] Matrix file {DATA_FILE} not found.")
    else:
        print(f"\n🚀 PROCESSING DISCOVERY MATRIX: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        
        target_col = 'Target_pIC50'
        meta_cols = ['Target', 'SMILES', target_col]
        feature_cols = [col for col in df.columns if col not in meta_cols]
        
        X = df[feature_cols]
        y = df[target_col].values
        
        quantiles_main = np.percentile(y, [20, 40, 60, 80])
        y_binned_main = np.digitize(y, bins=quantiles_main)
        
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, df.index, test_size=0.20, random_state=42, stratify=y_binned_main
        )
        
       
        dataset_pkl = "kidney_processed_splits.pkl"
        print(f"  💾 Exporting stratified dataset splits to {dataset_pkl}...")
        joblib.dump({
            'X_train': X_train, 'X_test': X_test, 
            'y_train': y_train, 'y_test': y_test,
            'idx_train': idx_train, 'idx_test': idx_test
        }, dataset_pkl)
        print("  ✓ Data splits serialized successfully.")
        
        
        train_meta_df = df.loc[idx_train]
        test_meta_df = df.loc[idx_test]
        
        print("  ↳ Sifting hyperparameter coordinates via Joint Optuna Sweep (3-Fold CV)...")
        def objective(trial):
            lgb_p = {
                'n_estimators': trial.suggest_int('lgb_n_estimators', 800, 2000),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.005, 0.04, log=True),
                'num_leaves': trial.suggest_int('lgb_num_leaves', 31, 150),
                'max_depth': trial.suggest_int('lgb_max_depth', 6, 16),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 25),
                'subsample': trial.suggest_float('lgb_subsample', 0.5, 0.85),
                'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.3, 0.7),
                'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.1, 100.0, log=True),
                'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0.1, 100.0, log=True),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            
            cat_p = {
                'iterations': trial.suggest_int('cat_iterations', 800, 2000),
                'learning_rate': trial.suggest_float('cat_learning_rate', 0.005, 0.04, log=True),
                'depth': trial.suggest_int('cat_depth', 5, 10),
                'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 3.0, 30.0),
                'random_seed': 42, 'thread_count': -1, 'verbose': 0
            }
            
            rf_p = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 200, 600),
                'max_depth': trial.suggest_int('rf_max_depth', 10, 25),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 8),
                'max_features': trial.suggest_float('rf_max_features', 0.2, 0.6),
                'random_state': 42, 'n_jobs': -1
            }
            
            kf_opt = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            
            scaler_tmp = StandardScaler()
            desc_cols_tmp = [c for c in X_train.columns if not c.startswith('Bit_')]
            bit_cols_tmp = [c for c in X_train.columns if c.startswith('Bit_')]
            
            for train_idx, val_idx in kf_opt.split(X_train):
                X_tr_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                
                xtr_s = np.hstack([scaler_tmp.fit_transform(X_tr_fold[desc_cols_tmp]), X_tr_fold[bit_cols_tmp].values])
                xval_s = np.hstack([scaler_tmp.transform(X_val_fold[desc_cols_tmp]), X_val_fold[bit_cols_tmp].values])
                
                # Apply structural bit variance filter
                var_mask = np.var(xtr_s[:, len(desc_cols_tmp):], axis=0) > 0.005
                total_mask = np.concatenate([np.ones(len(desc_cols_tmp), dtype=bool), var_mask])
                
                xtr_s = xtr_s[:, total_mask]
                xval_s = xval_s[:, total_mask]
                
                m_lgb = lgb.LGBMRegressor(**lgb_p)
                m_lgb.fit(xtr_s, y_tr_fold, eval_set=[(xval_s, y_val_fold)], callbacks=[lgb.early_stopping(30, verbose=False)])
                
                m_cat = CatBoostRegressor(**cat_p)
                m_cat.fit(xtr_s, y_tr_fold, eval_set=(xval_s, y_val_fold), early_stopping_rounds=30, verbose=False)
                
                m_rf = RandomForestRegressor(**rf_p)
                m_rf.fit(xtr_s, y_tr_fold)
                
                pred_blend = (m_lgb.predict(xval_s) * 0.35) + (m_cat.predict(xval_s) * 0.35) + (m_rf.predict(xval_s) * 0.30)
                cv_scores.append(r2_score(y_val_fold, pred_blend))
                
            return np.mean(cv_scores)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=45)
        
        best_params = study.best_params
        best_lgb = {k.replace('lgb_', ''): v for k, v in best_params.items() if k.startswith('lgb_')}
        best_cat = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}
        best_rf = {k.replace('rf_', ''): v for k, v in best_params.items() if k.startswith('rf_')}
        
        best_lgb.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
        best_cat.update({'random_seed': 42, 'thread_count': -1, 'verbose': 0})
        best_rf.update({'random_state': 42, 'n_jobs': -1})
        
        print("  ↳ Executing Heterogeneous Fold-Ensembled Cross-Conformal Calibration...")
        engine = PublicationStackedConformalRegressor(best_lgb, best_cat, best_rf)
        engine.fit_and_calibrate(X_train, y_train)
        
        preds, margin = engine.predict_with_confidence(X_test, confidence=0.90)
        
        print(f"\n SCORECARD METRICS (Target: SLC22A6):")
        print(f"  ↳ Validation $R^2$             : {r2_score(y_test, preds):.1f}")
        print(f"  ↳ Mean Absolute Error (MAE)     : {mean_absolute_error(y_test, preds):.4f}")
        print(f"  ↳ Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
        print(f"  ↳ 90% Conformal Range Margin   : \pm{margin:.4f} pIC50 units")
        
      
        generate_publication_figures(y_test, preds, margin, train_meta_df, test_meta_df, "SLC22A6")
        
        
        generate_descriptor_importance(engine, "SLC22A6")
        
       
        pipeline_filename = "pretrained_kidney_stacked_conformal_model.pkl"
        print(f"  Exporting pretrained pipeline artifact to {pipeline_filename}...")
        joblib.dump(engine, pipeline_filename)
        print("   Pipeline serialized successfully.")
        print("\n Packaging finalized Kidney / SLC22A6 evaluation matrix...")
        
        X_train_clean = engine._prepare_features(X_train, is_training=False)
        X_test_clean = engine._prepare_features(X_test, is_training=False)
        
        lgb_tr = np.mean([m.predict(X_train_clean) for m in engine.fold_lgb_models], axis=0)
        cat_tr = np.mean([m.predict(X_train_clean) for m in engine.fold_cat_models], axis=0)
        rf_tr = np.mean([m.predict(X_train_clean) for m in engine.fold_rf_models], axis=0)
        train_preds = engine.meta_learner.predict(np.column_stack([lgb_tr, cat_tr, rf_tr]))
        
        test_preds = preds
        
        df_train_track = df.loc[idx_train].copy()
        df_train_track['predicted_pIC50'] = train_preds
        df_train_track['dataset_assignment'] = 'Cross_Validation_Training'
        
        df_test_track = df.loc[idx_test].copy()
        df_test_track['predicted_pIC50'] = test_preds
        df_test_track['dataset_assignment'] = 'Validation_Holdout'
        
        df_export = pd.concat([df_train_track, df_test_track], axis=0, ignore_index=True)
        
        df_export['absolute_error'] = np.abs(df_export[target_col] - df_export['predicted_pIC50'])
        df_export['conformal_90_lower_bound'] = df_export['predicted_pIC50'] - margin
        df_export['conformal_90_upper_bound'] = df_export['predicted_pIC50'] + margin
        
        ordered_cols = [
            'Target', 'SMILES', target_col, 'predicted_pIC50', 
            'absolute_error', 'conformal_90_lower_bound', 
            'conformal_90_upper_bound', 'dataset_assignment'
        ]
        
        additional_metadata = [c for c in ['Compound_ID', 'MW', 'LogP', 'HBD', 'HBA'] if c in df_export.columns]
        final_column_ordering = additional_metadata + ordered_cols
        df_export = df_export[final_column_ordering]
        
        kidney_output_file = "FINAL_TRAINED_KIDNEY_SLC22A6_PREDICTIONS.csv"
        try:
            df_export.to_csv(os.path.join(OUTPUT_DIR, kidney_output_file), index=False)
            print(f" ✓ Dataset successfully saved as: '{kidney_output_file}'")
        except PermissionError:
            fallback_kidney = "FINAL_TRAINED_KIDNEY_SLC22A6_PREDICTIONS_NEW.csv"
            df_export.to_csv(os.path.join(OUTPUT_DIR, fallback_kidney), index=False)
            print(f" [!] Target file locked. Saved fallback asset as '{fallback_kidney}'")
            
        print(f"  Total unique chemical space exported: {len(df_export)} records.")
        
        print("\nPIPELINE COMPLETION VERIFIED.")