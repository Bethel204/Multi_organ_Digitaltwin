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


LABELS_FILE = "placenta_publication_ready.csv"
FEATURES_FILE = "placenta_training_matrix.csv"
OUTPUT_DIR = "."

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
        else:
            return bits[:, self.non_zero_bit_mask]

    def fit_and_calibrate(self, X_train_full, y_train_full):
        
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

def generate_publication_figures(y_true, y_pred, error_bound, train_df, test_df, name="Placenta Pretrained"):
    
    print(f"  Generating Figures for {name} Dataset...")
    
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    sort_idx = np.argsort(y_true)
    y_true_s, y_pred_s = y_true[sort_idx], y_pred[sort_idx]
    
    ax.fill_between(y_true_s, y_pred_s - error_bound, y_pred_s + error_bound, 
                    color='#8cc63f', alpha=0.15, label=f'90% Conformal Range ($\pm${error_bound:.2f})')
    ax.scatter(y_true, y_pred, color='#2e6f40', alpha=0.6, edgecolors='none', s=25, label='Placenta Test Scaffolds')
    
    limits = [min(min(y_true), min(y_pred)) - 0.5, max(max(y_true), max(y_pred)) + 0.5]
    ax.plot(limits, limits, color='#b22222', linestyle='--', linewidth=1.2, label='Ideal Fit ($y=x$)')
    
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f'{name} Dataset Model\nOptimized Cross-Conformal Ensemble $R^2$: {r2:.2f}', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Experimental Placenta $\mathrm{pIC}_{50}$', fontsize=9)
    ax.set_ylabel('Stacked Ensemble Predicted Placenta $\mathrm{pIC}_{50}$', fontsize=9)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', fontsize=8)
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig1_prediction_scatter_{name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
    residuals = y_true - y_pred
    
    ax1.scatter(y_pred, residuals, color='#4ca64c', alpha=0.6, edgecolors='none', s=20)
    ax1.axhline(y=0, color='#111111', linestyle='-', linewidth=0.8)
    ax1.axhline(y=error_bound, color='#b22222', linestyle=':', linewidth=1.0, label=f'Upper Boundary (+{error_bound:.2f})')
    ax1.axhline(y=-error_bound, color='#b22222', linestyle=':', linewidth=1.0, label=f'Lower Boundary (-{error_bound:.2f})')
    ax1.set_xlabel('Predicted Placenta $\mathrm{pIC}_{50}$ Values', fontsize=9)
    ax1.set_ylabel('Residual ($\mathrm{Exp} - \mathrm{Pred}$)', fontsize=9)
    ax1.set_xlim([min(y_pred)-0.5, max(y_pred)+0.5])
    ax1.set_ylim([min(residuals)-0.5, max(residuals)+0.5])
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=7)
    sns.despine(ax=ax1, trim=True)
    
    sns.histplot(y=residuals, ax=ax2, color='#4ca64c', kde=True, bins=15, edgecolor='#ffffff', linewidth=0.5, alpha=0.6)
    ax2.axhline(y=0, color='#111111', linestyle='-', linewidth=0.8)
    ax2.axhline(y=error_bound, color='#b22222', linestyle=':', linewidth=1.0)
    ax2.axhline(y=-error_bound, color='#b22222', linestyle=':', linewidth=1.0)
    ax2.set_xlabel('Error Density Frequency', fontsize=9)
    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    ax2.set_ylim([min(residuals)-0.5, max(residuals)+0.5])
    sns.despine(ax=ax2, trim=True)
    
    fig.suptitle(f'{name} Dataset - Error Distribution Profile & Residual Histogram', fontsize=10, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig2_residuals_{name.lower().replace(' ', '_')}.png"), dpi=300)
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
    
    ax.scatter(coords[:n_train, 0], coords[:n_train, 1], color='#cccccc', alpha=0.4, s=20, label='Placenta Training Pool')
    ax.scatter(coords[n_train:, 0], coords[n_train:, 1], color='#e24a33', alpha=0.85, edgecolors='w', linewidths=0.4, s=35, label='Novel Placenta Test Scaffolds')
    
    ax.set_title(f'{name} Dataset\nChemical Space Structural Coverage Map', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('MDS Coordinates Dimension 1', fontsize=9)
    ax.set_ylabel('MDS Coordinates Dimension 2', fontsize=9)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=8)
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig3_chemical_space_{name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()

def generate_descriptor_importance(engine, name):
    
    print(f"  Generating Feature Importance Visualization for {name}...")
    
    filtered_bit_names = np.array(engine.bit_cols)[engine.non_zero_bit_mask]
    lgb_importances = np.mean([model.feature_importances_ for model in engine.fold_lgb_models], axis=0)
    lgb_importances = 100.0 * (lgb_importances / np.sum(lgb_importances))
    
    feat_imp_df = pd.DataFrame({
        'Structural Coordinate': filtered_bit_names,
        'Importance': lgb_importances
    })
    
    top_features = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    sns.barplot(
        x='Importance', y='Structural Coordinate', data=top_features, 
        color='#2e6f40', ax=ax, edgecolor='none', alpha=0.85
    )
    
    ax.set_title(f'{name} Dataset\nFingerprint Substructure Attribution Profile', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Relative Structural Contribution (%)', fontsize=9)
    ax.set_ylabel('Top Contributing Fingerprint Substructure Coordinates', fontsize=9)
    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig4_feature_importance_{name.lower().replace(' ', '_')}.png"), dpi=300)
    plt.close()
    print("  ✓ Visual assets compiled successfully.")

if __name__ == '__main__':
    if not (os.path.exists(LABELS_FILE) and os.path.exists(FEATURES_FILE)):
        print("[!] Dataset source tracks missing. Ensure matrix and label files are placed in execution root.")
    else:
        print(f"\n JOINING AND COMPILING PLACENTA DATA ASSETS...")
        labels_df = pd.read_csv(LABELS_FILE)
        features_df = pd.read_csv(FEATURES_FILE)
        
        
        merged_df = pd.merge(labels_df, features_df, on='SMILES', how='inner')
        print(f"  Successfully joined matrices. Total intersection cohort: {len(merged_df)} compounds.")
        
       
        matrix_csv_name = "placenta_pretrained_datamatrix.csv"
        matrix_pkl_name = "placenta_pretrained_datamatrix.pkl"
        
        print(f"   Exporting full merged data matrix to {matrix_csv_name}...")
        merged_df.to_csv(matrix_csv_name, index=False)
        
        print(f"   Exporting full merged data matrix to {matrix_pkl_name}...")
        joblib.dump(merged_df, matrix_pkl_name)
        print("  Pretrained data matrices saved successfully.")
       
        
        
        target_col = 'Target_pIC50_x' if 'Target_pIC50_x' in merged_df.columns else 'Target_pIC50'
        feature_cols = [col for col in merged_df.columns if col.startswith('Bit_')]
        
        X = merged_df[feature_cols]
        y = merged_df[target_col].values
        
        quantiles_main = np.percentile(y, [20, 40, 60, 80])
        y_binned_main = np.digitize(y, bins=quantiles_main)
        
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, merged_df.index, test_size=0.20, random_state=42, stratify=y_binned_main
        )
        
        
        dataset_pkl = "placenta_processed_splits.pkl"
        print(f"  Exporting stratified training splits to {dataset_pkl}...")
        joblib.dump({
            'X_train': X_train, 'X_test': X_test, 
            'y_train': y_train, 'y_test': y_test,
            'idx_train': idx_train, 'idx_test': idx_test
        }, dataset_pkl)
        print("  Data splits serialized successfully.")
        
        train_meta_df = merged_df.loc[idx_train]
        test_meta_df = merged_df.loc[idx_test]
        
        print("  Sifting hyperparameter coordinates via Joint Optuna Sweep (3-Fold CV)...")
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
            
            for train_idx, val_idx in kf_opt.split(X_train):
                X_tr_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                
                var_mask = np.var(X_tr_fold.values, axis=0) > 0.005
                xtr_s = X_tr_fold.values[:, var_mask]
                xval_s = X_val_fold.values[:, var_mask]
                
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
        engine = PlacentaStackedConformalRegressor(best_lgb, best_cat, best_rf)
        engine.fit_and_calibrate(X_train, y_train)
        
        preds, margin = engine.predict_with_confidence(X_test, confidence=0.90)
        
        print(f"\n JOURNAL SCORECARD METRICS (Target: PLACENTA ACCUMULATION):")
        print(f"   Validation $R^2$             : {r2_score(y_test, preds):.2f}")
        print(f"   Mean Absolute Error (MAE)     : {mean_absolute_error(y_test, preds):.4f}")
        print(f"   Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
        print(f"   90% Conformal Range Margin   : \pm{margin:.4f} pIC50 units")
        
        
        generate_publication_figures(y_test, preds, margin, train_meta_df, test_meta_df, "Placenta Pretrained")
        generate_descriptor_importance(engine, "Placenta Pretrained")
        
        
        pipeline_filename = "pretrained_placenta_stacked_conformal_model.pkl"
        print(f"   Exporting pretrained pipeline artifact to {pipeline_filename}...")
        joblib.dump(engine, pipeline_filename)
        print("   Pipeline serialized successfully.")
        
        print("\n PIPELINE COMPLETION VERIFIED.")