from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STAGE2_RESULTS_DIR_ENV = os.getenv("MULTIORGAN_RESULTS_DIR", "").strip()


def _first_existing(candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate).resolve()
    return None


def _result_candidates(filename: str):
    candidates = []
    if STAGE2_RESULTS_DIR_ENV:
        candidates.append(Path(STAGE2_RESULTS_DIR_ENV) / filename)
    for folder in ("multiorgan_results_latent", "multiorgan_results_crossfit", "multiorgan_results_liver_tuned", "multiorgan_results"):
        candidates.extend([BASE_DIR / folder / filename, PARENT_DIR / folder / filename])
    candidates.extend([DATA_DIR / filename, MODELS_DIR / filename, BASE_DIR / filename, PARENT_DIR / filename])
    return candidates


STAGE2_REFINERS_PATH = _first_existing(_result_candidates("MULTIORGAN_STAGE2_REFINERS.pkl"))
STAGE2_PREDICTIONS_PATH = _first_existing(_result_candidates("PRETRAINED_MULTIORGAN_PREDICTIONS_RECONSTRUCTED.csv"))
STAGE2_METRICS_PATH = _first_existing(_result_candidates("MULTIORGAN_STAGE2_METRICS.json"))
STAGE2_MANIFEST_PATH = _first_existing(_result_candidates("MULTIORGAN_MODEL_MANIFEST.json"))
LIVER_CHECKPOINT_PATH = _first_existing([MODELS_DIR / "best_transfer_model.pt", BASE_DIR / "best_transfer_model.pt", PARENT_DIR / "best_transfer_model.pt"])
LIVER_SCALERS_PATH = _first_existing(_result_candidates("liver_reconstructed_scalers.pkl"))
PLACENTA_MODEL_PATH = _first_existing([MODELS_DIR / "pretrained_placenta_stacked_conformal_model.pkl", BASE_DIR / "pretrained_placenta_stacked_conformal_model.pkl", PARENT_DIR / "pretrained_placenta_stacked_conformal_model.pkl"])
KIDNEY_MODEL_PATH = _first_existing([MODELS_DIR / "pretrained_kidney_stacked_conformal_model.pkl", BASE_DIR / "pretrained_kidney_stacked_conformal_model.pkl", PARENT_DIR / "pretrained_kidney_stacked_conformal_model.pkl"])

DEFAULT_DURATION_HRS = 72.0
TIME_STEP_HRS = 0.10
MONTE_CARLO_MIN_ITERATIONS = 40
POPULATION_FLOW_VARIANCE_COV = 0.15
MATERNAL_BLOOD_VOLUME_L = 5.0

MATERNAL_RENAL_CLEARANCE_BASE_L_HR = 7.2
MATERNAL_PLACENTA_PS_BASE_L_HR = 2.0
PLACENTA_FETAL_PS_BASE_L_HR = 1.2
FETAL_RENAL_CLEARANCE_BASE_L_HR = 0.5

TISSUE_KP_MATERNAL_LIVER = 1.0
TISSUE_KP_PLACENTA = 1.0
TISSUE_KP_FETAL_LIVER = 1.0
TISSUE_KP_FETAL_KIDNEY = 1.0

TOXICITY_EMAX = 10.0
TOXICITY_EC50_MG_L = 25.0
TOXICITY_HILL = 1.0

CYP_DEGRADATION_RATE_HR = 0.1
CYP_INDUCTION_EMAX = 1.0
CYP_INDUCTION_EC50_MG_L = 25.0
CYP_INDUCTION_HILL = 1.0

RESISTANCE_PRODUCTION_MAX_HR = 0.4
RESISTANCE_EC50_MG_L = 25.0
RESISTANCE_HILL = 1.0
RESISTANCE_DECAY_HR = 0.1

AI_INHIBITION_IMAX = 1.0
AI_INHIBITION_HILL = 1.0
AI_PLACENTA_COUPLING_ENABLED = True
AI_PLACENTA_REFERENCE_SCORE = 5.0
AI_PLACENTA_COUPLING_STRENGTH = 0.35
AI_PLACENTA_MIN_MULTIPLIER = 0.25
AI_PLACENTA_MAX_MULTIPLIER = 4.0

REPORTED_STAGE2_R2 = {"liver": 0.9028, "placenta": 0.7865, "kidney": 0.7584}
