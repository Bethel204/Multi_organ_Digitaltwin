import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Multiorganmodel.pt")
DATA_PATH = os.path.join(BASE_DIR, "data", "PRETRAINED_MULTIORGAN_PREDICTIONS_COMPLETED.csv")


if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "Multiorganmodel.pt")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "PRETRAINED_MULTIORGAN_PREDICTIONS_COMPLETED.csv")



DEFAULT_DURATION_HRS = 72.0       
TIME_STEP_HRS = 0.05              
AI_UPDATE_INTERVAL_HRS = 1.0     



FETAL_TOXICITY_CEILING = 45.0     
HILL_COEFFICIENT_K = 0.25         
CONCENTRATION_50_MAX = 20.0       



POPULATION_FLOW_VARIANCE_COV = 0.15  
MONTE_CARLO_MIN_ITERATIONS = 40      




MATERNAL_BLOOD_VOLUME_L = 5.0     
BASE_HEPATIC_CLEARANCE_L_HR = 55.0 
BASE_RENAL_CLEARANCE_L_HR = 7.5    