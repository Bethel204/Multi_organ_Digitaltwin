A Hybrid AI Multi-Organ Digital Twin Framework for Developmental Pharmacokinetics and Drug Resistance Prediction

This repository implements a real time hybrid multi organ digital twin framework for drug risk prediction across the liver (CYP3A7), placenta, and kidney. The system integrates machine learning (QSAR/ADMET), transfer learning, ensemble models, mechanistic PKPD simulation, and a streamlit based deployment interface for real time prediction and visualization.



**Models**



**Liver** 

	Transfer learning from CYP3A4 to CYP3A7 

	Leakage resistant GroupKFold stacking

	Output: Metabolic risk prediction

**Kidney** 

	CatBoost ensemble regression model

	Feature engineering using RDKit descriptors and fingerprints

	Five fold cross validation plus 80/20 holdout

**Placenta**

	XGBoost based gated ensemble 

	Feature selection through recursive feature elimination plus ridge cross validation



**System**

Integrated multi organ prediction with PKPD simulation and uncertainty estimation.



**Run Application**

pip install -r requirements.txt

Streamlit run deployment/app.py



**Dependencies**

	Python 3.9+

	RDKit

	Scikit-learn

	XGBoost

	LightGBM

	CatBoost 

	PyTorch

	Streamlit



**Output**

	Organ specific risk prediction

	Integrated fetal risk score

	PKPD simulation curves

	Uncertainty (Monte Carlo) and explainability

	Multiorgan concentration trajectories overtime

	Feature importance visualization

**Notes**

	This system is intended for **research use only.**

	**Not validated** for clinical decision making.

	Designed for simulation and predictive modeling in pharmacology research.



