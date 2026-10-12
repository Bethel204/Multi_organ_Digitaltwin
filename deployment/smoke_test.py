from ai_prediction import AIPredictionPipeline

p = AIPredictionPipeline()
print("Deployment status:")
print(p.status())
print("\nIf stage2_ready is False but prediction_cache_loaded is True, cached Stage-2 molecules can still be simulated.")
print("For novel molecules, both Stage-2 refiners and all Stage-1 models must be available.")
