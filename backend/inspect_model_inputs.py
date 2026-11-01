import joblib

models = {
    "Himalayas": "data/processed/regional_models_30oct/himalayas/random_forest_model.joblib",
    "Northeast India": "data/processed/regional_models_30oct/northeast_india/random_forest_model.joblib",
    "Western Maharashtra pilot": "data/processed/regional_models_30oct/western_maharashtra/random_forest_model.joblib",
}

for region, path in models.items():
    model = joblib.load(path)
    print(f"\n{region}")
    print("Expected features:", list(model.feature_names_in_))
