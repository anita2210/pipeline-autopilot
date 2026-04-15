import joblib, json, pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load('models/trained/best_model.joblib')
scaler = joblib.load('models/trained/scaler.joblib')
df = pd.read_csv('data/processed/processed_dataset.csv')

with open('models/trained/feature_names.json') as f:
    features = json.load(f)

X = df[features]
y = df['failed']

X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
print('Train Accuracy:', accuracy_score(y, preds))