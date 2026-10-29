import pandas as pd
import numpy as np
import urllib.request
import ssl
import certifi
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Loading Delaney dataset...")
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

ssl_context = ssl.create_default_context(cafile=certifi.where())
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req, context=ssl_context) as response:
    df = pd.read_csv(response)

print(f"Dataset loaded successfully. Total molecules: {len(df)}")

# Automatically find the solubility target column
target_col = [col for col in df.columns if 'solubility' in col.lower()][0]
print(f"Target solubility column identified: '{target_col}'")

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolLogP': Descriptors.MolLogP(mol),
        'MolWt': Descriptors.MolWt(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol)
    }

print("Calculating molecular descriptors...")
features_list = []
valid_indices = []

for idx, smiles in enumerate(df['smiles']):
    feats = extract_features(smiles)
    if feats is not None:
        features_list.append(feats)
        valid_indices.append(idx)

X = pd.DataFrame(features_list)
y = df.loc[valid_indices, target_col]

print(f"Features extracted for {len(X)} valid molecules.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
