import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import ssl
import certifi
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="ESOL Solubility Predictor",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 Aqueous Solubility Predictor (ESOL)")
st.markdown("Predict the aqueous solubility ($\log S$) of organic molecules using a **Random Forest Regressor** trained on the Delaney dataset and powered by **RDKit** molecular descriptors.")

# 1. Load data and train model with caching so it only runs once
@st.cache_resource
def load_and_train_model():
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, context=ssl_context) as response:
        df = pd.read_csv(response)
        
    target_col = [col for col in df.columns if 'solubility' in col.lower()][0]
    
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

    features_list = []
    valid_indices = []
    for idx, smiles in enumerate(df['smiles']):
        feats = extract_features(smiles)
        if feats is not None:
            features_list.append(feats)
            valid_indices.append(idx)

    X = pd.DataFrame(features_list)
    y = df.loc[valid_indices, target_col]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

with st.spinner("Initializing machine learning model and descriptors..."):
    model = load_and_train_model()

st.success("Model ready!")

# 2. User Input Section
st.subheader("Molecule Input")
example_smiles = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ethanol": "CCO",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
}

selected_example = st.selectbox("Choose a sample molecule or type your own SMILES below:", ["Custom"] + list(example_smiles.keys()))

if selected_example != "Custom":
    default_smiles = example_smiles[selected_example]
else:
    default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

smiles_input = st.text_input("Enter SMILES string:", value=default_smiles)

# 3. Prediction & Visualization Logic
if st.button("Predict Solubility", type="primary"):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("Invalid SMILES string. Please check your syntax and try again.")
    else:
        # Extract features for the input molecule
        input_features = {
            'MolLogP': Descriptors.MolLogP(mol),
            'MolWt': Descriptors.MolWt(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol)
        }
        
        input_df = pd.DataFrame([input_features])
        prediction = model.predict(input_df)[0]
        
        # Display results in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.metric(label="Predicted Log S (mol/L)", value=f"{prediction:.2f}")
            if prediction > -2:
                st.info("💧 **High Solubility**: Dissolves very readily in water.")
            elif prediction > -4:
                st.warning("⚖️ **Moderate Solubility**: Typical aqueous solubility.")
            else:
                st.error("🏜️ **Low Solubility**: Poorly soluble / hydrophobic.")
                
        with col2:
            st.write("**Molecular Structure:**")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, use_container_width=True)
            
        # Expandable breakdown of features
        with st.expander("View Calculated RDKit Descriptors"):
            st.json(input_features)
