import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

SEUIL = 0.35
DATA = "pima-indians-diabetes.data.csv"
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURES = COLUMNS[:-1]
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA, names=COLUMNS)
    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)

    X = df[FEATURES]
    y = df["Outcome"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(class_weight="balanced", max_depth=5,
                                       n_estimators=200, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


model = train_model()

st.set_page_config(page_title="Risque de diabète", page_icon="🩺")
st.title("🩺 Prédiction du risque de diabète")
st.write("Renseigne les mesures de la patiente pour estimer son risque de diabète.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Pression artérielle (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Épaisseur de la peau (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insuline (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("IMC", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
    dpf = st.number_input("Antécédents familiaux (score)", min_value=0.0, max_value=3.0,
                          value=0.5, step=0.01)
    age = st.number_input("Âge", min_value=18, max_value=100, value=35)

if st.button("Prédire"):
    row = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }])

    # Même traitement que dans le notebook : 0 = valeur manquante
    row[ZERO_COLS] = row[ZERO_COLS].replace(0, np.nan)

    proba = model.predict_proba(row[FEATURES])[0, 1]

    st.divider()
    if proba >= SEUIL:
        st.error(f"Risque élevé : probabilité de {proba:.1%} (seuil : {SEUIL}). "
                 "Un examen complémentaire est recommandé.")
    else:
        st.success(f"Risque plus faible : probabilité de {proba:.1%} (seuil : {SEUIL}).")

    st.caption("Estimation statistique issue d'un modèle de Machine Learning. "
               "Elle ne remplace pas un avis médical.")