import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load saved artifacts
# -----------------------------
model = load_model("nn_injury_model.keras", compile=False)
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
league_encoder = joblib.load("league_label_encoder.pkl")
kmeans = joblib.load("workload_kmeans.pkl")

st.title("⚽ Neural Network Injury Duration Predictor")
st.write("Enter the player's season statistics to estimate expected days missed.")

# -----------------------------
# User Inputs
# -----------------------------

# Main inputs
minutes_played = st.number_input("Minutes played this season", min_value=0)
games_played = st.number_input("Games played this season", min_value=0)
yellow = st.number_input("Yellow Cards", min_value=0)
second_yellow = st.number_input("Second Yellow Cards", min_value=0)
red = st.number_input("Red Cards", min_value=0)

age = st.number_input("Age", min_value=14, max_value=50)
height = st.number_input("Height (cm)", min_value=140, max_value=220)

# League dropdown
league_name = st.selectbox("League", league_encoder.classes_.tolist())
league_code = league_encoder.transform([league_name])[0]

# Previous season stats
prev_days = st.number_input("Previous season days missed", min_value=0)
prev_minutes = st.number_input("Previous season minutes played", min_value=0)
prev_games = st.number_input("Previous season games played", min_value=0)

# Preferred foot dropdown
foot_choice = st.selectbox("Preferred Foot", ["right", "left", "both"])

# -----------------------------
# Internal Feature Engineering
# -----------------------------

# Card score formula from training
card_score = yellow + second_yellow*2 + red*5

# Fatigue & workload
fatigue = minutes_played / max(games_played, 1)
workload = (minutes_played + prev_minutes) / max(games_played + prev_games, 1)

# Foot one-hot
foot_right = 1 if foot_choice == "right" else 0
foot_left = 1 if foot_choice == "left" else 0
foot_both = 1 if foot_choice == "both" else 0

# Workload cluster (KMeans)
cluster = kmeans.predict([[workload]])[0]
cluster_0 = 1 if cluster == 0 else 0
cluster_1 = 1 if cluster == 1 else 0

# Squared features
age_sq = age**2
fatigue_sq = fatigue**2
workload_sq = workload**2

# -----------------------------
# Build model input vector
# -----------------------------

data = {
    "minutes_played": minutes_played,
    "games_played": games_played,
    "card_score": card_score,
    "fatigue": fatigue,
    "workload": workload,
    "age": age,
    "height": height,
    "league_code": league_code,
    "prev_days_missed": prev_days,
    "prev_minutes_played": prev_minutes,
    "prev_fatigue": prev_minutes / max(prev_games, 1),
    "prev_games_played": prev_games,
    "age_squared": age_sq,
    "fatigue_squared": fatigue_sq,
    "workload_squared": workload_sq,
    "foot_left": foot_left,
    "foot_right": foot_right,
    "foot_both": foot_both,
    "cluster_0": cluster_0,
    "cluster_1": cluster_1,
}

# Convert to DataFrame with correct order
df = pd.DataFrame([data], columns=feature_cols)

# -----------------------------
# Predict
# -----------------------------
if st.button("🔮 Predict Injury Duration"):
    imputed = imputer.transform(df)
    scaled = scaler.transform(imputed)
    log_pred = model.predict(scaled)[0][0]
    final_pred = np.expm1(log_pred)

    st.success(f"⏳ Predicted Injury Duration: **{final_pred:.1f} days**")
