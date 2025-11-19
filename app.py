import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the NN model and preprocessors
model = load_model("nn_injury_model.keras", compile=False)
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
league_encoder = joblib.load("league_label_encoder.pkl")

st.title("⚽ Football Injury Days Predictor")
st.write("Enter the player's statistics below to estimate how many days they may be injured.")

# ===== USER INPUTS =====
minutes_played = st.number_input("Minutes played this season", 0, 6000)
games_played = st.number_input("Games played this season", 0, 80)
yellow_cards = st.number_input("Yellow Cards", 0, 30)
second_yellow_cards = st.number_input("Second Yellow Cards", 0, 10)
red_cards = st.number_input("Direct Red Cards", 0, 10)

age = st.number_input("Age", 15, 45)
height = st.number_input("Height (cm)", 140, 220)

previous_days_missed = st.number_input("Previous days missed due to injuries", 0, 1000)
previous_minutes = st.number_input("Previous season minutes played", 0, 6000)
previous_games = st.number_input("Previous season games played", 0, 80)

foot_choice = st.selectbox("Preferred Foot", ["left", "right", "both"])
foot_left = 1 if foot_choice == "left" else 0
foot_right = 1 if foot_choice == "right" else 0
foot_both = 1 if foot_choice == "both" else 0

league_list = league_encoder.classes_
league_choice = st.selectbox("League", league_list)
league_encoded = league_encoder.transform([league_choice])[0]

# ===== BUILD INPUT VECTOR =====
input_dict = {
    "minutes_played": minutes_played,
    "games_played": games_played,
    "yellow_cards": yellow_cards,
    "second_yellow_cards": second_yellow_cards,
    "direct_red_cards": red_cards,
    "age": age,
    "height": height,
    "previous_days_missed": previous_days_missed,
    "previous_minutes": previous_minutes,
    "previous_games": previous_games,
    "league_encoded": league_encoded,
    "foot_left": foot_left,
    "foot_right": foot_right,
    "foot_both": foot_both
}

df_input = pd.DataFrame([input_dict])
df_input = df_input.reindex(columns=feature_cols, fill_value=0)

# Preprocess
df_input = imputer.transform(df_input)
df_input = scaler.transform(df_input)

# ===== PREDICT =====
if st.button("Predict Injury Days"):
    pred = model.predict(df_input)[0][0]
    st.subheader(f"Predicted Injury Duration: **{pred:.1f} days**")
