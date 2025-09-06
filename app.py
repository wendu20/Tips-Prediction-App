import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================
# App Title
# =========================================
st.title("💰 Tips Prediction App")
st.write("Predict restaurant tips using Machine Learning models (Linear Regression, Decision Tree, Random Forest).")

# =========================================
# Load Dataset
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv("tips.csv")
    return df

df = load_data()
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================================
# Data Preprocessing
# =========================================
y = df["tip"]
X = df.drop("tip", axis=1)

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================================
# Train or Load Models
# =========================================
model_files = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl"
}

models = {}
predictions = {}
results = []

for name, filename in model_files.items():
    if os.path.exists(filename):
        # Load saved model
        model = joblib.load(filename)
    else:
        # Train new model and save it
        if name == "Linear Regression":
            model = LinearRegression()
        elif name == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif name == "Random Forest":
            model = RandomForestRegressor(random_state=42, n_estimators=100)
        
        model.fit(X_train, y_train)
        joblib.dump(model, filename)

    # Save model in dictionary
    models[name] = model

    # Predict
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R² Score": r2_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

# =========================================
# Show Model Performance
# =========================================
st.subheader("📈 Model Comparison")
st.dataframe(results_df)

# Plot Actual vs Predicted
st.subheader("🔍 Prediction Visualization")
selected_model = st.selectbox("Choose a model to visualize:", list(models.keys()))

fig, ax = plt.subplots()
ax.scatter(y_test, predictions[selected_model], alpha=0.7)
ax.set_xlabel("Actual Tips")
ax.set_ylabel("Predicted Tips")
ax.set_title(f"{selected_model}: Actual vs Predicted")
ax.plot([0, max(y_test)], [0, max(y_test)], 'r--')
st.pyplot(fig)

# =========================================
# User Prediction
# =========================================
st.subheader("📝 Try Your Own Prediction")

total_bill = st.number_input("Total Bill", min_value=0.0, step=0.1)
gender = st.selectbox("Gender", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.slider("Party Size", 1, 10, 2)

# Create input dataframe
input_dict = {
    "total_bill": total_bill,
    "size": size,
    "gender_Male": 1 if gender == "Male" else 0,
    "smoker_Yes": 1 if smoker == "Yes" else 0,
    "day_Fri": 1 if day == "Fri" else 0,
    "day_Mon": 1 if day == "Mon" else 0,
    "day_Sat": 1 if day == "Sat" else 0,
    "day_Sun": 1 if day == "Sun" else 0,
    "day_Thur": 1 if day == "Thur" else 0,
    "day_Tues": 1 if day == "Tues" else 0,
    "day_Wed": 1 if day == "Wed" else 0,
    "time_Lunch": 1 if time == "Lunch" else 0,
}
input_df = pd.DataFrame([input_dict])

# Ensure all columns match training set
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

# Predict with chosen model
chosen_model = st.selectbox("Choose model for prediction:", list(models.keys()))
prediction = models[chosen_model].predict(input_df)[0]

st.success(f"💡 Predicted Tip Amount: **{prediction:.2f}**")
