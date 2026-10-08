from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/result", methods=["POST", "GET"])
def result():

    # Get values from HTML form
    gender = request.form["gender"]
    age = int(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heart_disease = int(request.form["heart_disease"])

    ever_married = request.form["ever_married"]
    work_type = request.form["work_type"]
    Residence_type = request.form["Residence_type"]

    avg_glucose_level = float(request.form["avg_glucose_level"])
    bmi = float(request.form["bmi"])

    smoking_status = request.form["smoking_status"]


    # =========================================================
    # Convert categorical values into numbers
    # Same order as LabelEncoder used during model training
    # =========================================================

    gender_map = {
        "female": 0,
        "male": 1,
        "other": 2
    }

    ever_married_map = {
        "no": 0,
        "yes": 1
    }

    work_type_map = {
        "govt_job": 0,
        "never_worked": 1,
        "private": 2,
        "self-employed": 3,
        "children": 4
    }

    residence_map = {
        "rural": 0,
        "urban": 1
    }

    smoking_map = {
        "unknown": 0,
        "formerly smoked": 1,
        "never smoked": 2,
        "smokes": 3
    }


    # Convert text values to numbers
    gender = gender_map[gender.lower()]
    ever_married = ever_married_map[ever_married.lower()]
    work_type = work_type_map[work_type.lower()]
    Residence_type = residence_map[Residence_type.lower()]
    smoking_status = smoking_map[smoking_status.lower()]


    # =========================================================
    # Create input array
    # Order MUST match the training dataset
    # =========================================================

    x = np.array([
        gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        Residence_type,
        avg_glucose_level,
        bmi,
        smoking_status
    ]).reshape(1, -1)


    # =========================================================
    # Load StandardScaler
    # =========================================================

    scaler_path = os.path.join("models", "scaler.pkl")

    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Scale input
    x = scaler.transform(x)


    # =========================================================
    # Load Decision Tree model
    # =========================================================

    model_path = os.path.join("models", "dt.sav")

    dt = joblib.load(model_path)


    # =========================================================
    # Make prediction
    # =========================================================

    Y_pred = dt.predict(x)


    # =========================================================
    # Display result
    # =========================================================

    if Y_pred[0] == 0:
        return render_template("nostroke.html")
    else:
        return render_template("stroke.html")


if __name__ == "__main__":
    app.run(debug=True, port=7384)