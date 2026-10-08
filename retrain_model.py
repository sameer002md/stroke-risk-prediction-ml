import pandas as pd
import pickle
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv(
    r"dataset/healthcare-dataset-stroke-data.csv"
)

print("Dataset loaded successfully.")
print("Original shape:", data.shape)


# --------------------------------------------------
# 2. Handle missing BMI values
# --------------------------------------------------

data["bmi"].fillna(data["bmi"].mean(), inplace=True)


# --------------------------------------------------
# 3. Remove ID column
# --------------------------------------------------

data.drop("id", axis=1, inplace=True)


# --------------------------------------------------
# 4. Encode categorical columns
# --------------------------------------------------

enc = LabelEncoder()

gender = enc.fit_transform(data["gender"])

smoking_status = enc.fit_transform(data["smoking_status"])

work_type = enc.fit_transform(data["work_type"])

Residence_type = enc.fit_transform(data["Residence_type"])

ever_married = enc.fit_transform(data["ever_married"])


data["work_type"] = work_type
data["ever_married"] = ever_married
data["Residence_type"] = Residence_type
data["smoking_status"] = smoking_status
data["gender"] = gender


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X = data.drop("stroke", axis=1)

Y = data["stroke"]


print("Features:", list(X.columns))
print("Number of features:", X.shape[1])


# --------------------------------------------------
# 6. Train-test split
# --------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=10
)


# --------------------------------------------------
# 7. Standardize features
# --------------------------------------------------

std = StandardScaler()

X_train_std = std.fit_transform(X_train)

X_test_std = std.transform(X_test)


# --------------------------------------------------
# 8. Save new scaler
# --------------------------------------------------

scaler_path = os.path.join("models", "scaler.pkl")

with open(scaler_path, "wb") as scaler_file:
    pickle.dump(std, scaler_file)


print("New scaler saved:", scaler_path)


# --------------------------------------------------
# 9. Train Decision Tree
# --------------------------------------------------

dt = DecisionTreeClassifier()

dt.fit(X_train_std, Y_train)


# --------------------------------------------------
# 10. Test Decision Tree
# --------------------------------------------------

Y_pred = dt.predict(X_test_std)

accuracy = accuracy_score(Y_test, Y_pred)

print("Decision Tree Accuracy:", accuracy)


# --------------------------------------------------
# 11. Save new Decision Tree model
# --------------------------------------------------

model_path = os.path.join("models", "dt.sav")

joblib.dump(dt, model_path)


print("New Decision Tree model saved:", model_path)

print()
print("====================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("====================================")