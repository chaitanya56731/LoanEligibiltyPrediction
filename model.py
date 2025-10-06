import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("loan.csv")

# Features and target
features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

# Drop rows where Loan_Status is missing
df = df.dropna(subset=["Loan_Status"])

# Fill missing numeric values with median
for col in features:
    df[col] = df[col].fillna(df[col].median())

# Encode target variable
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = df[features]
y = df["Loan_Status"]

# Print stats
print("Class Distribution:")
print(y.value_counts())
print("\nFeature Stats:")
print(X.describe())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest with class balancing
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + feature list
with open("loan_model.pkl", "wb") as f:
    pickle.dump({"model": clf, "features": features}, f)

print("\n🎯 Model training completed and saved as loan_model.pkl")
