"""
model.py

Training the logistic regression model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def train_model(df):
    X = df.drop(columns=["completed"])
    y = df["completed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Applying SMOTE to address imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    """
    hours_per_week has high values while rating columns are 1-5
    therefore we transform every feature to have same scale
    """
    scaler = StandardScaler()
    #Learns the scale from training data
    X_train_scaled = scaler.fit_transform(X_train)
    #Applies same scale to test data
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:\n{report}")

    return model, scaler