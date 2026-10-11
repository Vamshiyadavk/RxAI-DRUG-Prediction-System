"""
train_model.py — Train & export Random Forest model for RxAI Drug Prediction
Trained with scikit-learn on drug_dataset.csv (1,200 clinical records, 49 classes).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))

def main():
    csv_path = os.path.join(BASE, "drug_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"drug_dataset.csv not found at: {csv_path}")

    print("Loading drug_dataset.csv...")
    df = pd.read_csv(csv_path)

    # Label Encoders
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_drug = LabelEncoder()

    df['Sex_enc'] = le_sex.fit_transform(df['Sex'])
    df['BP_enc'] = le_bp.fit_transform(df['BP'])
    df['Chol_enc'] = le_chol.fit_transform(df['Cholesterol'])
    df['Drug_enc'] = le_drug.fit_transform(df['Drug'])

    features = ['Age', 'Sex_enc', 'BP_enc', 'Chol_enc', 'Na_to_K']
    X = df[features].values
    y = df['Drug_enc'].values

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Optimized Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        max_features='sqrt',
        min_samples_split=2,
        random_state=42,
        n_jobs=1
    )

    print("Fitting Random Forest classifier...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))

    print(f"Running 5-fold cross validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    cv_acc = float(cv_scores.mean())

    # Build drug info lookup
    drug_lookup = {}
    for _, row in df.drop_duplicates('Drug').iterrows():
        drug_lookup[row['Drug']] = {
            'drug_class': row.get('DrugClass', ''),
            'indication': row.get('Indication', ''),
            'description': row.get('DrugDescription', ''),
        }

    bundle = {
        'model': model,
        'le_sex': le_sex,
        'le_bp': le_bp,
        'le_chol': le_chol,
        'le_drug': le_drug,
        'features': ['Age', 'Sex_enc', 'BP_enc', 'Chol_enc', 'Na_to_K'],
        'drug_lookup': drug_lookup,
        'test_accuracy': test_acc,
        'cv_accuracy': cv_acc,
        'n_classes': int(len(le_drug.classes_)),
        'n_trees': int(model.n_estimators),
    }

    out_path = os.path.join(BASE, "drug_model.pkl")
    joblib.dump(bundle, out_path)
    print(f"Saved: {out_path}")
    print(f"Test Accuracy: {test_acc*100:.2f}% | CV Accuracy: {cv_acc*100:.2f}%")
    return bundle

if __name__ == "__main__":
    main()
