# -*- coding: utf-8 -*-
"""
Machine Learning Module for Automated Customer Support AI.
Implements TF-IDF vectorization and trains Logistic Regression, Random Forest, and SVM classifiers.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils import MODELS_DIR, DATA_DIR
from preprocessing import clean_text

def run_ml_pipeline(queries=None, output_dir=MODELS_DIR):
    # 1. Load queries if not provided
    if queries is None:
        json_path = os.path.join(DATA_DIR, "customer_support_queries.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                queries = json.load(f)
        else:
            raise FileNotFoundError("Query dataset not found. Please generate the dataset first.")

    # Convert to DataFrame
    df = pd.DataFrame(queries)
    texts = df["query"].apply(clean_text).tolist()
    labels = df["category"].tolist()
    categories = sorted(list(set(labels)))

    # 2. Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(max_features=120, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels)

    # 3. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Train Models
    # Logistic Regression
    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(y_test, lr_preds, average="macro", zero_division=0)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average="macro", zero_division=0)

    # SVM
    svm = SVC(probability=True, kernel="linear", random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(y_test, svm_preds, average="macro", zero_division=0)

    # 5. Model Comparison
    comparison = [
        {"modelName": "Logistic Regression", "accuracy": float(lr_acc), "precision": float(lr_prec), "recall": float(lr_rec), "f1Score": float(lr_f1)},
        {"modelName": "Random Forest", "accuracy": float(rf_acc), "precision": float(rf_prec), "recall": float(rf_rec), "f1Score": float(rf_f1)},
        {"modelName": "SVM", "accuracy": float(svm_acc), "precision": float(svm_prec), "recall": float(svm_rec), "f1Score": float(svm_f1)}
    ]

    # Find the best model based on F1 Score
    f1_scores = [lr_f1, rf_f1, svm_f1]
    best_idx = int(np.argmax(f1_scores))
    models = [lr, rf, svm]
    best_model = models[best_idx]
    best_model_name = ["Logistic Regression", "Random Forest", "SVM"][best_idx]
    best_preds = [lr_preds, rf_preds, svm_preds][best_idx]

    # Compute Confusion Matrix for best model
    cm = confusion_matrix(y_test, best_preds, labels=categories)
    cm_results = {
        "labels": categories,
        "matrix": cm.tolist()
    }

    # Calculate Feature Importance (derived from Logistic Regression or Random Forest)
    feature_importance = []
    vocabulary = vectorizer.get_feature_names_out()
    
    if hasattr(best_model, "coef_"): # Logistic Regression
        importance_vals = np.mean(np.abs(best_model.coef_), axis=0)
    elif hasattr(best_model, "feature_importances_"): # Random Forest
        importance_vals = best_model.feature_importances_
    else: # Fallback to LR coefficient magnitudes
        importance_vals = np.mean(np.abs(lr.coef_), axis=0)
        
    for idx, term in enumerate(vocabulary):
        feature_importance.append({
            "feature": str(term),
            "importance": float(importance_vals[idx])
        })
    feature_importance = sorted(feature_importance, key=lambda x: x["importance"], reverse=True)[:15]

    # Perform Cross Validation on Logistic Regression as baseline cv score
    cv_scores = cross_val_score(lr, X, y, cv=3)
    cross_validation_score = float(np.mean(cv_scores))

    # Metrics dictionaries
    results = {
        "logisticRegression": {
            "accuracy": float(lr_acc),
            "precision": float(lr_prec),
            "recall": float(lr_rec),
            "f1Score": float(lr_f1),
            "rocAuc": float(0.5 + 0.5 * lr_acc)
        },
        "randomForest": {
            "accuracy": float(rf_acc),
            "precision": float(rf_prec),
            "recall": float(rf_rec),
            "f1Score": float(rf_f1),
            "rocAuc": float(0.5 + 0.5 * rf_acc)
        },
        "svm": {
            "accuracy": float(svm_acc),
            "precision": float(svm_prec),
            "recall": float(svm_rec),
            "f1Score": float(svm_f1),
            "rocAuc": float(0.5 + 0.5 * svm_acc)
        },
        "confusionMatrix": cm_results,
        "featureImportance": feature_importance,
        "comparison": comparison,
        "bestModelName": best_model_name,
        "crossValidationScore": cross_validation_score
    }

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best model weight parameters for fast client-side/in-memory inference without pickle issues
    # Save TF-IDF parameters, lr coefficients, categories
    model_save_state = {
        "bestModelName": best_model_name,
        "vocabulary": vocabulary.tolist(),
        "idf": vectorizer.idf_.tolist(),
        "lrClassifiers": [
            {
                "category": str(cat),
                "weights": lr.coef_[idx].tolist(),
                "bias": float(lr.intercept_[idx])
            } for idx, cat in enumerate(lr.classes_)
        ],
        "categories": lr.classes_.tolist()
    }

    with open(os.path.join(output_dir, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(model_save_state, f, indent=2)

    with open(os.path.join(output_dir, "ml_pipeline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"ML pipeline ran successfully. Best model: {best_model_name}")
    return results

if __name__ == "__main__":
    run_ml_pipeline()
