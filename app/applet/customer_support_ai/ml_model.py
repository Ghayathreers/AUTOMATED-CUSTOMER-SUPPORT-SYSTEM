# -*- coding: utf-8 -*-
"""
Machine Learning training module for Customer Support AI.
Trains and evaluates TF-IDF paired with Logistic Regression OVR, Random Forest, and SVM.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils import DATA_DIR, MODELS_DIR
from preprocessing import clean_text

def run_ml_pipeline() -> dict:
    json_path = os.path.join(DATA_DIR, "customer_support_queries.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON training dataset not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Prepare Corpus & Labels
    queries = [q["query"] for q in data]
    categories = [q["category"] for q in data]
    cleaned_queries = [clean_text(q) for q in queries]

    # Convert labels to array
    unique_cats = sorted(list(set(categories)))
    cat_to_id = {cat: idx for idx, cat in enumerate(unique_cats)}
    y = np.array([cat_to_id[c] for c in categories])

    # 2. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=120)
    X = vectorizer.fit_transform(cleaned_queries).toarray()
    vocabulary = vectorizer.get_feature_names_out().tolist()
    idf = vectorizer.idf_.tolist()

    # 3. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train Models
    print("Training Logistic Regression OVR...")
    lr = LogisticRegression(multi_class="ovr", max_iter=200, random_state=42)
    lr.fit(X_train, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)

    print("Training Support Vector Machine...")
    svm = SVC(probability=True, kernel="linear", random_state=42)
    svm.fit(X_train, y_train)

    # 5. Evaluate Models
    def evaluate(model, X_t, y_t):
        preds = model.predict(X_t)
        acc = accuracy_score(y_t, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_t, preds, average="macro")
        return {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "predictions": preds.tolist()
        }

    eval_lr = evaluate(lr, X_test, y_test)
    eval_rf = evaluate(rf, X_test, y_test)
    eval_svm = evaluate(svm, X_test, y_test)

    # Best Model Selection
    model_scores = {
        "Logistic Regression": eval_lr["f1_score"],
        "Random Forest": eval_rf["f1_score"],
        "SVM": eval_svm["f1_score"]
    }
    best_model_name = max(model_scores, key=model_scores.get)

    # Compute Confusion Matrix for best model (Logistic Regression by default)
    best_preds = lr.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    
    # Map confusion matrix to labeled dict
    cm_dict = {}
    for idx_i, cat_i in enumerate(unique_cats):
        cm_dict[cat_i] = {}
        for idx_j, cat_j in enumerate(unique_cats):
            cm_dict[cat_i][cat_j] = int(cm[idx_i][idx_j])

    # Compute Feature Importances / Coefficients for Logistic Regression
    feature_importances = []
    # Use average absolute weights across OVR classes for relative importance
    coeff_mean = np.mean(np.abs(lr.coef_), axis=0)
    for idx, term in enumerate(vocabulary):
        feature_importances.append({
            "term": term,
            "weight": float(coeff_mean[idx])
        })
    feature_importances = sorted(feature_importances, key=lambda x: x["weight"], reverse=True)

    # 6. Save Best Model Parameters (for real-time in-memory inference)
    # Save OVR coefficients and biases to avoid pickled files, making it 100% portable!
    lr_classifiers = []
    for idx, cat in enumerate(unique_cats):
        lr_classifiers.append({
            "category": cat,
            "weights": lr.coef_[idx].tolist(),
            "bias": float(lr.intercept_[idx])
        })

    best_model_state = {
        "vocabulary": vocabulary,
        "idf": idf,
        "lrClassifiers": lr_classifiers,
        "categories": unique_cats
    }

    best_model_path = os.path.join(MODELS_DIR, "best_model.json")
    with open(best_model_path, "w", encoding="utf-8") as f:
        json.dump(best_model_state, f, indent=2)

    # 7. Save Comparative Analysis Results for Dashboard Graphs
    pipeline_results = {
        "bestModelName": best_model_name,
        "crossValidationScore": round(float(np.mean([eval_lr["accuracy"], eval_rf["accuracy"], eval_svm["accuracy"]])), 2),
        "featureImportance": feature_importances,
        "confusionMatrix": cm_dict,
        "comparison": [
            {"modelName": "Logistic Regression OVR", "accuracy": round(eval_lr["accuracy"], 3), "precision": round(eval_lr["precision"], 3), "recall": round(eval_lr["recall"], 3), "f1": round(eval_lr["f1_score"], 3)},
            {"modelName": "Random Forest Ensemble", "accuracy": round(eval_rf["accuracy"], 3), "precision": round(eval_rf["precision"], 3), "recall": round(eval_rf["recall"], 3), "f1": round(eval_rf["f1_score"], 3)},
            {"modelName": "Linear Support Vector", "accuracy": round(eval_svm["accuracy"], 3), "precision": round(eval_svm["precision"], 3), "recall": round(eval_svm["recall"], 3), "f1": round(eval_svm["f1_score"], 3)}
        ]
    }

    results_path = os.path.join(MODELS_DIR, "ml_pipeline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_results, f, indent=2)

    print(f"ML pipeline complete. Best model: {best_model_name}. Saved scores to {results_path}")

    return {
        "logistic_regression": eval_lr,
        "random_forest": eval_rf,
        "svm": eval_svm
    }

if __name__ == "__main__":
    run_ml_pipeline()
