# -*- coding: utf-8 -*-
"""Generate missing LLD diagrams and ROC curve for the customer support project.
Run from customer_support_ai folder:
    python generate_lld_and_roc.py
"""
from pathlib import Path
import json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

BASE = Path(__file__).resolve().parent
DOCS = BASE / "docs"
OUTS = BASE / "outputs"
DOCS.mkdir(exist_ok=True)
OUTS.mkdir(exist_ok=True)

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def box(ax, xy, w, h, text, fontsize=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", linewidth=1.4, facecolor="white", edgecolor="black"))
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=12, linewidth=1.2))

def generate_roc():
    df = pd.read_csv(BASE / "data" / "customer_support_queries.csv")
    texts = df["query"].map(clean_text)
    y = df["category"].astype(str).values
    classes = sorted(pd.Series(y).unique())
    X = TfidfVectorizer(max_features=120, stop_words="english").fit_transform(texts).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    y_test_bin = label_binarize(y_test, classes=classes)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42),
        "SVM": SVC(probability=True, kernel="linear", random_state=42),
    }
    meta = {}
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), score.ravel())
        roc_auc = auc(fpr, tpr)
        meta[name] = float(roc_auc)
        plt.plot(fpr, tpr, label=f"{name} micro AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Customer Query Intent Classification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTS / "roc_curve.png", dpi=200)
    plt.savefig(DOCS / "roc_curve.png", dpi=200)
    plt.close()
    (OUTS / "roc_auc_results.json").write_text(json.dumps({"classes": classes, "micro_auc": meta}, indent=2), encoding="utf-8")
    return meta

# Diagrams are already included in docs in this modified zip. This script focuses on reproducible ROC generation.
if __name__ == "__main__":
    print("Generating ROC curve...")
    print(generate_roc())
    print("Saved docs/roc_curve.png and outputs/roc_curve.png")
