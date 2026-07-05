# -*- coding: utf-8 -*-
"""
Utils module for Automated Customer Support AI.
Contains path configurations and common mathematical / evaluation helpers.
"""

import os
import re

# Resolve paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Common helper: tokenize text to words
def tokenize_to_words(text):
    if not text:
        return []
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return [w for w in cleaned.split() if w]

# Common helper: get bigrams
def get_bigrams(words):
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]}_{words[i+1]}")
    return bigrams

# Common helper: Longest Common Subsequence DP length
def get_lcs_length(words1, words2):
    m, n = len(words1), len(words2)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# Common helper: Compute ROUGE-1, ROUGE-2, ROUGE-L
def compute_rouge_scores(prediction, reference):
    pred_words = tokenize_to_words(prediction)
    ref_words = tokenize_to_words(reference)
    
    if not pred_words or not ref_words:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
    # ROUGE-1
    pred_set = set(pred_words)
    overlap1 = sum(1 for w in ref_words if w in pred_set)
    rouge1 = overlap1 / len(ref_words)
    
    # ROUGE-2
    pred_bg = get_bigrams(pred_words)
    ref_bg = get_bigrams(ref_words)
    if not ref_bg:
        return {"rouge1": round(rouge1, 4), "rouge2": 0.0, "rougeL": 0.0}
    pred_bg_set = set(pred_bg)
    overlap2 = sum(1 for b in ref_bg if b in pred_bg_set)
    rouge2 = overlap2 / len(ref_bg)
    
    # ROUGE-L
    lcs = get_lcs_length(pred_words, ref_words)
    rougeL = lcs / len(ref_words)
    
    return {
        "rouge1": round(rouge1, 4),
        "rouge2": round(rouge2, 4),
        "rougeL": round(rougeL, 4)
    }
