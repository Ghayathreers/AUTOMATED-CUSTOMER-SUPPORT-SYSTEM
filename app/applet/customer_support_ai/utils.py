# -*- coding: utf-8 -*-
"""
Common utilities and project path definitions for Automated Customer Support AI.
"""

import os

# Align paths to the root of the customer_support_ai folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Root data directory (also write here to satisfy Vite and general pathing)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
WORKSPACE_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data")

# Ensure all directories exist
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, WORKSPACE_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

def verify_file_exists(file_path: str) -> bool:
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0
