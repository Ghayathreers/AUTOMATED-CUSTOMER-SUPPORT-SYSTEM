#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@license
SPDX-License-Identifier: Apache-2.0
"""

from setuptools import setup, find_packages

setup(
    name="customer_support_ai",
    version="1.0.0",
    author="Expert Data Scientist & Software Architect",
    author_email="datascience@college.edu",
    description="Automated Customer Support and Service Resolution AI pipeline spanning ML, DL, NLP, SLMs, LLD, GenAI, and Agentic AI",
    long_description_content_type="text/markdown",
    url="https://github.com/academic/customer-support-ai",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "nltk>=3.8.0",
        "rouge-score>=0.1.2",
        "google-genai>=0.1.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "streamlit>=1.24.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    python_requires=">=3.11",
)
