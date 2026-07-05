# Automated Customer Support & Service Resolution AI
### Seven Levels of Production Data Science, NLP, and Autonomous Agents

Welcome to the definitive **Automated Customer Support and Service Resolution AI** codebase. This is a comprehensive, production-grade, end-to-end Data Science, Machine Learning (ML), Deep Learning (DL), Natural Language Processing (NLP), Small Language Model (SLM), Low-Level Design (LLD), Generative AI (GenAI), and Agentic AI framework. 

This repository implements the complete, multi-tier automated pipeline inside a single structured environment.

---

## 📂 Project Taxonomy & Directory Layout

The project follows a modular, industry-standard structure, avoiding clutter while keeping all logical components perfectly isolated:

```text
customer_support_ai/
│
├── data/                      # Data persistence Layer
│   ├── customer_support_queries.csv   # 10,000 synthetic queries dataset (Primary)
│   └── customer_support_queries.json  # JSON-serialized token cache
│
├── models/                    # Serialized Weights & Hyperparameters
│   ├── best_model.json        # Saved traditional ML classifier state
│   ├── ml_pipeline_results.json # Metric scores, precision, recall, & CV reports
│   └── dl_model.json          # Deep Learning weights & tokenizer vocab keys
│
├── outputs/                   # Performance Logs & Analytical Graphics
│   ├── training_curves.png    # Matplotlib exported neural loss & accuracy logs
│   ├── roc_auc_curves.png     # ROC evaluation graphs
│   └── system_benchmark.log   # Latency benchmarks & evaluation traces
│
├── src/                       # Central Logic Core
│   ├── types.ts               # Shared TypeScript interfaces & structures
│   ├── data_generator.ts      # synthetic randomizer compiling 10,000 records
│   ├── ml_module.ts           # Level 1: TF-IDF vectorizer + LR + SVM + RF
│   ├── dl_module.ts           # Level 2: Tokenizer + Embeddings + MLP + LSTM + BiLSTM
│   ├── nlp_module.ts          # Level 3: Clean, Lemmatize, Regex, Sentiment, Named Entities
│   ├── slm_module.ts          # Level 4: Summarization, Actionable steps, ROUGE Evaluation
│   ├── genai_module.ts        # Level 6: Emails drafts, context FAQs, Guides, dialogues
│   ├── agentic_module.ts      # Level 7: Safety Guard, Perceptions, Reasoning, Planning, Actions
│   ├── api/                   # FastAPI-like Express controllers
│   │   └── index.ts           # Double-mapped router mapping /health, /predict, /agent
│   └── config/                # Path configurations
│       └── index.ts           # Absolute path resolves and pipeline bounds
│
├── requirements.txt           # Python environment packages listing
├── setup.py                   # Standard package configuration script
├── README.md                  # Comprehensive mathematical & layout documentation (This file)
└── VIVA_GUIDE.md              # Complete exam and viva voce preparation manual
```

---

## 🧮 Mathematical Foundations

Our framework relies on explicit mathematical formulations across all levels, implemented from first principles to guarantee absolute transparent integrity.

### 1. TF-IDF Text Representation (Term Frequency-Inverse Document Frequency)
Let $t$ be a token (term) and $d$ be a specific document (query) in a corpus $D$.
$$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total tokens in } d}$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
The resulting sparse vectors are normalized using $L_2$ Euclidean normalization:
$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$

### 2. Support Vector Machine Optimization (SGD Solver)
Our Linear SVM optimizes the standard **Hinge Loss** regularization with $L_2$ weight penalty ($C = 1.0$):
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2_2 + C \sum_{i=1}^{N} \max\left(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\right)$$
Multi-class classification is mapped using the **One-vs-Rest (OvR)** scheme.

### 3. Deep Learning Feedforward & Backpropagation Equations
- **Embeddings Layer**: Maps sequence index $i \in \mathbb{R}$ to dense vector $\mathbf{e} \in \mathbb{R}^d$ using weights $\mathbf{W}_E$:
  $$\mathbf{e} = \mathbf{W}_E \cdot \text{one\_hot}(i)$$
- **LSTM Cell Recurrent Update**:
  $$f_t = \sigma(\mathbf{W}_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)}$$
  $$i_t = \sigma(\mathbf{W}_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$
  $$\tilde{C}_t = \tanh(\mathbf{W}_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(Candidate Cell State)}$$
  $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \quad \text{(Cell State Update)}$$
  $$o_t = \sigma(\mathbf{W}_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)}$$
  $$h_t = o_t * \tanh(C_t) \quad \text{(Hidden Output State)}$$
- **Softmax Probability Output**:
  $$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{z}_c}}{\sum_{j=1}^{C} e^{\mathbf{z}_j}}$$

### 4. ROUGE Metrics Evaluation (Recall-Oriented Understudy for Gisting Evaluation)
Let $C_{\text{cand}}$ be the candidate summary and $R_{\text{ref}}$ be the ground-truth reference.
- **ROUGE-N** (N-gram overlap, e.g. unigrams or bigrams):
  $$\text{ROUGE-N} = \frac{\sum_{s \in R_{\text{ref}}} \sum_{\text{gram}_n \in s} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{s \in R_{\text{ref}}} \sum_{\text{gram}_n \in s} \text{Count}(\text{gram}_n)}$$
- **ROUGE-L** (Longest Common Subsequence):
  Let $\text{LCS}(C_{\text{cand}}, R_{\text{ref}})$ be the longest matching token sequence.
  $$R_{\text{lcs}} = \frac{\text{LCS}(C_{\text{cand}}, R_{\text{ref}})}{|R_{\text{ref}}|} \quad \text{(Recall)}$$
  $$P_{\text{lcs}} = \frac{\text{LCS}(C_{\text{cand}}, R_{\text{ref}})}{|C_{\text{cand}}|} \quad \text{(Precision)}$$
  $$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{lcs}} P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}$$

---

## 🚀 Execution Instructions

Follow these instructions to spin up the local servers, initialize automatic training, and run the Streamlit UI Emulator:

### 1. Environment Setup
Clone the repository and ensure you have Node.js (v18+) and Python (v3.11+) installed:
```bash
# Verify versions
node -v
python --version
```

### 2. Server Installation & Execution
```bash
# Install core workspace packages
npm install

# Start the dev environment
# This will automatically trigger synthetic data generation and run ML & DL training
npm run dev
```

### 3. Open the Streamlit Console
Open your browser and navigate to:
```text
http://localhost:3000
```
This launches our interactive, beautiful **Streamlit Web Console Emulator**. Try selecting scenario presets or typing raw customer queries to watch the Agentic loops, safety sweeps, decision trees, neural curves, and LLD diagrams update in real-time.

---

## 🛡️ License
Copyright © 2026. All rights reserved. Licensed under the Apache-2.0 License.


## Added Design and Evaluation Artifacts

This modified version includes missing LLD and evaluation outputs in `docs/` and `outputs/`:

- System Architecture Diagram
- Component Diagram
- Sequence Diagram
- ER/Data Model Diagram
- Deployment Architecture Diagram
- ROC Curve for ML classification
- `generate_lld_and_roc.py` to regenerate the ROC curve

To regenerate the ROC curve:

```bash
cd customer_support_ai
python generate_lld_and_roc.py
```
