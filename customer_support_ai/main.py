# -*- coding: utf-8 -*-
"""
FastAPI Backend for Automated Customer Support AI.
Serves Python ML, DL, NLP, SLM, GenAI, and Agentic AI features,
and manages customer search, order fetching, and dashboard statistics.
"""

import os
import sys
import json
import asyncio
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add current folder to system path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DATA_DIR, MODELS_DIR
from preprocessing import clean_text
from data_generator import generate_full_dataset, save_full_dataset_to_csv_and_json
from ml_model import run_ml_pipeline
from dl_model import run_dl_pipeline
from nlp_module import run_nlp_pipeline
from slm_module import run_slm_task
from genai_module import run_genai_tasks
from agentic_ai import run_agentic_ai

app = FastAPI(
    title="Customer Support AI Backend",
    description="Python FastAPI backend powering Machine Learning, Deep Learning, NLP, and Agentic loops.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- IN-MEMORY CACHE FOR CSV DATA ---
_cached_customers = None
_cached_orders = None
_cached_queries = None

def load_customers_df():
    global _cached_customers
    if _cached_customers is not None:
        return _cached_customers
    path = os.path.join(DATA_DIR, "customers.csv")
    if os.path.exists(path):
        _cached_customers = pd.read_csv(path)
    else:
        _cached_customers = pd.DataFrame()
    return _cached_customers

def load_orders_df():
    global _cached_orders
    if _cached_orders is not None:
        return _cached_orders
    path = os.path.join(DATA_DIR, "orders.csv")
    if os.path.exists(path):
        _cached_orders = pd.read_csv(path)
    else:
        _cached_orders = pd.DataFrame()
    return _cached_orders

def load_queries_df():
    global _cached_queries
    if _cached_queries is not None:
        return _cached_queries
    path = os.path.join(DATA_DIR, "support_queries.csv")
    if os.path.exists(path):
        _cached_queries = pd.read_csv(path)
    else:
        _cached_queries = pd.DataFrame()
    return _cached_queries

def clear_cache():
    global _cached_customers, _cached_orders, _cached_queries
    _cached_customers = None
    _cached_orders = None
    _cached_queries = None

# Ensure dataset exists on startup
dataset_path = os.path.join(DATA_DIR, "customers.csv")
if not os.path.exists(dataset_path):
    print("Dataset not found on startup. Generating 10,000 customers...")
    ds = generate_full_dataset(10000)
    save_full_dataset_to_csv_and_json(ds, DATA_DIR)
    
# Ensure models are trained on startup if best_model.json is missing
model_check_path = os.path.join(MODELS_DIR, "best_model.json")
if not os.path.exists(model_check_path):
    print("Trained models not found on startup. Running ML & DL pipelines...")
    try:
        ml_res = run_ml_pipeline()
        run_dl_pipeline(baseline_accuracy=ml_res["svm"]["accuracy"])
    except Exception as e:
        print(f"Startup model training failed: {e}")

# --- PYDANTIC SCHEMAS ---
class QueryRequest(BaseModel):
    query: str

class ChatRequest(BaseModel):
    query: str
    category: Optional[str] = "Technical"

class GenAIRequest(BaseModel):
    query: str
    category: Optional[str] = "Technical"
    customerName: Optional[str] = "Customer"
    sentiment: Optional[str] = "Neutral"
    orderId: Optional[str] = ""

class AgentRequest(BaseModel):
    query: str
    customerName: Optional[str] = "Customer"

class ChatHistoryItem(BaseModel):
    sender: str  # "user" or "agent"
    text: str

class InteractiveChatRequest(BaseModel):
    history: List[ChatHistoryItem]
    customerName: Optional[str] = "Customer"
    category: Optional[str] = "General Support"
    query: Optional[str] = ""

# --- ENDPOINTS ---

# 1. GET /api/health
@app.get("/api/health")
@app.get("/health")
def health_check():
    dataset_exists = os.path.exists(os.path.join(DATA_DIR, "support_queries.csv"))
    model_exists = os.path.exists(os.path.join(MODELS_DIR, "best_model.json"))
    dl_model_exists = os.path.exists(os.path.join(MODELS_DIR, "dl_model.json"))

    return {
        "status": "healthy",
        "service": "Automated Customer Support and Service Resolution AI (Python FastAPI Engine)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "environment": {
            "platform": "Python",
            "framework": "FastAPI (Express proxy integration)",
            "port": 8000
        },
        "pipelineState": {
            "datasetGenerated": bool(dataset_exists),
            "mlModelTrained": bool(model_exists),
            "dlModelTrained": bool(dl_model_exists)
        }
    }

# 2. POST /api/predict
@app.post("/api/predict")
@app.post("/predict")
def predict_intent(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing required parameter: query")

    try:
        best_model_path = os.path.join(MODELS_DIR, "best_model.json")
        results_path = os.path.join(MODELS_DIR, "ml_pipeline_results.json")

        if not os.path.exists(best_model_path) or not os.path.exists(results_path):
            raise HTTPException(status_code=503, detail="ML Pipeline is currently training. Please try again shortly.")

        with open(best_model_path, "r", encoding="utf-8") as f:
            model_state = json.load(f)
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Run vectorizer + manual Logistic Regression predict probs
        import numpy as np
        vocab = model_state["vocabulary"]
        idf = np.array(model_state["idf"])
        vocab_map = {term: idx for idx, term in enumerate(vocab)}

        # Transform query
        cleaned = clean_text(req.query)
        tokens = cleaned.split()
        tf = np.zeros(len(vocab))
        for t in tokens:
            if t in vocab_map:
                tf[vocab_map[t]] += 1
        
        tf_idf = tf * idf
        norm = np.linalg.norm(tf_idf)
        if norm > 0:
            tf_idf = tf_idf / norm

        probs = {}
        for clf in model_state["lrClassifiers"]:
            cat = clf["category"]
            w = np.array(clf["weights"])
            b = clf["bias"]
            z = np.dot(tf_idf, w) + b
            prob = 1.0 / (1.0 + np.exp(-z))
            probs[cat] = float(prob)

        # Normalize probabilities
        total_p = sum(probs.values())
        if total_p > 0:
            for k in probs:
                probs[k] /= total_p

        predicted_category = max(probs, key=probs.get)

        return {
            "query": req.query,
            "predictedCategory": predicted_category,
            "confidenceScores": probs,
            "pipelineSummary": {
                "bestModelName": results.get("bestModelName", "Logistic Regression"),
                "crossValidationScore": results.get("crossValidationScore", 0.76),
                "traditionalMLComparisons": results.get("comparison", []),
                "featureImportances": results.get("featureImportance", [])[:10],
                "confusionMatrix": results.get("confusionMatrix", {})
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Predict endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal Model Inference Failure")

# 3. POST /api/analyze
@app.post("/api/analyze")
@app.post("/analyze")
def analyze_nlp(req: QueryRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing required parameter: query")
    try:
        return run_nlp_pipeline(req.query)
    except Exception as e:
        print(f"Analyze endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal NLP Analysis Failure")

# 4. POST /api/chat
@app.post("/api/chat")
@app.post("/chat")
async def slm_chat(req: ChatRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing required parameter: query")
    try:
        return await run_slm_task(req.query, req.category)
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal SLM Inference Failure")

# 5. POST /api/generate
@app.post("/api/generate")
@app.post("/generate")
async def genai_pipeline(req: GenAIRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing required parameter: query")
    try:
        return await run_genai_tasks(
            req.query, req.category, req.customerName, req.sentiment, req.orderId
        )
    except Exception as e:
        print(f"Generate endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal Generative AI Failure")

# 6. POST /api/agent
@app.post("/api/agent")
@app.post("/agent")
async def agentic_loop(req: AgentRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Missing required parameter: query")
    try:
        return await run_agentic_ai(req.query, req.customerName)
    except Exception as e:
        print(f"Agent endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal Agentic Loop Failure")

# 7. POST /api/chat-message
@app.post("/api/chat-message")
@app.post("/chat-message")
async def chat_message_interactive(req: InteractiveChatRequest):
    if not req.history:
        raise HTTPException(status_code=400, detail="Missing or empty interactive history")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    customer_name = req.customerName or "Customer"
    category = req.category or "General Support"
    initial_query = req.query or ""

    if api_key and api_key != "MY_GEMINI_API_KEY" and api_key != "":
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            system_prompt = (
                f"You are an empathetic, smart, and professional customer support assistant assisting {customer_name}.\n"
                f"The original customer ticket query was: \"{initial_query}\" (Category: {category}).\n"
                "Your task is to respond to the customer's messages in this ongoing chat session.\n"
                "Keep your responses helpful, polite, concise (max 2-3 sentences), and highly relevant to the context. "
                "Do not break character. Do not repeat greeting messages. Keep the tone human-like and professional."
            )

            contents = []
            for h in req.history:
                contents.append(
                    types.Content(
                        role="user" if h.sender == "user" else "model",
                        parts=[types.Part.from_text(text=h.text)]
                    )
                )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7
                )
            )
            return {"reply": response.text or "I am processing your request. Please let me know if you have any other details."}
        except Exception as e:
            print(f"Gemini interactive chat failed gracefully: {e}")

    # Fallback Rule Engine
    last_msg = req.history[-1].text.lower() if req.history else ""
    name = customer_name

    if any(w in last_msg for w in ["hello", "hi", "hey"]):
        reply = f"Hello {name}! How can I help you with your {category.lower()} query today?"
    elif any(w in last_msg for w in ["refund", "charge", "money", "billing"]):
        reply = f"I understand this billing item is important, {name}. Our financial reversal pipelines are currently queued for your case. Refunds typically credit within 3 business days depending on your bank."
    elif any(w in last_msg for w in ["track", "ship", "package", "delay"]):
        reply = f"I have locked the logistics scanner. The package is currently processing at our sorting depot and is prioritized for dispatch. I will update your account with live carrier updates."
    elif any(w in last_msg for w in ["password", "login", "account", "auth"]):
        reply = f"For security, please ensure you use a strong unique password. We can trigger an encrypted profile password-reset link directly to your email on request."
    elif any(w in last_msg for w in ["error", "crash", "fail", "freeze"]):
        reply = "Our DevOps monitors are actively tracking this error log. Please try clearing your local cache or reloading the application to resolve any sync conflicts."
    elif any(w in last_msg for w in ["thank", "appreciate", "solved", "ok"]):
        reply = f"You're very welcome, {name}! I'm glad I could assist. Please let me know if there's anything else I can do for you."
    else:
        reply = f"I have logged your request: \"{req.history[-1].text}\". Our service resolution engine is monitoring your ticket and we will ensure everything is settled. Is there any other detail you'd like to share?"

    return {"reply": reply}

# 8. GET /api/customers/search
@app.get("/api/customers/search")
@app.get("/customers/search")
def search_customers(
    q: Optional[str] = "",
    category: Optional[str] = ""
):
    df_c = load_customers_df()
    df_q = load_queries_df()

    if df_c.empty:
        return []

    filtered = df_c

    # Category filter
    if category:
        cat_lower = category.lower()
        matching_q = df_q[df_q["Query Category"].str.lower() == cat_lower]
        matching_cust_ids = set(matching_q["Customer ID"].dropna().unique())
        filtered = filtered[filtered["Customer ID"].isin(matching_cust_ids)]

    # Search query filter
    if q:
        q_lower = q.lower()
        filtered = filtered[filtered["Customer Name"].str.lower().str.contains(q_lower, na=False)]

    # Slice and map to format
    res_list = filtered.head(10).to_dict(orient="records")
    out = []
    for c in res_list:
        out.append({
            "id": c["Customer ID"],
            "name": c["Customer Name"],
            "email": c["Email"],
            "phone": c["Phone Number"],
            "city": c["City"],
            "state": c["State"],
            "segment": c["Customer Segment"],
            "support_history_count": int(c["Support History Count"] or 0)
        })
    return out

# 9. GET /api/customers/{id}/orders
@app.get("/api/customers/{id}/orders")
@app.get("/customers/{id}/orders")
def get_customer_orders(id: str):
    df_o = load_orders_df()
    if df_o.empty:
        return []

    filtered = df_o[df_o["Customer ID"] == id]
    res_list = filtered.to_dict(orient="records")
    out = []
    for o in res_list:
        out.append({
            "orderId": o["Order ID"],
            "customerId": o["Customer ID"],
            "productName": o["Product Name"],
            "productCategory": o["Product Category"],
            "orderDate": o["Order Date"],
            "orderAmount": float(o["Order Amount"] or 0.0),
            "paymentStatus": o["Payment Status"],
            "deliveryStatus": o["Delivery Status"],
            "expectedDeliveryDate": o["Expected Delivery Date"],
            "refundStatus": o["Refund Status"],
            "warrantyStatus": o["Warranty Status"]
        })
    return out

# 10. GET /api/orders/search
@app.get("/api/orders/search")
@app.get("/orders/search")
def search_orders(q: Optional[str] = ""):
    df_o = load_orders_df()
    df_c = load_customers_df()

    if df_o.empty:
        return []

    if not q:
        sample_list = df_o.head(15).to_dict(orient="records")
    else:
        q_lower = q.lower()
        mask = (
            df_o["Order ID"].astype(str).str.lower().str.contains(q_lower, na=False) |
            df_o["Product Name"].astype(str).str.lower().str.contains(q_lower, na=False) |
            df_o["Customer ID"].astype(str).str.lower().str.contains(q_lower, na=False)
        )
        sample_list = df_o[mask].head(15).to_dict(orient="records")

    out = []
    for o in sample_list:
        cust_match = df_c[df_c["Customer ID"] == o["Customer ID"]]
        city = "Austin"
        if not cust_match.empty:
            city = cust_match.iloc[0]["City"]

        out.append({
            "orderId": o["Order ID"],
            "customerId": o["Customer ID"],
            "productName": o["Product Name"],
            "productCategory": o["Product Category"],
            "orderDate": o["Order Date"],
            "orderAmount": float(o["Order Amount"] or 0.0),
            "paymentStatus": o["Payment Status"],
            "deliveryStatus": o["Delivery Status"],
            "expectedDeliveryDate": o["Expected Delivery Date"],
            "refundStatus": o["Refund Status"],
            "warrantyStatus": o["Warranty Status"],
            "shippingCity": city
        })
    return out

# 11. GET /api/analytics
@app.get("/api/analytics")
@app.get("/analytics")
def get_dashboard_analytics():
    try:
        df_c = load_customers_df()
        df_o = load_orders_df()
        df_q = load_queries_df()

        if df_c.empty or df_o.empty or df_q.empty:
            return {
                "totalCustomers": 0,
                "totalOrders": 0,
                "totalSupportTickets": 0,
                "escalationCount": 0,
                "averageResolutionTime": 14.5,
                "categoryDistribution": {},
                "sentimentDistribution": {},
                "resolutionStatusDistribution": {}
            }

        # Distributions
        cat_dist = df_q["Query Category"].value_counts().to_dict()
        sent_dist = df_q["Sentiment"].value_counts().to_dict()
        res_dist = df_q["Resolution Status"].value_counts().to_dict()

        escalations = int((df_q["Escalation Status"] == "Yes").sum() + (df_q["Resolution Status"] == "escalated").sum())

        # Average Resolution Time computation based on Priority mapping
        total_time = 0.0
        resolved_df = df_q[df_q["Resolution Status"] == "resolved"]
        resolved_count = len(resolved_df)

        for _, q in resolved_df.iterrows():
            p = str(q["Priority"]).lower()
            if p == "urgent":
                total_time += 2.0
            elif p == "high":
                total_time += 6.0
            elif p == "medium":
                total_time += 12.0
            else:
                total_time += 24.0

        average_resolution_time = round(total_time / resolved_count, 1) if resolved_count > 0 else 14.5

        return {
            "totalCustomers": len(df_c),
            "totalOrders": len(df_o),
            "totalSupportTickets": len(df_q),
            "escalationCount": escalations,
            "averageResolutionTime": average_resolution_time,
            "categoryDistribution": {str(k): int(v) for k, v in cat_dist.items()},
            "sentimentDistribution": {str(k): int(v) for k, v in sent_dist.items()},
            "resolutionStatusDistribution": {str(k): int(v) for k, v in res_dist.items()}
        }
    except Exception as e:
        print(f"Analytics computation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to load analytics dashboard data")

# 12. POST /api/regenerate-dataset
@app.post("/api/regenerate-dataset")
@app.post("/regenerate-dataset")
def regenerate_dataset():
    try:
        print("Regenerating 10,000 records programmatically in Python...")
        ds = generate_full_dataset(10000)
        save_full_dataset_to_csv_and_json(ds, DATA_DIR)
        
        # Clear pandas caches
        clear_cache()

        print("Retraining models dynamically in Python...")
        ml_res = run_ml_pipeline()
        run_dl_pipeline(baseline_accuracy=ml_res["svm"]["accuracy"])

        return {
            "status": "success",
            "message": "Dataset successfully regenerated with 10,000 customers. Models retrained.",
            "totalCustomers": 10000,
            "totalSupportTickets": 10000
        }
    except Exception as e:
        print(f"Regeneration endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate dataset: {str(e)}")
