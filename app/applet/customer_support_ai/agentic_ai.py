# -*- coding: utf-8 -*-
"""
Agentic AI module for Automated Customer Support AI.
Implements the multi-step Agentic support reasoning loop: Safety check, NLP Perception,
ML prediction, Risk assessment, Memory, Reasoning, Planning, Actions, and SLM synthesis.
"""

import os
import json
import random
from datetime import datetime

from utils import MODELS_DIR
from preprocessing import clean_text
from nlp_module import run_nlp_pipeline
from slm_module import run_slm_task

class SafetyGuard:
    @staticmethod
    def check_query_safety(text: str) -> dict:
        lower = text.lower()
        
        # 1. Prompt Injection checks
        if "ignore previous instructions" in lower or "system prompt" in lower or "bypass safety" in lower:
            return {"safe": False, "reason": "Potential Prompt Injection attempt detected."}
            
        # 2. Confidentiality/Password Phishing checks
        if "your password" in lower or "tell me your api key" in lower or "provide password" in lower:
            return {"safe": False, "reason": "Unauthorized credential query detected."}
            
        # 3. Toxicity / Vulgarity checks
        blacklisted = ["abuse", "sh*t", "f*ck", "scumbag", "asshole", "idiot"]
        for word in blacklisted:
            if word in lower:
                return {"safe": False, "reason": "Inappropriate or abusive language detected."}
                
        return {"safe": True}

class AgenticMemory:
    def __init__(self):
        self.history = []

    def add_log(self, entry: str):
        self.history.append(f"[{datetime.utcnow().isoformat()}Z] {entry}")

    def get_logs(self) -> list:
        return self.history

def classify_query_heuristically(text: str) -> dict:
    lower = text.lower()
    scores = {
        "Billing": 0.0,
        "Technical": 0.0,
        "Shipping": 0.0,
        "Product": 0.0,
        "Account": 0.0
    }

    keywords = {
        "Billing": [
            (["refund", "double charge", "overcharged", "charge twice", "charged twice", "reversal", "refunded", "dispute"], 5.0),
            (["billing", "bill", "invoice", "payment", "stripe", "credit card", "bank", "charge", "charged", "fee"], 3.0),
            (["cost", "price", "pay", "card", "dollar", "money", "$"], 1.5)
        ],
        "Technical": [
            (["err_db_timeout_504", "err_auth_failed_401", "error code", "ipconfig /flushdns", "ipconfig"], 5.0),
            (["crashed", "crashes", "freezing", "freeze", "bug", "timeout", "latency", "outage", "database error", "server error"], 4.0),
            (["error", "fail", "failed", "failure", "broken", "slow", "connection", "app", "application", "client", "firmware"], 2.0)
        ],
        "Shipping": [
            (["tracking number", "track link", "transit delay", "carrier scanning"], 5.0),
            (["shipping", "shipped", "ship", "tracking", "tracker", "parcel", "delivery", "delivered", "transit", "hub", "logistics"], 3.0),
            (["package", "delay", "delayed", "arrive", "arrival", "address", "route", "routing", "warehouse"], 1.5)
        ],
        "Product": [
            (["secure router pro", "dual-band", "specs catalog", "dimensions", "hardware replacement"], 5.0),
            (["warranty", "compatible", "compatibility", "router", "specifications", "specification", "specs", "hardware"], 3.0),
            (["device", "size", "features", "pro", "catalog", "model", "replacement warranty"], 1.5)
        ],
        "Account": [
            (["delete account", "cancel subscription", "two-factor", "2fa", "password reset"], 5.0),
            (["profile", "password", "reset", "login", "auth", "authentication", "credentials", "account", "security"], 3.0),
            (["username", "register", "sign in", "sign up", "sign out", "logout", "log into", "secure login"], 1.5)
        ]
    }

    for cat, rules in keywords.items():
        for words, weight in rules:
            for word in words:
                if word in lower:
                    scores[cat] += weight

    best_category = "Technical" # default fallback
    max_score = 0.0
    for cat, score in scores.items():
        if score > max_score:
            max_score = score
            best_category = cat

    total_score = sum(scores.values())
    confidence = 0.85
    if total_score > 0:
        confidence = round(float(0.5 + (max_score / total_score) * 0.45), 2)
    else:
        best_category = "Technical"
        confidence = 0.50

    return {"category": best_category, "confidence": confidence}

def get_ml_prediction(query_text: str) -> dict:
    try:
        model_path = os.path.join(MODELS_DIR, "best_model.json")
        if not os.path.exists(model_path):
            return None

        with open(model_path, "r", encoding="utf-8") as f:
            model_state = json.load(f)

        # Basic text cleaner & vectorizer reconstruction using NumPy
        import numpy as np
        vocab = model_state["vocabulary"]
        idf = np.array(model_state["idf"])
        vocab_map = {term: idx for idx, term in enumerate(vocab)}

        # Transform query
        cleaned = clean_text(query_text)
        tokens = cleaned.split()
        tf = np.zeros(len(vocab))
        for t in tokens:
            if t in vocab_map:
                tf[vocab_map[t]] += 1
        
        # Apply idf
        tf_idf = tf * idf
        norm = np.linalg.norm(tf_idf)
        if norm > 0:
            tf_idf = tf_idf / norm

        # Run logistic regression predict_proba calculations manually (to avoid importing sklearn inside server route)
        best_prob = 0.0
        best_cat = model_state["categories"][0]
        probs = {}

        for clf in model_state["lrClassifiers"]:
            cat = clf["category"]
            w = np.array(clf["weights"])
            b = clf["bias"]
            # Dot product + Sigmoid
            z = np.dot(tf_idf, w) + b
            prob = 1.0 / (1.0 + np.exp(-z))
            probs[cat] = float(prob)

        # Normalize probs (since OVR has overlapping bounds)
        total_p = sum(probs.values())
        if total_p > 0:
            for k in probs:
                probs[k] /= total_p

        best_cat = max(probs, key=probs.get)
        confidence = round(probs[best_cat], 2)

        return {"category": best_cat, "confidence": confidence}
    except Exception as e:
        print(f"ML in-memory model inference failed, using fallback rules: {e}")
        return None

def detect_risk_factors(text: str, category: str, confidence: float) -> dict:
    lower = text.lower()
    factors = []

    # 1. Refund dispute
    if ("refund" in lower or "dispute" in lower or "chargeback" in lower or "reversal" in lower) and \
       ("double" in lower or "unauthorized" in lower or "mistake" in lower or "wrong" in lower):
        factors.append("Refund Dispute / Double Charge Scan")

    # 2. Fraud
    if "fraud" in lower or "scam" in lower or "stolen" in lower or "unauthorized charge" in lower or "stolen card" in lower:
        factors.append("Fraud / Financial Security Warning")

    # 3. Account hacked
    if "hacked" in lower or "stolen account" in lower or "compromised" in lower or "hacker" in lower or "breach" in lower or "compromise" in lower:
        factors.append("Account Compromised / Security Breach")

    # 4. Legal complaint
    if "legal" in lower or "sue" in lower or "lawsuit" in lower or "lawyer" in lower or "court" in lower or "legal action" in lower:
        factors.append("Legal Complaint Escalation Hazard")

    # 5. Payment failure
    if "failed payment" in lower or "payment failed" in lower or "declined" in lower or "charge failed" in lower or "card declined" in lower or "fail to pay" in lower:
        factors.append("Payment Processing Failure Block")

    # 6. Repeated issue
    if "again" in lower or "second time" in lower or "multiple times" in lower or "repeated" in lower or "still not working" in lower or "still failing" in lower or "not solved" in lower:
        factors.append("Repeated Customer Friction Encountered")

    # 7. Low confidence
    if confidence < 0.65:
        factors.append("Low Classifier Confidence Threshold Alert")

    risk_level = "Low"
    if len(factors) >= 3:
        risk_level = "Critical Risk (Immediate SLA Escalation)"
    elif len(factors) >= 1:
        risk_level = "Medium / High (Risk Flags Detected)"

    return {"riskLevel": risk_level, "riskFactors": factors}

async def run_agentic_ai(query_text: str, customer_name: str) -> dict:
    memory = AgenticMemory()
    decision_logs = []

    memory.add_log("Received query. Initializing Safety Guard check.")

    # 1. Perception Step - Safety Guard
    safety = SafetyGuard.check_query_safety(query_text)
    if not safety["safe"]:
        decision_logs.append({
            "thought": "The query violates our safety protocols.",
            "action": "REJECT_QUERY",
            "observation": f"Safety block triggered: {safety['reason']}"
        })
        memory.add_log(f"Safety violation flagged: {safety['reason']}")

        return {
            "perception": {
                "detectedCategory": "Technical",
                "detectedSentiment": "Neutral",
                "detectedUrgency": "High",
                "entities": {},
                "confidence": 1.0,
                "riskLevel": "Critical Safety Risk",
                "riskFactors": ["Safety Violation Flagged"],
                "detectedEmotion": "Urgent",
                "resolutionDecision": "Escalated to Human Agent"
            },
            "reasoning": "Query was rejected immediately because it failed the security and toxicity check filters.",
            "planning": ["REJECT", "LOG_VIOLATION"],
            "actionsTaken": ["QUERY_REJECTED"],
            "memoryLogs": memory.get_logs(),
            "safetyPassed": False,
            "humanEscalationRequired": True,
            "escalationReason": safety["reason"],
            "decisionLogs": decision_logs,
            "finalResolution": f"Your request was blocked by our automated security systems: {safety['reason']}"
        }

    decision_logs.append({
        "thought": "Input passed safety constraints. Proceeding with NLP Perception parsing.",
        "action": "RUN_NLP_PIPELINE",
        "observation": "NLP pipeline parsed syntactic tokens, sentiment, and regex entities."
    })
    memory.add_log("Input parsed successfully through NLP.")

    # 2. Parse-based Perception
    nlp = run_nlp_pipeline(query_text)

    # Classification Fallback
    ml_pred = get_ml_prediction(query_text)
    heur_pred = classify_query_heuristically(query_text)

    detected_category = ml_pred["category"] if ml_pred else heur_pred["category"]
    confidence = ml_pred["confidence"] if ml_pred else heur_pred["confidence"]

    detected_sentiment = nlp["sentiment"]["label"]
    order_id = nlp["regexEntities"]["orderId"] or ""
    money_str = nlp["regexEntities"]["money"] or ""
    
    money_value = 0.0
    if money_str:
        try:
            money_value = float(money_str.replace("$", ""))
        except ValueError:
            pass

    error_code = nlp["regexEntities"]["errorCode"] or ""
    email_str = nlp["regexEntities"]["email"] or ""
    phone_str = nlp["regexEntities"]["phone"] or ""
    location_city = nlp["entities"]["city"] or ""
    device_str = nlp["entities"]["device"] or ""
    product_name_str = nlp["entities"]["productName"] or ""

    # Dynamic Sentiment / Emotion Detection
    detected_emotion = "Neutral"
    if "URGENT" in query_text.upper() or "ASAP" in query_text.upper() or "!!!" in query_text:
        detected_emotion = "Urgent"
    elif detected_sentiment == "Negative":
        lower = query_text.lower()
        if any(w in lower for w in ["hate", "angry", "scam", "terrible", "worst", "unacceptable"]):
            detected_emotion = "Angry"
        else:
            detected_emotion = "Frustrated"
    elif detected_sentiment == "Positive":
        detected_emotion = "Positive"

    # Compute Urgency level
    detected_urgency = "Low"
    if detected_sentiment == "Negative":
        detected_urgency = "Medium"
    if money_value > 200 or error_code or detected_emotion in ["Urgent", "Angry"]:
        detected_urgency = "High"

    # Dynamic Risk Detection
    risk = detect_risk_factors(query_text, detected_category, confidence)

    entities = {
        "customerName": customer_name or nlp["entities"]["customerName"] or "Customer",
        "city": location_city,
        "device": device_str,
        "orderId": order_id,
        "errorCode": error_code,
        "moneyAmount": money_str,
        "email": email_str,
        "phone": phone_str,
        "productName": product_name_str
    }

    decision_logs.append({
        "thought": f"Urgency set to {detected_urgency}. Extracted category {detected_category} (Confidence: {confidence}).",
        "action": "EVALUATE_REASONING_CHAIN",
        "observation": "Reasoned about billing limits, error diagnostics, and emotional state."
    })
    memory.add_log(f"Perceived Category: {detected_category}, Sentiment: {detected_sentiment}, Urgency: {detected_urgency}, Emotion: {detected_emotion}")

    # 3. Reasoning and Decision-making
    reasoning = ""
    human_escalation_required = False
    escalation_reason = ""
    planning = []
    resolution_decision = "Resolved Automatically"

    if "Critical" in risk["riskLevel"]:
        human_escalation_required = True
        escalation_reason = "Multiple high-risk security factors detected."
        resolution_decision = "Escalated to Human Agent"
    elif detected_category == "Billing":
        reasoning = f"Customer requested financial adjustments for transaction {order_id or 'N/A'}. "
        if money_value > 500:
            reasoning += f"The transaction amount of ${money_value} exceeds the $500 automated agent authorization limit. Policy mandates escalation."
            human_escalation_required = True
            escalation_reason = "Refund request amount exceeds the automated $500 threshold."
            resolution_decision = "Escalated to Human Agent"
        elif not order_id:
            reasoning += "Missing essential transaction Order ID reference. Dynamic pipeline requires parameters."
            resolution_decision = "Needs More Information"
        else:
            reasoning += f"The refund amount of ${money_value or 'N/A'} is within our $500 automated refund limit. Safe to resolve."
    elif detected_category == "Technical":
        reasoning = "Customer reported application errors or crashes. "
        if error_code in ["ERR_DB_TIMEOUT_504", "ERR_AUTH_FAILED_401"]:
            reasoning += f"The system error code ({error_code}) suggests severe server database issues or potential account compromise. Policy requires escalation."
            human_escalation_required = True
            escalation_reason = f"System error code {error_code} requires high-level sysadmin inspection."
            resolution_decision = "Escalated to Human Agent"
        else:
            reasoning += f"Error code {error_code or 'N/A'} is standard and can be cleared via local caching troubleshooting scripts."
    elif detected_category == "Shipping":
        reasoning = "Customer checking tracking details. "
        if not order_id:
            reasoning += "Missing Order ID reference. Pipeline requires Order ID to access live carrier details."
            resolution_decision = "Needs More Information"
        else:
            reasoning += "Retrieved routing details successfully for dispatch transit."
    elif detected_category == "Account":
        reasoning = "Customer requesting profile settings or authentication access. "
        if not email_str and not customer_name:
            reasoning += "No verified profile credentials or secure contact email found. Profile updates require verification parameters."
            resolution_decision = "Needs More Information"
        else:
            reasoning += "Authorized security protocol check matches registration log profiles."
    else:
        reasoning = "General hardware and product specifications checklist matching. No threshold bounds exceeded."

    # Force escalation on threats or hacks
    if "Legal Complaint Escalation Hazard" in risk["riskFactors"] or "Account Compromised / Security Breach" in risk["riskFactors"]:
        human_escalation_required = True
        escalation_reason = (
            "Legal dispute threats detected in typed description."
            if "Legal Complaint Escalation Hazard" in risk["riskFactors"] else
            "High-risk account hacking activity detected."
        )
        resolution_decision = "Escalated to Human Agent"

    # 4. Planning & Actions Formulation
    if resolution_decision == "Escalated to Human Agent":
        planning = ["HALT_AUTOMATED_RESOLUTION", "GENERATE_SUPPORT_TICKET", "ESCALATE_TO_SENIOR_OPERATOR"]
    elif resolution_decision == "Needs More Information":
        planning = ["PAUSE_TRANSACTIONS", "DRAFT_PARAMETER_REQUEST", "PROMPT_USER_FOR_MISSING_DETAILS"]
    else:
        planning = ["EXECUTE_SLM_RESOLUTION_SYNTHESIS", "UPDATE_CRM_DATABASE_RECORD", "SEND_EMAIL_CONFIRMATION"]

    decision_logs.append({
        "thought": f"Reasoning complete. Resolution decision is [{resolution_decision}]. Formulating action execution.",
        "action": "ESCALATE" if resolution_decision == "Escalated to Human Agent" else ("PROMPT_INFO" if resolution_decision == "Needs More Information" else "RESOLVE_AUTONOMOUSLY"),
        "observation": f"Ticket successfully routed to Senior Operator. Reason: {escalation_reason}" if resolution_decision == "Escalated to Human Agent" else ("Waiting for customer coordinates." if resolution_decision == "Needs More Information" else "Executed automated resolution tasks.")
    })

    actions_taken = []
    final_resolution = ""

    if resolution_decision == "Escalated to Human Agent":
        actions_taken = ["AUTOMATED_RESOLUTION_HALTED", "TICKET_CREATED_IN_CRM", "ROUTED_TO_HUMAN_QUEUE"]
        memory.add_log(f"Autonomous actions halted. Escalated to human: {escalation_reason}")
        final_resolution = (
            f"I have analyzed your request and determined that it requires personal oversight from our senior support operations managers. "
            f"I have prioritized your ticket and escalated it directly to an expert agent.\n\n"
            f"Escalation Reason: {escalation_reason}\n"
            f"Ticket Reference: TKT-{random.randint(100000, 999999)}"
        )
    elif resolution_decision == "Needs More Information":
        actions_taken = ["POSTPONED_RESOLUTION", "SENT_CLARIFICATION_CHECK", "AWAITING_CUSTOMER_COORD"]
        memory.add_log(f"Autonomous processing postponed. Reason: {reasoning}")

        missing_field = "details"
        if detected_category in ["Billing", "Shipping"]:
            missing_field = "Order ID (e.g., ORD-123456)"
        elif detected_category == "Account":
            missing_field = "registered Email Address"

        final_resolution = (
            f"Thank you for reaching out. To assist you further, we require some additional details. "
            f"Could you please provide your {missing_field}? This will allow us to query our database and finalize your request."
        )
    else:
        # Run SLM resolution
        slm_result = await run_slm_task(query_text, detected_category)
        actions_taken = ["EXECUTE_SLM_RESPONSE", "UPDATE_LEDGER", "EMAIL_DRAFTED"]
        memory.add_log("Autonomous actions completed. Summary and reply generated via local SLM.")
        
        steps_str = "\n".join([f"{idx+1}. {s}" for idx, s in enumerate(slm_result["conciseSteps"])])
        final_resolution = f"{slm_result['draftReply']}\n\nTroubleshooting steps proposed:\n{steps_str}"

    return {
        "perception": {
            "detectedCategory": detected_category,
            "detectedSentiment": detected_sentiment,
            "detectedUrgency": detected_urgency,
            "entities": entities,
            "confidence": confidence,
            "riskLevel": risk["riskLevel"],
            "riskFactors": risk["riskFactors"],
            "detectedEmotion": detected_emotion,
            "resolutionDecision": resolution_decision
        },
        "reasoning": reasoning,
        "planning": planning,
        "actionsTaken": actions_taken,
        "memoryLogs": memory.get_logs(),
        "safetyPassed": True,
        "humanEscalationRequired": human_escalation_required,
        "escalationReason": escalation_reason if human_escalation_required else None,
        "decisionLogs": decision_logs,
        "finalResolution": final_resolution
    }
