# -*- coding: utf-8 -*-
"""
Natural Language Processing (NLP) module for Customer Support AI.
Implements regex extraction, rule-based NER, keyword parsing, and sentiment analysis.
"""

import re
from preprocessing import clean_text

# Sentiment Lexicons
POSITIVE_WORDS = {
    "great", "excellent", "awesome", "perfect", "good", "happy", "pleased", "helpful",
    "love", "amazing", "wonderful", "satisfied", "solve", "solved", "thanks", "thank",
    "correct", "working", "resolved", "appreciate", "superb", "fast", "speedy"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "worst", "broken", "angry", "hate", "unhappy", "frustrated",
    "annoyed", "useless", "fail", "failed", "failure", "crash", "crashed", "crashes",
    "freeze", "freezes", "freezing", "bug", "error", "timeout", "slow", "delay",
    "delayed", "wrong", "broken", "damage", "damaged", "scam", "cheat", "overcharged",
    "unacceptable", "poor", "difficult", "unauthorized", "dispute", "disputed"
}

# Rule-based NER Pools
CITIES_POOL = {
    "austin", "san francisco", "new york", "seattle", "denver", "miami", "chicago",
    "boston", "portland", "los angeles", "atlanta", "phoenix", "dallas", "san diego",
    "salt lake city", "nashville"
}

PRODUCTS_POOL = {
    "secure router pro", "mesh extender dual-band", "mesh extender", "gigabit switch 8-port",
    "gigabit switch", "saas cloud firewall plus", "saas cloud firewall",
    "enterprise vpn client license", "enterprise vpn client", "premium support sla ticket",
    "premium support", "router"
}

DEVICES_POOL = {
    "iphone", "android", "macbook", "windows", "ipad", "router", "switch", "laptop",
    "mobile", "desktop", "hardware", "firmware", "device"
}

NAMES_POOL = {
    "alex", "jordan", "taylor", "morgan", "sam", "jamie", "chris", "pat", "robin"
}

def analyze_sentiment(text: str) -> dict:
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    score = 0.0
    if pos_count > 0 or neg_count > 0:
        score = float(pos_count - neg_count) / float(pos_count + neg_count)

    label = "Neutral"
    if score > 0.15:
        label = "Positive"
    elif score < -0.15:
        label = "Negative"

    return {
        "score": round(score, 2),
        "label": label,
        "positiveWordCount": pos_count,
        "negativeWordCount": neg_count
    }

def extract_regex_entities(text: str) -> dict:
    lower = text.lower()
    
    # 1. Order ID (e.g., ORD-123456 or ORD-123456-1)
    order_match = re.search(r"\b(ord-\d{5,6}(-\d+)?)\b", lower)
    order_id = order_match.group(1).upper() if order_match else None

    # 2. Email
    email_match = re.search(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", lower)
    email = email_match.group(1) if email_match else None

    # 3. Phone Number
    phone_match = re.search(r"\b(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b", text)
    phone = phone_match.group(1) if phone_match else None

    # 4. Error Code
    error_match = re.search(r"\b(err_[a-zA-Z0-9_]+)\b", text)
    error_code = error_match.group(1).upper() if error_match else None

    # 5. Money Amount (e.g., $199.99 or 199.99)
    money_match = re.search(r"(\$\s?\d+(\.\d{2})?)", text)
    money = money_match.group(1) if money_match else None

    return {
        "orderId": order_id,
        "email": email,
        "phone": phone,
        "errorCode": error_code,
        "money": money
    }

def extract_named_entities(text: str) -> dict:
    lower = text.lower()
    
    city = None
    for c in CITIES_POOL:
        if c in lower:
            city = c.title()
            break

    product_name = None
    for p in PRODUCTS_POOL:
        if p in lower:
            product_name = p.title()
            break

    device = None
    for d in DEVICES_POOL:
        if d in lower:
            device = d.title()
            break

    customer_name = None
    for n in NAMES_POOL:
        if n in lower:
            customer_name = n.title()
            break

    return {
        "city": city,
        "productName": product_name,
        "device": device,
        "customerName": customer_name
    }

def extract_keywords(text: str) -> list:
    cleaned = clean_text(text)
    words = cleaned.split()
    # Exclude basic stopwords
    from preprocessing import STOPWORDS
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 3]
    # Deduplicate and return top 8
    seen = set()
    uniq = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq[:8]

def run_nlp_pipeline(text: str) -> dict:
    sentiment = analyze_sentiment(text)
    regex_entities = extract_regex_entities(text)
    named_entities = extract_named_entities(text)
    keywords = extract_keywords(text)

    # Reconstruct tokens
    cleaned = clean_text(text)
    tokens = cleaned.split()

    return {
        "query": text,
        "tokensCount": len(tokens),
        "sentiment": sentiment,
        "regexEntities": regex_entities,
        "entities": named_entities,
        "keywords": keywords
    }

if __name__ == "__main__":
    q = "Help! My Secure Router Pro is crashing with ERR_DB_TIMEOUT_504 in Austin. My order is ORD-88172 and refund is $299."
    print(run_nlp_pipeline(q))
