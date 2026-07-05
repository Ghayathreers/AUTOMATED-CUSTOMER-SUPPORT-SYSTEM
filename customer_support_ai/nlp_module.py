# -*- coding: utf-8 -*-
"""
NLP Module for Automated Customer Support AI.
Implements sentiment analysis, Named Entity Recognition (NER), regex-based entity extraction,
and keyword extraction.
"""

import re
from preprocessing import STOPWORDS, lemmatize_token

# Sentiment lists
POSITIVE_WORDS = {
    "love", "great", "excellent", "perfect", "good", "happy", "pleased", "amazing", "awesome",
    "thank", "thanks", "appreciate", "resolved", "fantastic", "wonderful", "helpful", "speedy", "fast"
}

NEGATIVE_WORDS = {
    "hate", "terrible", "bad", "worst", "unhappy", "angry", "frustrated", "error", "fail", "failed",
    "failure", "crash", "crashed", "crashes", "freeze", "freezing", "bug", "timeout", "broken",
    "slow", "latency", "double", "charge", "refund", "delayed", "delay", "waiting", "dispute", "scam"
}

# Entity dictionaries
CITIES = {"austin", "boston", "chicago", "denver", "el paso", "fort worth", "gary", "houston",
          "indianapolis", "jacksonville", "kansas city", "los angeles", "miami", "nashville",
          "oakland", "phoenix", "queens", "raleigh", "seattle", "tampa", "urbana", "vancouver",
          "wichita", "xenia", "yonkers", "zion"}

PRODUCTS = {"pro headphones x", "quantum laptop 15", "secure router pro", "ultra smartwatch v2",
            "eco charging pad", "noise-cancelling bud s", "4k hdr monitor"}

DEVICES = {"iphone 14 pro", "samsung galaxy s23", "macbook pro m2", "dell xps 13", "ipad air",
           "windows desktop", "android tablet"}

NAMES = {"alex", "jordan", "taylor", "morgan", "sam", "casey", "jamie", "riley", "robin", "drew",
         "skyler", "cameron", "chris", "pat", "amelia", "benjamin", "chloe", "daniel", "emily"}

def analyze_sentiment(text: str) -> dict:
    lower = text.lower()
    words = re.sub(r'[^\w\s]', '', lower).split()
    
    pos_score = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_score = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    score = pos_score - neg_score
    
    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"
        
    return {
        "score": score,
        "label": label,
        "positiveWordCount": pos_score,
        "negativeWordCount": neg_score
    }

def extract_named_entities(text: str) -> dict:
    lower = text.lower()
    
    # Simple substring scan
    city = ""
    for c in CITIES:
        if c in lower:
            city = c.title()
            break
            
    product_name = ""
    for p in PRODUCTS:
        if p in lower:
            product_name = p.title()
            break
            
    device = ""
    for d in DEVICES:
        if d in lower:
            device = d.title()
            break
            
    customer_name = ""
    for n in NAMES:
        if n in lower:
            customer_name = n.title()
            break
            
    return {
        "city": city,
        "productName": product_name,
        "device": device,
        "customerName": customer_name
    }

def extract_regex_entities(text: str) -> dict:
    # 1. Order ID
    order_id_match = re.search(r'ord-\d{6}', text, re.IGNORECASE)
    order_id = order_id_match.group(0).upper() if order_id_match else ""
    
    # 2. Email Address
    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    email = email_match.group(0) if email_match else ""
    
    # 3. Phone Number
    phone_match = re.search(r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    phone = phone_match.group(0) if phone_match else ""
    
    # 4. Error Code
    error_match = re.search(r'err_[a-zA-Z0-9_]+', text, re.IGNORECASE)
    error_code = error_match.group(0).upper() if error_match else ""
    
    # 5. Money Amount
    money_match = re.search(r'\$\d+(?:\.\d{2})?', text)
    money = money_match.group(0) if money_match else ""
    
    return {
        "orderId": order_id,
        "email": email,
        "phone": phone,
        "errorCode": error_code,
        "money": money
    }

def extract_keywords(text: str) -> list:
    lower = text.lower()
    words = re.sub(r'[^\w\s]', '', lower).split()
    
    # Filter stopwords
    filtered = [lemmatize_token(w) for w in words if w not in STOPWORDS and len(w) > 2]
    
    # Calculate word frequency
    freq = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
        
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w[0] for w in sorted_words[:5]]

def run_nlp_pipeline(text: str) -> dict:
    sentiment = analyze_sentiment(text)
    entities = extract_named_entities(text)
    regex_entities = extract_regex_entities(text)
    keywords = extract_keywords(text)
    
    return {
        "sentiment": sentiment,
        "entities": entities,
        "regexEntities": regex_entities,
        "keywords": keywords
    }

if __name__ == "__main__":
    sample_text = "Hello Jordan, I see two identical charges of $299 on my card for order ORD-123456. My iPhone 14 Pro got error ERR_AUTH_FAILED_401 in Austin."
    print(run_nlp_pipeline(sample_text))
