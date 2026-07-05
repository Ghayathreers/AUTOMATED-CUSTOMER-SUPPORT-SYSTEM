# -*- coding: utf-8 -*-
"""
Preprocessing module for Automated Customer Support AI.
Provides clean_text, tokenize, DLTokenizer, pad_sequences, and custom lemmatization.
"""

import re

# Stopwords set
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
}

# Rule-based lemmatizer mapping for common customer support terms
LEMMA_DICTIONARY = {
    "billed": "bill", "billing": "bill", "bills": "bill", "charged": "charge", "charges": "charge", "charging": "charge",
    "refunded": "refund", "refunds": "refund", "refunding": "refund",
    "delivered": "deliver", "delivering": "deliver", "delivery": "deliver", "deliveries": "deliver",
    "shipped": "ship", "shipping": "ship", "ships": "ship", "packages": "package",
    "crashed": "crash", "crashes": "crash", "crashing": "crash",
    "errors": "error", "errored": "error", "devices": "device",
    "accounts": "account", "profiles": "profile", "passwords": "password",
    "purchased": "purchase", "purchases": "purchase", "purchasing": "purchase"
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Lowercase, remove punctuation and squeeze whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> list:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [t for t in cleaned.split(" ") if len(t) > 1]

def lemmatize_token(token: str) -> str:
    t = token.lower()
    if t in LEMMA_DICTIONARY:
        return LEMMA_DICTIONARY[t]
    
    # Basic stemming rules
    if t.endswith("s") and not t.endswith("ss") and not t.endswith("is"):
        return t[:-1]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    if t.endswith("ing") and len(t) > 5:
        return t[:-3]
    return t

class DLTokenizer:
    def __init__(self, max_features: int = 300):
        self.max_features = max_features
        self.vocab = {"<PAD>": 0, "<OOV>": 1}
        self.reverse_vocab = ["<PAD>", "<OOV>"]
        self.next_idx = 2

    def fit(self, texts: list):
        word_counts = {}
        for t in texts:
            tokens = re.sub(r'[^\w\s]', '', t.lower()).split()
            for tok in tokens:
                if tok:
                    word_counts[tok] = word_counts.get(tok, 0) + 1
                    
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_features - 2]
        
        for word, _ in sorted_words:
            if word not in self.vocab:
                self.vocab[word] = self.next_idx
                self.reverse_vocab.append(word)
                self.next_idx += 1

    def texts_to_sequences(self, texts: list) -> list:
        sequences = []
        for t in texts:
            tokens = re.sub(r'[^\w\s]', '', t.lower()).split()
            seq = [self.vocab.get(tok, 1) for tok in tokens if tok] # 1 is OOV
            sequences.append(seq)
        return sequences

def pad_sequences(sequences: list, max_len: int = 20) -> list:
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            p_seq = [0] * (max_len - len(seq)) + seq
            padded.append(p_seq)
    return padded
