# -*- coding: utf-8 -*-
"""
Preprocessing and tokenization functions for Automated Customer Support AI.
"""

import re
from typing import List

# Basic English Stopwords
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Strip HTML-like tags
    text = re.sub(r"<[^>]*>", "", text)
    # Strip non-alphanumeric (keep spaces, apostrophes and dashes)
    text = re.sub(r"[^a-zA-Z0-9\s\'\-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    cleaned = clean_text(text)
    tokens = cleaned.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

class DLTokenizer:
    def __init__(self, num_words: int = 1000):
        self.num_words = num_words
        self.word_index = {"<PAD>": 0, "<OOV>": 1}
        self.index_word = {0: "<PAD>", 1: "<OOV>"}
        self.word_counts = {}

    def fit_on_texts(self, texts: List[str]):
        for text in texts:
            tokens = tokenize(text, remove_stopwords=False)
            for t in tokens:
                self.word_counts[t] = self.word_counts.get(t, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        # Select top num_words
        for idx, (word, _) in enumerate(sorted_words[:self.num_words - 2]):
            actual_idx = idx + 2
            self.word_index[word] = actual_idx
            self.index_word[actual_idx] = word

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for text in texts:
            tokens = tokenize(text, remove_stopwords=False)
            seq = []
            for t in tokens:
                seq.append(self.word_index.get(t, 1)) # fallback to <OOV>
            sequences.append(seq)
        return sequences

def pad_sequences(sequences: List[List[int]], maxlen: int = 20, padding: str = "post", truncating: str = "post") -> List[List[int]]:
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == "pre":
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]
        
        if len(seq) < maxlen:
            diff = maxlen - len(seq)
            if padding == "pre":
                seq = [0] * diff + seq
            else:
                seq = seq + [0] * diff
        padded.append(seq)
    return padded
