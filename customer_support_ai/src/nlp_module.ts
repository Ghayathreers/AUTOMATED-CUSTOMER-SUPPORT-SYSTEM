/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Sentiment, NLPResult, QueryCategory } from "./types.js";

// Stopwords set
const STOPWORDS = new Set([
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
]);

// Rule-based lemmatizer mapping for common customer support terms
const LEMMA_DICTIONARY: Record<string, string> = {
  "billed": "bill", "billing": "bill", "bills": "bill", "charged": "charge", "charges": "charge", "charging": "charge",
  "refunded": "refund", "refunds": "refund", "refunding": "refund",
  "delivered": "deliver", "delivering": "deliver", "delivery": "deliver", "deliveries": "deliver",
  "shipped": "ship", "shipping": "ship", "ships": "ship", "packages": "package",
  "crashed": "crash", "crashes": "crash", "crashing": "crash",
  "errors": "error", "errored": "error", "devices": "device",
  "accounts": "account", "profiles": "profile", "passwords": "password",
  "purchased": "purchase", "purchases": "purchase", "purchasing": "purchase"
};

// Lemmatize a single token
export function lemmatizeToken(token: string): string {
  const t = token.toLowerCase();
  if (LEMMA_DICTIONARY[t]) return LEMMA_DICTIONARY[t];
  
  // Basic plural / past tense stemming rules
  if (t.endsWith("s") && !t.endsWith("ss") && !t.endsWith("is")) return t.slice(0, -1);
  if (t.endsWith("ed") && t.length > 4) return t.slice(0, -2);
  if (t.endsWith("ing") && t.length > 5) return t.slice(0, -3);
  
  return t;
}

// Custom simple Lexicon-based Sentiment Analyzer
const SENTIMENT_LEXICON: Record<string, number> = {
  "refund": 0.2, "please": 0.3, "thanks": 0.5, "thank": 0.5, "good": 0.6, "great": 0.8, "awesome": 0.9, "happy": 0.8,
  "helpful": 0.6, "excellent": 0.9, "solved": 0.7, "love": 0.8, "perfect": 0.9,
  "issue": -0.3, "problem": -0.4, "error": -0.4, "fail": -0.5, "failed": -0.5, "failure": -0.6, "crash": -0.6,
  "crashes": -0.6, "broken": -0.5, "damaged": -0.6, "bad": -0.5, "wrong": -0.4, "incorrect": -0.4, "stuck": -0.4,
  "delayed": -0.3, "delay": -0.3, "outage": -0.5, "frozen": -0.4, "slow": -0.3, "hate": -0.7, "frustrated": -0.7,
  "annoyed": -0.6, "useless": -0.7, "worst": -0.8, "suspended": -0.5, "delete": -0.2
};

export function analyzeSentiment(text: string): { label: Sentiment; score: number } {
  const tokens = text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/);
  let score = 0;
  let matches = 0;

  tokens.forEach(tok => {
    if (SENTIMENT_LEXICON[tok] !== undefined) {
      score += SENTIMENT_LEXICON[tok];
      matches++;
    }
  });

  // Normalize score between -1 and 1
  let finalScore = matches > 0 ? score / matches : 0;
  
  // Boost negative if urgent/caps phrases exist
  if (text.includes("!!!") || text.includes("ASAP") || text.includes("URGENT")) {
    finalScore -= 0.25;
  }
  
  finalScore = Math.min(Math.max(finalScore, -1.0), 1.0);

  let label = Sentiment.NEUTRAL;
  if (finalScore > 0.15) label = Sentiment.POSITIVE;
  else if (finalScore < -0.15) label = Sentiment.NEGATIVE;

  return { label, score: parseFloat(finalScore.toFixed(2)) };
}

// Named Entity Recognition (dictionary + context matched)
const CITIES = ["new york", "san francisco", "chicago", "austin", "seattle", "miami", "denver", "boston", "los angeles", "atlanta"];
const DEVICES = ["iphone", "samsung", "macbook", "dell", "ipad", "windows", "android", "pc", "laptop", "tablet"];
const FIRST_NAMES = ["alex", "jordan", "taylor", "morgan", "sam", "casey", "jamie", "riley", "robin", "drew", "skyler", "cameron", "chris", "pat"];
const PRODUCTS = ["secure router pro", "router pro", "secure router", "smart camera", "smart thermostat", "power charger", "wireless mouse", "mechanical keyboard"];

export function extractNamedEntities(text: string): {
  customerName?: string;
  city?: string;
  device?: string;
  timestamp?: string;
  productName?: string;
} {
  const lowerText = text.toLowerCase();
  const res: { customerName?: string; city?: string; device?: string; timestamp?: string; productName?: string } = {};

  // Match locations
  for (const city of CITIES) {
    if (lowerText.includes(city)) {
      res.city = city.split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
      break;
    }
  }

  // Match devices
  for (const dev of DEVICES) {
    if (lowerText.includes(dev)) {
      if (dev === "pc") res.device = "Windows Desktop";
      else res.device = dev.charAt(0).toUpperCase() + dev.slice(1);
      break;
    }
  }

  // Match products
  for (const prod of PRODUCTS) {
    if (lowerText.includes(prod)) {
      res.productName = prod.split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
      break;
    }
  }

  // Match names
  for (const name of FIRST_NAMES) {
    if (lowerText.includes(name)) {
      res.customerName = name.charAt(0).toUpperCase() + name.slice(1);
      break;
    }
  }

  // Look for context indicators like "my customer name is Jordan Smith"
  const nameContextMatch = text.match(/customer name is ([\w\s]{2,15})(?:\.|\s|$)/i);
  if (nameContextMatch) {
    res.customerName = nameContextMatch[1].trim();
  }

  // Set current timestamp as fallback or extract Date
  res.timestamp = new Date().toISOString();

  return res;
}

// Keyword Extraction using frequency and token lengths
export function extractKeywords(text: string, count: number = 5): string[] {
  const tokens = text.toLowerCase()
    .replace(/[^\w\s]/g, "")
    .split(/\s+/)
    .filter(t => t.length > 3 && !STOPWORDS.has(t));

  const freqs: Record<string, number> = {};
  tokens.forEach(tok => {
    freqs[tok] = (freqs[tok] || 0) + 1;
  });

  return Object.entries(freqs)
    .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length)
    .slice(0, count)
    .map(([word]) => word);
}

// Regex Entity Extraction Suite
export function runRegexExtraction(text: string): {
  orderId?: string;
  productId?: string;
  money?: string;
  email?: string;
  phone?: string;
  errorCode?: string;
} {
  const res: {
    orderId?: string;
    productId?: string;
    money?: string;
    email?: string;
    phone?: string;
    errorCode?: string;
  } = {};

  // 1. Order ID (ORD-XXXXXX or ORDXXXXXX)
  const orderMatch = text.match(/ORD-?\d{6}/i);
  if (orderMatch) res.orderId = orderMatch[0].toUpperCase();

  // 2. Money amount ($XXXX or $XXX.XX)
  const moneyMatch = text.match(/\$\d+(?:\.\d{2})?/);
  if (moneyMatch) res.money = moneyMatch[0];

  // 3. Email Address
  const emailMatch = text.match(/[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}/);
  if (emailMatch) res.email = emailMatch[0].toLowerCase();

  // 4. Phone Number (standard USA and common patterns)
  const phoneMatch = text.match(/\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/);
  if (phoneMatch) res.phone = phoneMatch[0];

  // 5. Error Code (ERR_XXX_XXX or ERR-XXX)
  const errorMatch = text.match(/ERR_[A-Z_0-9]+|ERR-\d{3}/i);
  if (errorMatch) res.errorCode = errorMatch[0].toUpperCase();

  // 6. Product ID simulation / match
  const productMatch = text.match(/PROD-?\d{4}/i);
  if (productMatch) {
    res.productId = productMatch[0].toUpperCase();
  }

  return res;
}

// FULL PIPELINE CONSOLE EXECUTION
export function runNLPPipeline(text: string): NLPResult {
  const cleanedText = text.replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
  const tokens = cleanedText.split(" ").filter(t => t.length > 0);
  const stopwordsRemoved = tokens.filter(t => !STOPWORDS.has(t.toLowerCase()));
  const lemmatizedTokens = stopwordsRemoved.map(lemmatizeToken);
  const sentiment = analyzeSentiment(text);
  const keywords = extractKeywords(text, 6);
  const entities = extractNamedEntities(text);
  const regexEntities = runRegexExtraction(text);

  return {
    cleanedText,
    tokens,
    stopwordsRemoved,
    lemmatizedTokens,
    sentiment,
    keywords,
    entities,
    regexEntities
  };
}
