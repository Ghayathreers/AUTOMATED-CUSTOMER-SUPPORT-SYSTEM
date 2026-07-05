import React from "react";
import { Search, ArrowRight } from "lucide-react";
import { motion } from "motion/react";

interface NlpTabProps {
  queryInput: string;
}

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

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .split(/\s+/)
    .filter(t => t.length > 1);
}

export function NlpTab({ queryInput }: NlpTabProps) {
  const tokens = tokenize(queryInput);

  return (
    <motion.div
      key="nlp"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.15 }}
      className="space-y-6"
    >
      <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl p-6 shadow-sm backdrop-blur-xs">
        <h2 className="text-lg font-bold text-zinc-100 mb-1 flex items-center gap-2 font-display">
          <Search className="text-indigo-500 h-5 w-5" />
          st.write("Level 3: Natural Language Processing")
        </h2>
        <p className="text-xs text-zinc-400 mb-6">
          Linguistic token analysis, lemmatized patterns, regex entity extractors, lexicon-based sentiment polarity, and named-entity extraction.
        </p>

        <div className="bg-zinc-950 border border-zinc-900 p-5 rounded-xl space-y-4">
          <div className="bg-zinc-900/20 border border-zinc-900 p-4 rounded-lg">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block mb-1.5 font-mono">Target string:</span>
            <p className="text-xs font-mono bg-zinc-950 text-zinc-300 p-3 rounded-lg border border-zinc-900 leading-relaxed">
              {queryInput}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Cleaning and Lemmas */}
            <div className="space-y-4">
              <div className="bg-zinc-900/20 border border-zinc-900 p-4 rounded-lg">
                <h4 className="font-bold text-xs text-zinc-400 mb-2 font-mono">Cleaned Syntax Token Matrix:</h4>
                <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto">
                  {tokens.length > 0 ? tokens.map((tok, idx) => (
                    <span key={idx} className="text-[10px] font-mono bg-zinc-900 border border-zinc-800 px-2 py-1 rounded text-zinc-400">
                      {tok}
                    </span>
                  )) : (
                    <span className="text-xs text-zinc-500 italic">No tokens found.</span>
                  )}
                </div>
              </div>

              <div className="bg-zinc-900/20 border border-zinc-900 p-4 rounded-lg">
                <h4 className="font-bold text-xs text-zinc-400 mb-2 font-mono">English Lemmas (Stemmed Words):</h4>
                <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto">
                  {tokens.length > 0 ? tokens.filter(t => !STOPWORDS.has(t)).map((tok, idx) => {
                    const lemma = LEMMA_DICTIONARY[tok] || tok;
                    return (
                      <span key={idx} className="text-[10px] font-mono bg-indigo-950/40 border border-indigo-900/40 px-2 py-1 rounded text-indigo-300 font-medium flex items-center gap-1">
                        {tok} <ArrowRight className="h-2.5 w-2.5" /> {lemma}
                      </span>
                    );
                  }) : (
                    <span className="text-xs text-zinc-500 italic">No lemmas found.</span>
                  )}
                </div>
              </div>
            </div>

            {/* Regex extracts */}
            <div className="bg-zinc-900/20 border border-zinc-900 p-4 rounded-lg">
              <h4 className="font-bold text-xs text-zinc-400 mb-3 font-mono">st.write("Regex Entity Extractor Matrix:")</h4>
              <div className="space-y-2 font-mono text-[10px]">
                <div className="flex justify-between border-b border-zinc-900 pb-2">
                  <span className="text-zinc-500">Order Reference Regex:</span>
                  <span className="text-indigo-400 font-bold">{queryInput.match(/ORD-?\d{6}/i)?.[0]?.toUpperCase() || "None"}</span>
                </div>
                <div className="flex justify-between border-b border-zinc-900 pb-2">
                  <span className="text-zinc-500">Product ID Regex:</span>
                  <span className="text-indigo-400 font-bold">{queryInput.match(/PROD-?\d{4}/i)?.[0]?.toUpperCase() || "None"}</span>
                </div>
                <div className="flex justify-between border-b border-zinc-900 pb-2">
                  <span className="text-zinc-500">Currency Money Regex:</span>
                  <span className="text-indigo-400 font-bold">{queryInput.match(/\$\d+(?:\.\d{2})?/)?.[0] || "None"}</span>
                </div>
                <div className="flex justify-between border-b border-zinc-900 pb-2">
                  <span className="text-zinc-500">Email Address Regex:</span>
                  <span className="text-indigo-400 font-bold truncate max-w-[200px]">{queryInput.match(/[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}/)?.[0] || "None"}</span>
                </div>
                <div className="flex justify-between border-b border-zinc-900 pb-2">
                  <span className="text-zinc-500">Phone Number Regex:</span>
                  <span className="text-indigo-400 font-bold">{queryInput.match(/\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/)?.[0] || "None"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Error Code Registry Regex:</span>
                  <span className="text-indigo-400 font-bold">{queryInput.match(/ERR_[A-Z_0-9]+|ERR-\d{3}/i)?.[0]?.toUpperCase() || "None"}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
