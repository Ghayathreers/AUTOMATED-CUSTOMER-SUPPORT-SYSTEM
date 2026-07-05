/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { AgenticResult, QueryCategory, Sentiment, AgenticStep } from "./types.js";
import { runNLPPipeline } from "./nlp_module.js";
import { runSLMTask } from "./slm_module.js";
import { TFIDFVectorizer, LogisticRegressionOVR, LogisticRegressionBinary } from "./ml_module.js";
import { MODELS_DIR } from "./config/index.js";
import * as fs from "fs";
import * as path from "path";

// Safety Guard blocking toxic inputs, prompt injections, or password phishes
export class SafetyGuard {
  static checkQuerySafety(text: string): { safe: boolean; reason?: string } {
    const lower = text.toLowerCase();
    
    // Prompt Injection checks
    if (lower.includes("ignore previous instructions") || lower.includes("system prompt") || lower.includes("bypass safety")) {
      return { safe: false, reason: "Potential Prompt Injection attempt detected." };
    }

    // Confidentiality/Password Phishing checks
    if (lower.includes("your password") || lower.includes("tell me your API key") || lower.includes("provide password")) {
      return { safe: false, reason: "Unauthorized credential query detected." };
    }

    // Toxicity / Vulgarity checks
    const blacklisted = ["abuse", "sh*t", "f*ck", "scumbag", "asshole", "idiot"];
    for (const word of blacklisted) {
      if (lower.includes(word)) {
        return { safe: false, reason: "Inappropriate or abusive language detected." };
      }
    }

    return { safe: true };
  }
}

// Memory logs container
export class AgenticMemory {
  history: string[] = [];

  addLog(entry: string) {
    this.history.push(`[${new Date().toISOString()}] ${entry}`);
  }

  getLogs(): string[] {
    return this.history;
  }
}

// Heuristic fallback category classifier
export function classifyQueryHeuristically(text: string): { category: QueryCategory; confidence: number } {
  const lower = text.toLowerCase();
  const scores: Record<QueryCategory, number> = {
    [QueryCategory.BILLING]: 0,
    [QueryCategory.TECHNICAL]: 0,
    [QueryCategory.SHIPPING]: 0,
    [QueryCategory.PRODUCT]: 0,
    [QueryCategory.ACCOUNT]: 0,
  };

  // Keywords with weights
  const keywords: Record<QueryCategory, { words: string[]; weight: number }[]> = {
    [QueryCategory.BILLING]: [
      { words: ["refund", "double charge", "overcharged", "charge twice", "charged twice", "reversal", "refunded", "dispute"], weight: 5 },
      { words: ["billing", "bill", "invoice", "payment", "stripe", "credit card", "bank", "charge", "charged", "fee"], weight: 3 },
      { words: ["cost", "price", "pay", "card", "dollar", "money", "$"], weight: 1.5 }
    ],
    [QueryCategory.TECHNICAL]: [
      { words: ["err_db_timeout_504", "err_auth_failed_401", "error code", "ipconfig /flushdns", "ipconfig"], weight: 5 },
      { words: ["crashed", "crashes", "freezing", "freeze", "bug", "timeout", "latency", "outage", "database error", "server error"], weight: 4 },
      { words: ["error", "fail", "failed", "failure", "broken", "slow", "connection", "app", "application", "client", "firmware"], weight: 2 }
    ],
    [QueryCategory.SHIPPING]: [
      { words: ["tracking number", "track link", "transit delay", "carrier scanning"], weight: 5 },
      { words: ["shipping", "shipped", "ship", "tracking", "tracker", "parcel", "delivery", "delivered", "transit", "hub", "logistics"], weight: 3 },
      { words: ["package", "delay", "delayed", "arrive", "arrival", "address", "route", "routing", "warehouse"], weight: 1.5 }
    ],
    [QueryCategory.PRODUCT]: [
      { words: ["secure router pro", "dual-band", "specs catalog", "dimensions", "hardware replacement"], weight: 5 },
      { words: ["warranty", "compatible", "compatibility", "router", "specifications", "specification", "specs", "hardware"], weight: 3 },
      { words: ["device", "size", "features", "pro", "catalog", "model", "replacement warranty"], weight: 1.5 }
    ],
    [QueryCategory.ACCOUNT]: [
      { words: ["delete account", "cancel subscription", "two-factor", "2fa", "password reset"], weight: 5 },
      { words: ["profile", "password", "reset", "login", "auth", "authentication", "credentials", "account", "security"], weight: 3 },
      { words: ["username", "register", "sign in", "sign up", "sign out", "logout", "log into", "secure login"], weight: 1.5 }
    ]
  };

  for (const cat of Object.keys(keywords) as QueryCategory[]) {
    for (const rule of keywords[cat]) {
      for (const word of rule.words) {
        if (lower.includes(word)) {
          scores[cat] += rule.weight;
        }
      }
    }
  }

  // Find category with highest score
  let bestCategory = QueryCategory.TECHNICAL; // default fallback
  let maxScore = 0;
  for (const cat of Object.keys(scores) as QueryCategory[]) {
    if (scores[cat] > maxScore) {
      maxScore = scores[cat];
      bestCategory = cat;
    }
  }

  // Calculate confidence score dynamically
  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  let confidence = 0.85; // baseline default
  if (totalScore > 0) {
    confidence = parseFloat((0.5 + (maxScore / totalScore) * 0.45).toFixed(2));
  } else {
    // default category if no keywords match
    bestCategory = QueryCategory.TECHNICAL;
    confidence = 0.50;
  }

  return { category: bestCategory, confidence };
}

// Get dynamic prediction from the trained ML model state
export function getMLPrediction(queryText: string): { category: QueryCategory; confidence: number } | null {
  try {
    const modelPath = path.join(MODELS_DIR, "best_model.json");
    if (!fs.existsSync(modelPath)) return null;

    const modelState = JSON.parse(fs.readFileSync(modelPath, "utf-8"));
    const vectorizer = new TFIDFVectorizer(modelState.vocabulary.length);
    vectorizer.vocabulary = modelState.vocabulary;
    vectorizer.idf = modelState.idf;
    vectorizer.vocabMap = new Map(modelState.vocabulary.map((term, idx) => [term, idx]));

    const lrModel = new LogisticRegressionOVR();
    lrModel.categories = modelState.categories;
    modelState.lrClassifiers.forEach((clf: any) => {
      const bClf = new LogisticRegressionBinary();
      bClf.weights = clf.weights;
      bClf.bias = clf.bias;
      lrModel.classifiers[clf.category] = bClf;
    });

    const vector = vectorizer.transform(queryText);
    const predictedCategory = lrModel.predict(vector) as QueryCategory;
    const probs = lrModel.predictProbs(vector);
    const confidence = parseFloat((probs[predictedCategory] || 0.85).toFixed(2));

    return { category: predictedCategory, confidence };
  } catch (err) {
    console.warn("ML Model online prediction failed, falling back to heuristics:", err);
    return null;
  }
}

// Risk factors detection suite
export function detectRiskFactors(text: string, category: QueryCategory, confidence: number): { riskLevel: string; riskFactors: string[] } {
  const lower = text.toLowerCase();
  const factors: string[] = [];

  // 1. Refund dispute
  if ((lower.includes("refund") || lower.includes("dispute") || lower.includes("chargeback") || lower.includes("reversal")) && (lower.includes("double") || lower.includes("unauthorized") || lower.includes("mistake") || lower.includes("wrong"))) {
    factors.push("Refund Dispute / Double Charge Scan");
  }

  // 2. Fraud
  if (lower.includes("fraud") || lower.includes("scam") || lower.includes("stolen") || lower.includes("unauthorized charge") || lower.includes("stolen card")) {
    factors.push("Fraud / Financial Security Warning");
  }

  // 3. Account hacked
  if (lower.includes("hacked") || lower.includes("stolen account") || lower.includes("compromised") || lower.includes("hacker") || lower.includes("breach") || lower.includes("compromise")) {
    factors.push("Account Compromised / Security Breach");
  }

  // 4. Legal complaint
  if (lower.includes("legal") || lower.includes("sue") || lower.includes("lawsuit") || lower.includes("lawyer") || lower.includes("court") || lower.includes("legal action")) {
    factors.push("Legal Complaint Escalation Hazard");
  }

  // 5. Payment failure
  if (lower.includes("failed payment") || lower.includes("payment failed") || lower.includes("declined") || lower.includes("charge failed") || lower.includes("card declined") || lower.includes("fail to pay")) {
    factors.push("Payment Processing Failure Block");
  }

  // 6. Repeated issue
  if (lower.includes("again") || lower.includes("second time") || lower.includes("multiple times") || lower.includes("repeated") || lower.includes("still not working") || lower.includes("still failing") || lower.includes("not solved")) {
    factors.push("Repeated Customer Friction Encountered");
  }

  // 7. Low confidence
  if (confidence < 0.65) {
    factors.push("Low Classifier Confidence Threshold Alert");
  }

  let riskLevel = "Low";
  if (factors.length >= 3) {
    riskLevel = "Critical Risk (Immediate SLA Escalation)";
  } else if (factors.length >= 1) {
    riskLevel = "Medium / High (Risk Flags Detected)";
  }

  return { riskLevel, riskFactors: factors };
}

// -------------------------------------------------------------
// CORE AGENTIC ENGINE
// -------------------------------------------------------------
export async function runAgenticAI(queryText: string, customerName: string): Promise<AgenticResult> {
  const memory = new AgenticMemory();
  const decisionLogs: AgenticStep[] = [];
  
  memory.addLog("Received query. Initializing Safety Guard check.");

  // 1. Perception Step - Safety Guard
  const safety = SafetyGuard.checkQuerySafety(queryText);
  if (!safety.safe) {
    decisionLogs.push({
      thought: "The query violates our safety protocols.",
      action: "REJECT_QUERY",
      observation: `Safety block triggered: ${safety.reason}`
    });
    memory.addLog(`Safety violation flagged: ${safety.reason}`);

    return {
      perception: {
        detectedCategory: QueryCategory.TECHNICAL,
        detectedSentiment: Sentiment.NEUTRAL,
        detectedUrgency: "High",
        entities: {},
        confidence: 1.0,
        riskLevel: "Critical Safety Risk",
        riskFactors: ["Safety Violation Flagged"],
        detectedEmotion: "Urgent",
        resolutionDecision: "Escalated to Human Agent"
      },
      reasoning: "Query was rejected immediately because it failed the security and toxicity check filters.",
      planning: ["REJECT", "LOG_VIOLATION"],
      actionsTaken: ["QUERY_REJECTED"],
      memoryLogs: memory.getLogs(),
      safetyPassed: false,
      humanEscalationRequired: true,
      escalationReason: safety.reason,
      decisionLogs,
      finalResolution: `Your request was blocked by our automated security systems: ${safety.reason}`
    };
  }

  decisionLogs.push({
    thought: "Input passed safety constraints. Proceeding with NLP Perception parsing.",
    action: "RUN_NLP_PIPELINE",
    observation: "NLP pipeline parsed syntactic tokens, sentiment, and regex entities."
  });
  memory.addLog("Input parsed successfully through NLP.");

  // 2. Parse-based Perception
  const nlp = runNLPPipeline(queryText);

  // Dynamic Intent Classification: Try ML model, fall back to Heuristic Keywords
  const mlPred = getMLPrediction(queryText);
  const heurPred = classifyQueryHeuristically(queryText);
  
  const detectedCategory = mlPred ? mlPred.category : heurPred.category;
  const confidence = mlPred ? mlPred.confidence : heurPred.confidence;

  const detectedSentiment = nlp.sentiment.label;
  const orderId = nlp.regexEntities.orderId || "";
  const moneyStr = nlp.regexEntities.money || "";
  const moneyValue = moneyStr ? parseFloat(moneyStr.replace("$", "")) : 0;
  const errorCode = nlp.regexEntities.errorCode || "";
  const emailStr = nlp.regexEntities.email || "";
  const phoneStr = nlp.regexEntities.phone || "";
  const locationCity = nlp.entities.city || "";
  const deviceStr = nlp.entities.device || "";
  const productNameStr = nlp.entities.productName || "";

  // Dynamic Sentiment / Emotion Detection
  let detectedEmotion = "Neutral";
  if (queryText.toUpperCase().includes("URGENT") || queryText.toUpperCase().includes("ASAP") || queryText.includes("!!!")) {
    detectedEmotion = "Urgent";
  } else if (detectedSentiment === Sentiment.NEGATIVE) {
    const lower = queryText.toLowerCase();
    if (lower.includes("hate") || lower.includes("angry") || lower.includes("scam") || lower.includes("terrible") || lower.includes("worst") || lower.includes("unacceptable")) {
      detectedEmotion = "Angry";
    } else {
      detectedEmotion = "Frustrated";
    }
  } else if (detectedSentiment === Sentiment.POSITIVE) {
    detectedEmotion = "Positive";
  }

  // Compute Urgency level
  let detectedUrgency: "Low" | "Medium" | "High" = "Low";
  if (detectedSentiment === Sentiment.NEGATIVE) detectedUrgency = "Medium";
  if (moneyValue > 200 || errorCode || detectedEmotion === "Urgent" || detectedEmotion === "Angry") {
    detectedUrgency = "High";
  }

  // Dynamic Risk Detection
  const risk = detectRiskFactors(queryText, detectedCategory, confidence);

  const entities: Record<string, string> = {
    customerName: customerName || nlp.entities.customerName || "Customer",
    city: locationCity,
    device: deviceStr,
    orderId,
    errorCode,
    moneyAmount: moneyStr,
    email: emailStr,
    phone: phoneStr,
    productName: productNameStr
  };

  decisionLogs.push({
    thought: `Urgency set to ${detectedUrgency}. Extracted category ${detectedCategory} (Confidence: ${confidence}).`,
    action: "EVALUATE_REASONING_CHAIN",
    observation: `Reasoned about billing limits, error diagnostics, and emotional state.`
  });
  memory.addLog(`Perceived Category: ${detectedCategory}, Sentiment: ${detectedSentiment}, Urgency: ${detectedUrgency}, Emotion: ${detectedEmotion}`);

  // 3. Reasoning and Decision-making
  let reasoning = "";
  let humanEscalationRequired = false;
  let escalationReason = "";
  const planning: string[] = [];

  // Determine Resolution Decision ("resolved automatically" | "needs more information" | "escalated to human agent")
  let resolutionDecision = "Resolved Automatically";

  // Check safety escalation thresholds first
  if (risk.riskLevel.includes("Critical")) {
    humanEscalationRequired = true;
    escalationReason = "Multiple high-risk security factors detected.";
    resolutionDecision = "Escalated to Human Agent";
  } else if (detectedCategory === QueryCategory.BILLING) {
    reasoning = `Customer requested financial adjustments for transaction ${orderId || 'N/A'}. `;
    if (moneyValue > 500) {
      reasoning += `The transaction amount of $${moneyValue} exceeds the $500 automated agent authorization limit. Policy mandates escalation.`;
      humanEscalationRequired = true;
      escalationReason = "Refund request amount exceeds the automated $500 threshold.";
      resolutionDecision = "Escalated to Human Agent";
    } else if (!orderId) {
      reasoning += "Missing essential transaction Order ID reference. Dynamic pipeline requires parameters.";
      resolutionDecision = "Needs More Information";
    } else {
      reasoning += `The refund amount of $${moneyValue || 'N/A'} is within our $500 automated refund limit. Safe to resolve.`;
    }
  } else if (detectedCategory === QueryCategory.TECHNICAL) {
    reasoning = `Customer reported application errors or crashes. `;
    if (errorCode === "ERR_DB_TIMEOUT_504" || errorCode === "ERR_AUTH_FAILED_401") {
      reasoning += `The system error code (${errorCode}) suggests severe server database issues or potential account compromise. Policy requires escalation.`;
      humanEscalationRequired = true;
      escalationReason = `System error code ${errorCode} requires high-level sysadmin inspection.`;
      resolutionDecision = "Escalated to Human Agent";
    } else {
      reasoning += `Error code ${errorCode || 'N/A'} is standard and can be cleared via local caching troubleshooting scripts.`;
    }
  } else if (detectedCategory === QueryCategory.SHIPPING) {
    reasoning = `Customer checking tracking details. `;
    if (!orderId) {
      reasoning += "Missing Order ID reference. Pipeline requires Order ID to access live carrier details.";
      resolutionDecision = "Needs More Information";
    } else {
      reasoning += `Retrieved routing details successfully for dispatch transit.`;
    }
  } else if (detectedCategory === QueryCategory.ACCOUNT) {
    reasoning = `Customer requesting profile settings or authentication access. `;
    if (!emailStr && !customerName) {
      reasoning += "No verified profile credentials or secure contact email found. Profile updates require verification parameters.";
      resolutionDecision = "Needs More Information";
    } else {
      reasoning += `Authorized security protocol check matches registration log profiles.`;
    }
  } else {
    reasoning = `General hardware and product specifications checklist matching. No threshold bounds exceeded.`;
  }

  // Force escalation on legal threats
  if (risk.riskFactors.includes("Legal Complaint Escalation Hazard") || risk.riskFactors.includes("Account Compromised / Security Breach")) {
    humanEscalationRequired = true;
    escalationReason = risk.riskFactors.includes("Legal Complaint Escalation Hazard")
      ? "Legal dispute threats detected in typed description."
      : "High-risk account hacking activity detected.";
    resolutionDecision = "Escalated to Human Agent";
  }

  // 4. Planning & Actions Formulation
  if (resolutionDecision === "Escalated to Human Agent") {
    planning.push("HALT_AUTOMATED_RESOLUTION", "GENERATE_SUPPORT_TICKET", "ESCALATE_TO_SENIOR_OPERATOR");
  } else if (resolutionDecision === "Needs More Information") {
    planning.push("PAUSE_TRANSACTIONS", "DRAFT_PARAMETER_REQUEST", "PROMPT_USER_FOR_MISSING_DETAILS");
  } else {
    planning.push("EXECUTE_SLM_RESOLUTION_SYNTHESIS", "UPDATE_CRM_DATABASE_RECORD", "SEND_EMAIL_CONFIRMATION");
  }

  decisionLogs.push({
    thought: `Reasoning complete. Resolution decision is [${resolutionDecision}]. Formulating action execution.`,
    action: resolutionDecision === "Escalated to Human Agent" ? "ESCALATE" : (resolutionDecision === "Needs More Information" ? "PROMPT_INFO" : "RESOLVE_AUTONOMOUSLY"),
    observation: resolutionDecision === "Escalated to Human Agent" 
      ? `Ticket successfully routed to Senior Operator. Reason: ${escalationReason}` 
      : (resolutionDecision === "Needs More Information" ? "Waiting for customer coordinates." : "Executed automated resolution tasks.")
  });

  const actionsTaken: string[] = [];
  let finalResolution = "";

  if (resolutionDecision === "Escalated to Human Agent") {
    actionsTaken.push("AUTOMATED_RESOLUTION_HALTED", "TICKET_CREATED_IN_CRM", "ROUTED_TO_HUMAN_QUEUE");
    memory.addLog(`Autonomous actions halted. Escalated to human: ${escalationReason}`);
    finalResolution = `I have analyzed your request and determined that it requires personal oversight from our senior support operations managers. I have prioritized your ticket and escalated it directly to an expert agent.\n\nEscalation Reason: ${escalationReason}\nTicket Reference: TKT-${Math.floor(Math.random() * 900000 + 100000)}`;
  } else if (resolutionDecision === "Needs More Information") {
    actionsTaken.push("POSTPONED_RESOLUTION", "SENT_CLARIFICATION_CHECK", "AWAITING_CUSTOMER_COORD");
    memory.addLog(`Autonomous processing postponed. Reason: ${reasoning}`);
    
    let missingField = "details";
    if (detectedCategory === QueryCategory.BILLING || detectedCategory === QueryCategory.SHIPPING) missingField = "Order ID (e.g., ORD-123456)";
    else if (detectedCategory === QueryCategory.ACCOUNT) missingField = "registered Email Address";
    
    finalResolution = `Thank you for reaching out. To assist you further, we require some additional details. Could you please provide your ${missingField}? This will allow us to query our database and finalize your request.`;
  } else {
    // Run SLM resolution
    const slm = await runSLMTask(queryText, detectedCategory);
    actionsTaken.push("EXECUTE_SLM_RESPONSE", "UPDATE_LEDGER", "EMAIL_DRAFTED");
    memory.addLog("Autonomous actions completed. Summary and reply generated via local SLM.");
    finalResolution = `${slm.draftReply}\n\nTroubleshooting steps proposed:\n${slm.conciseSteps.map((s, idx) => `${idx + 1}. ${s}`).join("\n")}`;
  }

  return {
    perception: {
      detectedCategory,
      detectedSentiment,
      detectedUrgency,
      entities,
      confidence,
      riskLevel: risk.riskLevel,
      riskFactors: risk.riskFactors,
      detectedEmotion,
      resolutionDecision
    },
    reasoning,
    planning,
    actionsTaken,
    memoryLogs: memory.getLogs(),
    safetyPassed: true,
    humanEscalationRequired,
    escalationReason: humanEscalationRequired ? escalationReason : undefined,
    decisionLogs,
    finalResolution
  };
}
