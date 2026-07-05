/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export enum QueryCategory {
  BILLING = "Billing",
  TECHNICAL = "Technical",
  SHIPPING = "Shipping",
  PRODUCT = "Product",
  ACCOUNT = "Account"
}

export enum Sentiment {
  POSITIVE = "Positive",
  NEUTRAL = "Neutral",
  NEGATIVE = "Negative"
}

export interface SupportQuery {
  id: string;
  query: string;
  category: QueryCategory;
  orderId: string;
  city: string;
  product: string;
  amount: number;
  device: string;
  errorCode: string;
  customerName: string;
  sentiment: Sentiment;
  timestamp: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rocAuc: number;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][]; // [actual][predicted]
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelComparison {
  modelName: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface MLPipelineResults {
  logisticRegression: ModelMetrics;
  randomForest: ModelMetrics;
  svm: ModelMetrics;
  confusionMatrix: ConfusionMatrix;
  featureImportance: FeatureImportance[];
  comparison: ModelComparison[];
  bestModelName: string;
  crossValidationScore: number;
}

export interface DLPipelineResults {
  mlpMetrics: ModelMetrics;
  lstmMetrics: ModelMetrics;
  bilstmMetrics: ModelMetrics;
  mlpLossCurve: number[];
  mlpAccCurve: number[];
  lstmLossCurve: number[];
  lstmAccCurve: number[];
  earlyStoppedEpoch: number;
  comparisonWithML: {
    bestMLAccuracy: number;
    bestDLAccuracy: number;
    improvement: number;
  };
}

export interface NLPResult {
  cleanedText: string;
  tokens: string[];
  stopwordsRemoved: string[];
  lemmatizedTokens: string[];
  sentiment: {
    label: Sentiment;
    score: number; // between -1 and 1
  };
  keywords: string[];
  entities: {
    customerName?: string;
    city?: string;
    device?: string;
    timestamp?: string;
    productName?: string;
  };
  regexEntities: {
    orderId?: string;
    productId?: string;
    money?: string;
    email?: string;
    phone?: string;
    errorCode?: string;
  };
}

export interface SLMResult {
  summary: string;
  conciseSteps: string[];
  draftReply: string;
  evaluation: {
    rouge1: number;
    rouge2: number;
    rougeL: number;
    latencyMs: number;
  };
}

export interface GenAIResult {
  personalizedEmail: string;
  faqList: { question: string; answer: string }[];
  troubleshootingGuide: string;
  simulatedConversation: { speaker: string; text: string }[];
}

export interface AgenticStep {
  thought: string;
  action: string;
  observation: string;
}

export interface AgenticResult {
  perception: {
    detectedCategory: QueryCategory;
    detectedSentiment: Sentiment;
    detectedUrgency: "Low" | "Medium" | "High";
    entities: Record<string, string>;
    confidence?: number;
    riskLevel?: string;
    riskFactors?: string[];
    detectedEmotion?: string;
    resolutionDecision?: string;
  };
  reasoning: string;
  planning: string[];
  actionsTaken: string[];
  memoryLogs: string[];
  safetyPassed: boolean;
  humanEscalationRequired: boolean;
  escalationReason?: string;
  decisionLogs: AgenticStep[];
  finalResolution: string;
}
