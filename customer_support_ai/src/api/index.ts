/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from "express";
import * as fs from "fs";
import * as path from "path";
import { DATA_DIR, MODELS_DIR } from "../config/index.js";
import { runMLPipeline, TFIDFVectorizer, LogisticRegressionOVR, LogisticRegressionBinary } from "../ml_module.js";
import { runDLPipeline } from "../dl_module.js";
import { runNLPPipeline } from "../nlp_module.js";
import { runSLMTask } from "../slm_module.js";
import { runGenAITasks } from "../genai_module.js";
import { runAgenticAI } from "../agentic_module.js";
import { QueryCategory, Sentiment } from "../types.js";

export const apiRouter = Router();

// Helper to load or train baseline ML models
function ensureModelsTrained() {
  const modelPath = path.join(MODELS_DIR, "best_model.json");
  if (!fs.existsSync(modelPath)) {
    console.log("No trained models found. Running ML and DL training pipelines on startup.");
    // Load dataset (ensure it exists)
    const jsonDatasetPath = path.join(DATA_DIR, "customer_support_queries.json");
    if (fs.existsSync(jsonDatasetPath)) {
      try {
        const data = JSON.parse(fs.readFileSync(jsonDatasetPath, "utf-8"));
        const mlResults = runMLPipeline(data, MODELS_DIR);
        runDLPipeline(data, mlResults.svm.accuracy, MODELS_DIR);
        console.log("ML & DL training pipelines complete on startup.");
      } catch (err) {
        console.error("Failed to train models on startup:", err);
      }
    }
  }
}

// Ensure models are trained on import
setTimeout(ensureModelsTrained, 1000);

// 1. GET /health
apiRouter.get("/health", (req, res) => {
  const datasetExists = fs.existsSync(path.join(DATA_DIR, "customer_support_queries.csv"));
  const modelExists = fs.existsSync(path.join(MODELS_DIR, "best_model.json"));
  const dlModelExists = fs.existsSync(path.join(MODELS_DIR, "dl_model.json"));

  res.json({
    status: "healthy",
    service: "Automated Customer Support and Service Resolution AI",
    timestamp: new Date().toISOString(),
    environment: {
      platform: "Node.js",
      framework: "Express (FastAPI emulator)",
      port: 3000
    },
    pipelineState: {
      datasetGenerated: datasetExists,
      mlModelTrained: modelExists,
      dlModelTrained: dlModelExists
    }
  });
});

// 2. POST /predict
apiRouter.post("/predict", (req, res) => {
  const { query } = req.body;
  if (!query || typeof query !== "string") {
    res.status(400).json({ error: "Missing required parameter: query" });
    return;
  }

  try {
    const modelPath = path.join(MODELS_DIR, "best_model.json");
    const resultsPath = path.join(MODELS_DIR, "ml_pipeline_results.json");

    if (!fs.existsSync(modelPath) || !fs.existsSync(resultsPath)) {
      res.status(503).json({ error: "ML Pipeline is currently training. Please try again shortly." });
      return;
    }

    const modelState = JSON.parse(fs.readFileSync(modelPath, "utf-8"));
    const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));

    // Set up vectorizer with trained vocabulary
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

    const vector = vectorizer.transform(query);
    const predictedCategory = lrModel.predict(vector);
    const confidenceScores = lrModel.predictProbs(vector);

    res.json({
      query,
      predictedCategory,
      confidenceScores,
      pipelineSummary: {
        bestModelName: results.bestModelName,
        crossValidationScore: results.crossValidationScore,
        traditionalMLComparisons: results.comparison,
        featureImportances: results.featureImportance.slice(0, 10),
        confusionMatrix: results.confusionMatrix
      }
    });
  } catch (err) {
    console.error("Predict endpoint error:", err);
    res.status(500).json({ error: "Internal Model Inference Failure" });
  }
});

// 3. POST /analyze (NLP Pipeline)
apiRouter.post("/analyze", (req, res) => {
  const { query } = req.body;
  if (!query || typeof query !== "string") {
    res.status(400).json({ error: "Missing required parameter: query" });
    return;
  }

  try {
    const nlpResult = runNLPPipeline(query);
    res.json(nlpResult);
  } catch (err) {
    console.error("Analyze endpoint error:", err);
    res.status(500).json({ error: "Internal NLP Analysis Failure" });
  }
});

// 4. POST /chat (SLM Pipeline)
apiRouter.post("/chat", async (req, res) => {
  const { query, category } = req.body;
  if (!query || typeof query !== "string") {
    res.status(400).json({ error: "Missing required parameter: query" });
    return;
  }

  const resolvedCategory = category as QueryCategory || QueryCategory.TECHNICAL;

  try {
    const slmResult = await runSLMTask(query, resolvedCategory);
    res.json(slmResult);
  } catch (err) {
    console.error("Chat endpoint error:", err);
    res.status(500).json({ error: "Internal SLM Inference Failure" });
  }
});

// 5. POST /generate (GenAI Pipeline)
apiRouter.post("/generate", async (req, res) => {
  const { query, category, customerName, sentiment, orderId } = req.body;
  if (!query || typeof query !== "string") {
    res.status(400).json({ error: "Missing required parameter: query" });
    return;
  }

  const cat = category as QueryCategory || QueryCategory.TECHNICAL;
  const name = customerName || "Customer";
  const sent = sentiment as Sentiment || Sentiment.NEUTRAL;

  try {
    const genAIResult = await runGenAITasks(query, cat, name, sent, orderId);
    res.json(genAIResult);
  } catch (err) {
    console.error("Generate endpoint error:", err);
    res.status(500).json({ error: "Internal Generative AI Failure" });
  }
});

// 6. POST /agent (Agentic Pipeline)
apiRouter.post("/agent", async (req, res) => {
  const { query, customerName } = req.body;
  if (!query || typeof query !== "string") {
    res.status(400).json({ error: "Missing required parameter: query" });
    return;
  }

  const name = customerName || "";

  try {
    const agentResult = await runAgenticAI(query, name);
    res.json(agentResult);
  } catch (err) {
    console.error("Agent endpoint error:", err);
    res.status(500).json({ error: "Internal Agentic Loop Failure" });
  }
});

// Dynamic Conversational Support Assistant
import { GoogleGenAI } from "@google/genai";

const getGenAIClient = (): GoogleGenAI => {
  return new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });
};

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function callWithRetry<T>(fn: () => Promise<T>, retries = 2, delayMs = 1000): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err: any) {
      attempt++;
      const errMsg = err?.message || "";
      const isUnavailable = errMsg.includes("503") || errMsg.includes("UNAVAILABLE") || err?.status === 503;
      if (isUnavailable && attempt <= retries) {
        console.log(`Gemini API returned 503 (Unavailable). Retrying attempt ${attempt}/${retries} in ${delayMs}ms...`);
        await delay(delayMs * Math.pow(2, attempt - 1));
        continue;
      }
      throw err;
    }
  }
}

export async function runInteractiveChat(
  history: { sender: "user" | "agent"; text: string }[],
  customerName: string,
  category: string,
  initialQuery: string
): Promise<string> {
  const name = customerName || "Customer";
  const cat = category || "General Support";

  if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== "MY_GEMINI_API_KEY" && process.env.GEMINI_API_KEY !== "") {
    try {
      const ai = getGenAIClient();
      const systemPrompt = `You are an empathetic, smart, and professional customer support assistant assisting ${name}.
The original customer ticket query was: "${initialQuery}" (Category: ${cat}).
Your task is to respond to the customer's messages in this ongoing chat session.
Keep your responses helpful, polite, concise (max 2-3 sentences), and highly relevant to the context. Do not break character. Do not repeat greeting messages. Keep the tone human-like and professional.`;

      const formattedContents = history.map(h => ({
        role: h.sender === "user" ? "user" : "model",
        parts: [{ text: h.text }]
      }));

      const response = await callWithRetry(() => ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: formattedContents,
        config: {
          systemInstruction: systemPrompt,
          temperature: 0.7
        }
      }));

      return response.text || "I am processing your request. Please let me know if you have any other details.";
    } catch (err: any) {
      console.log(`Gemini interactive chat failed gracefully, falling back to local rule engine. Error: ${err?.message || err}`);
    }
  }

  // Local Rule Engine fallback (sophisticated conversational system)
  const lastMessage = history[history.length - 1]?.text || "";
  const lowerMsg = lastMessage.toLowerCase();

  if (lowerMsg.includes("hello") || lowerMsg.includes("hi") || lowerMsg.includes("hey")) {
    return `Hello ${name}! How can I help you with your ${cat.toLowerCase()} query today?`;
  }
  if (lowerMsg.includes("refund") || lowerMsg.includes("charge") || lowerMsg.includes("money") || lowerMsg.includes("billing")) {
    return `I understand this billing item is important, ${name}. Our financial reversal pipelines are currently queued for your case. Refunds typically credit within 3 business days depending on your bank.`;
  }
  if (lowerMsg.includes("track") || lowerMsg.includes("ship") || lowerMsg.includes("package") || lowerMsg.includes("delay")) {
    return `I have locked the logistics scanner. The package is currently processing at our sorting depot and is prioritized for dispatch. I will update your account with live carrier updates.`;
  }
  if (lowerMsg.includes("password") || lowerMsg.includes("login") || lowerMsg.includes("account") || lowerMsg.includes("auth")) {
    return `For security, please ensure you use a strong unique password. We can trigger an encrypted profile password-reset link directly to your email on request.`;
  }
  if (lowerMsg.includes("error") || lowerMsg.includes("crash") || lowerMsg.includes("fail") || lowerMsg.includes("freeze")) {
    return `Our DevOps monitors are actively tracking this error log. Please try clearing your local cache or reloading the application to resolve any sync conflicts.`;
  }
  if (lowerMsg.includes("thank") || lowerMsg.includes("appreciate") || lowerMsg.includes("solved") || lowerMsg.includes("ok")) {
    return `You're very welcome, ${name}! I'm glad I could assist. Please let me know if there's anything else I can do for you.`;
  }

  return `I have logged your request: "${lastMessage}". Our service resolution engine is monitoring your ticket and we will ensure everything is settled. Is there any other detail you'd like to share?`;
}

// 7. POST /chat-message (Interactive Chat Service)
apiRouter.post("/chat-message", async (req, res) => {
  const { history, customerName, category, query } = req.body;
  if (!history || !Array.isArray(history)) {
    res.status(400).json({ error: "Missing or invalid parameter: history" });
    return;
  }

  try {
    const replyText = await runInteractiveChat(history, customerName, category, query);
    res.json({ reply: replyText });
  } catch (err) {
    console.error("Interactive chat message endpoint error:", err);
    res.status(500).json({ error: "Internal Chat Generation Failure" });
  }
});

// CSV Parsing Utility for Data Science Pipelines
function parseCSV(filePath: string): any[] {
  if (!fs.existsSync(filePath)) {
    console.warn("CSV file not found:", filePath);
    return [];
  }
  const content = fs.readFileSync(filePath, "utf-8");
  const lines = content.split(/\r?\n/).filter(line => line.trim() !== "");
  if (lines.length === 0) return [];

  // Parse header
  const headers = lines[0].split(",").map(h => h.trim().replace(/^"(.*)"$/, "$1"));
  const records = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    const row: string[] = [];
    let insideQuote = false;
    let currentVal = "";

    for (let charIdx = 0; charIdx < line.length; charIdx++) {
      const char = line[charIdx];
      if (char === '"') {
        insideQuote = !insideQuote;
      } else if (char === ',' && !insideQuote) {
        row.push(currentVal.trim());
        currentVal = "";
      } else {
        currentVal += char;
      }
    }
    row.push(currentVal.trim());

    const obj: any = {};
    headers.forEach((h, idx) => {
      let val = row[idx] || "";
      if (val.startsWith('"') && val.endsWith('"')) {
        val = val.substring(1, val.length - 1);
      }
      obj[h] = val;
    });
    records.push(obj);
  }

  return records;
}

let cachedCustomers: any[] | null = null;
let cachedOrders: any[] | null = null;
let cachedQueries: any[] | null = null;

function loadCustomers(): any[] {
  if (cachedCustomers) return cachedCustomers;
  const filePath = path.join(DATA_DIR, "customers.csv");
  cachedCustomers = parseCSV(filePath);
  return cachedCustomers;
}

function loadOrders(): any[] {
  if (cachedOrders) return cachedOrders;
  const filePath = path.join(DATA_DIR, "orders.csv");
  cachedOrders = parseCSV(filePath);
  return cachedOrders;
}

function loadQueries(): any[] {
  if (cachedQueries) return cachedQueries;
  const filePath = path.join(DATA_DIR, "support_queries.csv");
  cachedQueries = parseCSV(filePath);
  return cachedQueries;
}

// 8. GET /customers/search (Autosuggest for 10k Customers)
apiRouter.get("/customers/search", (req, res) => {
  const query = (req.query.q as string || "").trim().toLowerCase();
  const categoryFilter = (req.query.category as string || "").trim().toLowerCase();
  const customers = loadCustomers();
  const queries = loadQueries();
  
  let filtered = customers;

  if (categoryFilter !== "") {
    const matchingCustIds = new Set(
      queries
        .filter(q => q["Query Category"] && q["Query Category"].toLowerCase() === categoryFilter)
        .map(q => q["Customer ID"])
    );
    filtered = customers.filter(c => matchingCustIds.has(c["Customer ID"]));
  }
  
  if (query !== "") {
    filtered = filtered.filter(c => 
      c["Customer Name"] && c["Customer Name"].toLowerCase().includes(query)
    );
  }

  res.json(filtered.slice(0, 10).map(c => ({
    id: c["Customer ID"],
    name: c["Customer Name"],
    email: c["Email"],
    phone: c["Phone Number"],
    city: c["City"],
    state: c["State"],
    segment: c["Customer Segment"],
    support_history_count: parseInt(c["Support History Count"] || "0")
  })));
});

// 9. GET /customers/:id/orders (Retrieve orders associated with customer)
apiRouter.get("/customers/:id/orders", (req, res) => {
  const customerId = req.params.id;
  const orders = loadOrders();
  const filtered = orders.filter(o => o["Customer ID"] === customerId);

  res.json(filtered.map(o => ({
    orderId: o["Order ID"],
    customerId: o["Customer ID"],
    productName: o["Product Name"],
    productCategory: o["Product Category"],
    orderDate: o["Order Date"],
    orderAmount: parseFloat(o["Order Amount"] || "0"),
    paymentStatus: o["Payment Status"],
    deliveryStatus: o["Delivery Status"],
    expectedDeliveryDate: o["Expected Delivery Date"],
    refundStatus: o["Refund Status"],
    warrantyStatus: o["Warranty Status"]
  })));
});

// 9.5 GET /orders/search (Query purchase orders dynamically)
apiRouter.get("/orders/search", (req, res) => {
  const query = (req.query.q as string || "").trim().toLowerCase();
  const orders = loadOrders();
  const customers = loadCustomers();
  
  if (query === "") {
    res.json(orders.slice(0, 15).map(o => {
      const cust = customers.find(c => c["Customer ID"] === o["Customer ID"]);
      return {
        orderId: o["Order ID"],
        customerId: o["Customer ID"],
        productName: o["Product Name"],
        productCategory: o["Product Category"],
        orderDate: o["Order Date"],
        orderAmount: parseFloat(o["Order Amount"] || "0"),
        paymentStatus: o["Payment Status"],
        deliveryStatus: o["Delivery Status"],
        expectedDeliveryDate: o["Expected Delivery Date"],
        refundStatus: o["Refund Status"],
        warrantyStatus: o["Warranty Status"],
        shippingCity: cust ? cust["City"] : "Austin"
      };
    }));
    return;
  }

  const filtered = orders.filter(o => 
    (o["Order ID"] && o["Order ID"].toLowerCase().includes(query)) ||
    (o["Product Name"] && o["Product Name"].toLowerCase().includes(query)) ||
    (o["Customer ID"] && o["Customer ID"].toLowerCase().includes(query))
  );

  res.json(filtered.slice(0, 15).map(o => {
    const cust = customers.find(c => c["Customer ID"] === o["Customer ID"]);
    return {
      orderId: o["Order ID"],
      customerId: o["Customer ID"],
      productName: o["Product Name"],
      productCategory: o["Product Category"],
      orderDate: o["Order Date"],
      orderAmount: parseFloat(o["Order Amount"] || "0"),
      paymentStatus: o["Payment Status"],
      deliveryStatus: o["Delivery Status"],
      expectedDeliveryDate: o["Expected Delivery Date"],
      refundStatus: o["Refund Status"],
      warrantyStatus: o["Warranty Status"],
      shippingCity: cust ? cust["City"] : "Austin"
    };
  }));
});

// 10. GET /analytics (Dynamic Data Science Dashboard Stats)
apiRouter.get("/analytics", (req, res) => {
  try {
    const customers = loadCustomers();
    const orders = loadOrders();
    const queries = loadQueries();

    // Compute distributions
    const categoryDistribution: Record<string, number> = {};
    const sentimentDistribution: Record<string, number> = {};
    const resolutionStatusDistribution: Record<string, number> = {};
    let escalationCount = 0;

    queries.forEach(q => {
      const cat = q["Query Category"] || "General";
      categoryDistribution[cat] = (categoryDistribution[cat] || 0) + 1;

      const sent = q["Sentiment"] || "Neutral";
      sentimentDistribution[sent] = (sentimentDistribution[sent] || 0) + 1;

      const resStatus = q["Resolution Status"] || "new";
      resolutionStatusDistribution[resStatus] = (resolutionStatusDistribution[resStatus] || 0) + 1;

      if (q["Escalation Status"] === "Yes" || q["Resolution Status"] === "escalated") {
        escalationCount++;
      }
    });

    // Realistic Average Resolution Time computation based on Priority distribution
    let totalTime = 0;
    let resolvedCount = 0;
    queries.forEach(q => {
      const status = q["Resolution Status"];
      if (status === "resolved") {
        resolvedCount++;
        const p = q["Priority"];
        if (p === "urgent") totalTime += 2;
        else if (p === "high") totalTime += 6;
        else if (p === "medium") totalTime += 12;
        else totalTime += 24;
      }
    });
    const averageResolutionTime = resolvedCount > 0 ? parseFloat((totalTime / resolvedCount).toFixed(1)) : 14.5;

    res.json({
      totalCustomers: customers.length,
      totalOrders: orders.length,
      totalSupportTickets: queries.length,
      escalationCount,
      averageResolutionTime,
      categoryDistribution,
      sentimentDistribution,
      resolutionStatusDistribution
    });
  } catch (err: any) {
    console.error("Analytics fetch failed:", err);
    res.status(500).json({ error: "Failed to load analytics dashboard data" });
  }
});

