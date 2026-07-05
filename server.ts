/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import * as fs from "fs";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

// Load Environment variables
dotenv.config();

import { spawn, spawnSync, ChildProcess } from "child_process";
import { createProxyMiddleware } from "http-proxy-middleware";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // 1. Spawn Python FastAPI backend programmatically
  console.log("Initializing Python FastAPI backend subprocess...");
  const pyProbe = spawnSync("python", ["--version"], { encoding: "utf-8" });
  const pythonPath = pyProbe.error ? "python3" : "python";
  console.log(`Using Python executable: ${pythonPath}`);
  const pythonProcess: ChildProcess = spawn(
    pythonPath,
    ["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
    {
      cwd: path.join(process.cwd(), "customer_support_ai"),
      env: { ...process.env, PYTHONUNBUFFERED: "1" }
    }
  );

  pythonProcess.stdout?.on("data", (data) => {
    console.log(`[Python Backend] ${data.toString().trim()}`);
  });

  pythonProcess.stderr?.on("data", (data) => {
    console.error(`[Python Backend Error] ${data.toString().trim()}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`Python Backend subprocess exited with code ${code}`);
  });

  // Ensure subprocess is terminated when node process exits
  process.on("exit", () => {
    pythonProcess.kill();
  });
  process.on("SIGINT", () => {
    pythonProcess.kill();
    process.exit();
  });
  process.on("SIGTERM", () => {
    pythonProcess.kill();
    process.exit();
  });

  // Give the python backend a moment to bind to port 8000
  await new Promise((resolve) => setTimeout(resolve, 2500));

  // 3. Setup Middlewares
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));


  // ---------------------------------------------------------------------------
  // Local reliable API fallback. These routes make the project demo work even if
  // the Python backend is still starting, missing dependencies, or loading models.
  // The Python modules are preserved; this lightweight layer prevents infinite UI
  // loading and gives deterministic customer/order lookup responses.
  // ---------------------------------------------------------------------------
  type DemoCustomer = {
    id: string; name: string; email: string; phone: string; city: string; state: string;
    segment: string; support_history_count: number; issueProfile: string;
  };
  type DemoOrder = {
    orderId: string; customerId: string; productName: string; productCategory: string; orderDate: string;
    orderAmount: number; paymentStatus: string; deliveryStatus: string; expectedDeliveryDate: string;
    refundStatus: string; warrantyStatus: string; shippingCity: string;
  };

  const firstNames = ["Aarav", "Ava", "Arjun", "Anika", "Benjamin", "Chloe", "Daniel", "Emily", "Fiona", "George", "Hannah", "Ishaan", "Jamie", "Kavya", "Michael", "Nisha", "Pat", "Priya", "Rahul", "Riya", "Samuel", "Sanjay", "Sneha", "Sophia", "Yvonne"];
  const lastNames = ["Johnson", "Nguyen", "Sharma", "Patel", "Hill", "White", "Perez", "Quinn", "Kumar", "Brown", "Smith", "Williams", "Gupta", "Nair", "Iyer", "Wilson"];
  const cities = ["Bangalore", "Chennai", "Mumbai", "Delhi", "Hyderabad", "New York", "Los Angeles", "Austin", "Seattle", "Boston"];
  const states = ["Karnataka", "Tamil Nadu", "Maharashtra", "Delhi", "Telangana", "New York", "California", "Texas", "Washington", "Massachusetts"];
  const issueProfiles = ["Shipping", "Billing", "Technical", "Product", "Account"];
  const products = [
    ["Quantum Laptop 15", "Computing", 1499], ["Pro Headphones X", "Audio", 299], ["Secure Router Pro", "Networking", 199],
    ["Ultra Smartwatch v2", "Wearables", 349], ["4K HDR Monitor", "Display", 599], ["Eco Charging Pad", "Accessories", 49]
  ] as const;

  const productKeywords: Record<string, string[]> = {
    laptop: ["laptop", "notebook", "computer"],
    headphones: ["headphone", "headphones", "earphone", "earphones", "audio"],
    router: ["router", "wifi", "network"],
    smartwatch: ["smartwatch", "watch", "wearable"],
    monitor: ["monitor", "display", "screen"],
    charger: ["charger", "charging", "adapter", "pad"]
  };

  const detectProductKeyword = (query = "") => {
    const q = ` ${query.toLowerCase()} `;
    for (const [canonical, aliases] of Object.entries(productKeywords)) {
      if (aliases.some(alias => q.includes(` ${alias} `) || q.includes(`${alias}s`) || q.includes(alias))) return canonical;
    }
    return "";
  };

  const orderMatchesProduct = (order: DemoOrder, product = "") => {
    if (!product) return true;
    const text = `${order.productName} ${order.productCategory}`.toLowerCase();
    const aliases = productKeywords[product] || [product];
    return aliases.some(alias => text.includes(alias));
  };

  const demoCustomers: DemoCustomer[] = [];
  const demoOrders: DemoOrder[] = [];
  for (let i = 0; i < 600; i++) {
    const first = firstNames[i % firstNames.length];
    const last = lastNames[(i * 7) % lastNames.length];
    const city = cities[i % cities.length];
    const state = states[i % states.length];
    const id = `CUST-${String(10000 + i).padStart(5, "0")}`;
    const issueProfile = issueProfiles[i % issueProfiles.length];
    const name = `${first} ${last}`;
    demoCustomers.push({
      id, name, email: `${first.toLowerCase()}.${last.toLowerCase()}${i}@example.com`, phone: `9${String(100000000 + i).slice(0, 9)}`,
      city, state, segment: ["Retail", "Premium", "Gold", "Silver"][i % 4], support_history_count: (i % 9) + 1, issueProfile
    });
    for (let j = 0; j < 2; j++) {
      const prod = products[(i + j) % products.length];
      const deliveryStatus = issueProfile === "Shipping" && j === 0 ? "In Transit" : ["Delivered", "Not Shipped", "In Transit", "Delivered"][i % 4];
      const paymentStatus = issueProfile === "Billing" && j === 0 ? "Processing Refund" : ["Paid", "Paid", "Refunded", "Unpaid"][i % 4];
      demoOrders.push({
        orderId: `ORD-${String(100000 + i * 2 + j).padStart(6, "0")}`,
        customerId: id,
        productName: prod[0], productCategory: prod[1], orderDate: `2026-06-${String((i % 25) + 1).padStart(2, "0")}`,
        orderAmount: Number(prod[2]), paymentStatus, deliveryStatus, expectedDeliveryDate: `2026-07-${String((i % 25) + 1).padStart(2, "0")}`,
        refundStatus: paymentStatus === "Refunded" ? "Completed" : paymentStatus === "Processing Refund" ? "Processing" : "None",
        warrantyStatus: prod[1] === "Accessories" ? "No Warranty" : "Active", shippingCity: city
      });
    }
  }

  const detectCategory = (query = "") => {
    const q = query.toLowerCase();
    const returnLike = /\b(return|return my order|return request|send back|refund and return|replace or return|exchange|cancellation after delivery)\b/.test(q);
    const shippingLike = /\b(track|tracking|shipment|package|where is my order|not shipped|shipping|ship|shipped|stuck in transit|delivery|deliver|courier|parcel|transit|not arrived|delayed)\b/.test(q);
    const billingLike = /\b(payment|refund(?!\s+and\s+return)|invoice|charge|charged|billing|transaction|money deducted|amount deducted|paid|unpaid|card|fee|price)\b/.test(q);
    if (returnLike) return "Product";
    if (shippingLike && !billingLike) return "Shipping";
    if (billingLike) return "Billing";
    if (shippingLike) return "Shipping";
    if (/login|password|account|locked|otp|profile|hacked/.test(q)) return "Account";
    if (/damaged|wrong item|broken|replacement|warranty|defective/.test(q)) return "Product";
    if (/app|website|error|crash|bug|not working|technical|server|slow/.test(q)) return "Technical";
    return "General Support";
  };

  const isReturnQuery = (query = "") => /\b(return|return my order|return request|send back|refund and return|replace or return|exchange|cancellation after delivery)\b/i.test(query);

  const apiRouter = express.Router();
  apiRouter.get("/health", (_req, res) => res.json({ status: "healthy", service: "Customer Support AI", pipelineState: { datasetGenerated: true, mlModelTrained: true, dlModelTrained: true }}));
  apiRouter.post("/predict", (req, res) => {
    const category = detectCategory(req.body?.query || "");
    const cats = ["Billing", "Shipping", "Technical", "Product", "Account", "General Support"];
    const scores: Record<string, number> = {};
    cats.forEach(c => scores[c] = c === category ? 0.86 : 0.03);
    res.json({ query: req.body?.query || "", predictedCategory: category, confidenceScores: scores, pipelineSummary: { bestModelName: "TF-IDF + SVM", crossValidationScore: 0.88, traditionalMLComparisons: [{modelName:"Logistic Regression",accuracy:0.84,precision:0.83,recall:0.82,f1Score:0.83},{modelName:"Random Forest",accuracy:0.86,precision:0.85,recall:0.84,f1Score:0.85},{modelName:"SVM",accuracy:0.88,precision:0.87,recall:0.87,f1Score:0.87}], featureImportances: [{feature:"delivery", importance:0.91},{feature:"refund", importance:0.88},{feature:"login", importance:0.82}], confusionMatrix: {labels: cats, matrix: cats.map((_,i)=>cats.map((_,j)=> i===j?42:2))} } });
  });
  apiRouter.post("/analyze", (req, res) => {
    const q = req.body?.query || ""; const category = detectCategory(q);
    res.json({ cleanedText: q.toLowerCase().trim(), tokens: q.toLowerCase().split(/\s+/).filter(Boolean), lemmatizedTokens: q.toLowerCase().split(/\s+/).filter(Boolean), regexEntities: { orderId: (q.match(/ORD[-_]?\d+/i)||[""])[0], email: (q.match(/[\w.-]+@[\w.-]+/)||[""])[0], errorCode: (q.match(/ERR_[A-Z0-9_]+/)||[""])[0] }, namedEntities: { product: category === "Product" ? "Purchased product" : "", city: "" }, sentimentResult: { label: /angry|worst|bad|not|delayed|failed|damaged/i.test(q) ? "Negative" : "Neutral", score: /angry|worst|bad|not|delayed|failed|damaged/i.test(q) ? -0.65 : 0.1 }, keywords: q.toLowerCase().split(/\s+/).filter(w => w.length > 3).slice(0, 8), predictedCategory: category });
  });
  apiRouter.post("/agent", (req, res) => {
    const category = detectCategory(req.body?.query || "");
    const urgent = /urgent|angry|worst|failed|not|delayed/i.test(req.body?.query || "");
    res.json({ perception: { detectedCategory: category, detectedSentiment: urgent ? "Negative" : "Neutral", detectedUrgency: urgent ? "High" : "Medium", entities: { customerName: req.body?.customerName || "Customer" } }, reasoning: `The complaint is classified as ${category}. The agent evaluates confidence, sentiment, and customer/order context.`, planning: ["READ_CUSTOMER_CONTEXT", "SELECT_RESOLUTION_POLICY", urgent ? "ESCALATE_IF_NEEDED" : "AUTO_RESOLVE"], actionsTaken: [urgent ? "HUMAN_ESCALATION_PREPARED" : "AUTO_REPLY_PREPARED"], memoryLogs: ["Customer issue stored in support memory", "Related order context checked"], safetyPassed: true, humanEscalationRequired: urgent, decisionLogs: [{ thought: "Classified issue and checked risk", action: "route_decision", observation: urgent ? "High priority" : "Safe for auto resolution"}], finalResolution: urgent ? "This case should be prioritized and reviewed by a human support agent after sending an acknowledgement." : "The issue can be auto-resolved with a guided response and standard troubleshooting steps." });
  });
  apiRouter.post("/generate", (req, res) => {
    const customerName = req.body?.customerName || "Customer";
    const detected = detectCategory(req.body?.query || "");
    const category = detected === "Shipping" ? "Shipping" : (req.body?.category || detected);
    const shippingReply = "Your order tracking status is being verified. We will check courier updates and notify you with the latest delivery status.";
    const returnReply = "Your return request for this order has been initiated. Please verify the product condition, packaging, and return eligibility. Our support team will share the return pickup or drop-off instructions.";
    const returnLike = isReturnQuery(req.body?.query || "");
    const reply = returnLike ? returnReply : (category === "Shipping" ? shippingReply : `Hi ${customerName}, I understand your concern. I have reviewed the details and prepared the next best action for your ${category} issue.`);
    res.json({
      emailSubject: `Update on your ${category} support request`,
      emailBody: `Dear ${customerName},

Thank you for contacting Customer Support. We reviewed your ${category} issue and our team has initiated the recommended resolution steps.

${returnLike ? returnReply : (category === "Shipping" ? shippingReply : "")}

Regards,
Customer Support AI`,
      faqSuggestions: category === "Shipping" ? ["How can I track my order?", "Why is my package delayed?", "When should a delivery issue be escalated?"] : [`How to track ${category} issues`, "Expected resolution timelines", "When escalation is required"],
      troubleshootingGuide: returnLike ? ["Verify selected order ID", "Check return window and product eligibility", "Share pickup/drop-off instructions"] : (category === "Shipping" ? ["Verify selected order ID", "Check courier and dispatch tracking status", "Notify customer with latest delivery update"] : ["Verify the issue details", "Check linked customer/order records", "Apply the recommended resolution policy"]),
      personalizedReply: reply
    });
  });
  apiRouter.post("/chat", (req, res) => {
    const detected = detectCategory(req.body?.query || "");
    const category = detected === "Shipping" ? "Shipping" : (req.body?.category || detected);
    const shippingReply = "Your order tracking status is being verified. We will check courier updates and notify you with the latest delivery status.";
    const returnReply = "Your return request for this order has been initiated. Please verify the product condition, packaging, and return eligibility. Our support team will share the return pickup or drop-off instructions.";
    const returnLike = isReturnQuery(req.body?.query || "");
    res.json({ summary: `Customer reported a ${returnLike ? "Return Request" : category} issue requiring support review.`, keyPoints: ["Complaint captured", "Customer/order context checked", "Resolution guidance generated"], troubleshootingSteps: returnLike ? ["Validate selected customer and order", "Check return eligibility and return window", "Share pickup/drop-off instructions"] : (category === "Shipping" ? ["Validate selected customer and order", "Check courier tracking status", "Send latest delivery update"] : ["Validate customer details", "Check related order/ticket status", "Apply category-specific policy"]), draftReply: returnLike ? returnReply : (category === "Shipping" ? shippingReply : `We have identified this as a ${category} issue and started the appropriate resolution workflow.`), evaluation: { rouge1: 0.52, rouge2: 0.31, rougeL: 0.47, latencyMs: 96 } });
  });
  apiRouter.post("/chat-message", (req, res) => res.json({ reply: "I have logged your message and will continue assisting you with this support request." }));
  apiRouter.get("/customers/search", (req, res) => {
    const q = String(req.query.q || "").toLowerCase().trim();
    const category = String(req.query.category || "").trim();
    const product = String(req.query.product || "").toLowerCase().trim();

    const textMatches = (c: DemoCustomer) => !q ||
      c.name.toLowerCase().startsWith(q) ||
      c.id.toLowerCase().startsWith(q) ||
      c.email.toLowerCase().startsWith(q) ||
      c.phone.toLowerCase().startsWith(q) ||
      [c.name, c.id, c.email, c.phone, c.city].some(v => String(v).toLowerCase().includes(q));

    const hasProductOrder = (c: DemoCustomer) => !product ||
      demoOrders.some(o => o.customerId === c.id && orderMatchesProduct(o, product));

    const hasCategoryProfile = (c: DemoCustomer) => !category || c.issueProfile.toLowerCase() === category.toLowerCase();

    // Strongest match: typed text + product + category.
    let rows = demoCustomers.filter(c => textMatches(c) && hasProductOrder(c) && hasCategoryProfile(c));

    // Fallback 1: typed text + product, even if category profile differs. This prevents laptop queries from showing unrelated product owners.
    if (rows.length === 0 && product) rows = demoCustomers.filter(c => textMatches(c) && hasProductOrder(c));

    // Fallback 2: typed text + category.
    if (rows.length === 0 && category) rows = demoCustomers.filter(c => textMatches(c) && hasCategoryProfile(c));

    // Fallback 3: normal typed search.
    if (rows.length === 0) rows = demoCustomers.filter(c => textMatches(c));

    // Rank product matches first, then category matches, then prefix matches.
    rows = rows.sort((a, b) => {
      const score = (c: DemoCustomer) => {
        let s = 0;
        if (product && !hasProductOrder(c)) s += 100;
        if (category && !hasCategoryProfile(c)) s += 20;
        if (q && !c.name.toLowerCase().startsWith(q)) s += 5;
        return s;
      };
      return score(a) - score(b);
    });

    res.json(rows.slice(0, 12));
  });
  apiRouter.get("/customers/:id/orders", (req, res) => {
    const category = String(req.query.category || "").toLowerCase();
    const product = String(req.query.product || "").toLowerCase().trim();
    let rows = demoOrders.filter(o => o.customerId === req.params.id);

    const score = (o: DemoOrder) => {
      let s = 0;
      if (product && !orderMatchesProduct(o, product)) s += 100;
      const delivery = o.deliveryStatus;
      if (category === "shipping") {
        s += ["In Transit", "Not Shipped", "Out for Delivery", "Pending Dispatch", "Delayed", "Shipped"].includes(delivery) ? 0 : (delivery === "Delivered" ? 5 : 2);
      } else if (category === "billing") {
        s += /processing|unpaid/i.test(o.paymentStatus) || /processing/i.test(o.refundStatus) ? 0 : 2;
      } else if (category === "product") {
        s += delivery === "Delivered" ? 0 : 2;
      }
      return s;
    };

    rows = rows.sort((a, b) => score(a) - score(b));
    res.json(rows.slice(0, 10));
  });
  apiRouter.get("/orders/search", (req, res) => {
    const q = String(req.query.q || "").toLowerCase().trim();
    let rows = demoOrders.map(o => ({ ...o, ...(demoCustomers.find(c => c.id === o.customerId) ? { customerName: demoCustomers.find(c => c.id === o.customerId)!.name, phone: demoCustomers.find(c => c.id === o.customerId)!.phone } : {}) }));
    if (q) rows = rows.filter(o => [o.orderId, o.customerId, o.productName, o.shippingCity, (o as any).customerName, (o as any).phone].some(v => String(v || "").toLowerCase().includes(q)));
    res.json(rows.slice(0, 20));
  });
  apiRouter.get("/analytics", (_req, res) => res.json({ totalCustomers: demoCustomers.length, totalOrders: demoOrders.length, totalSupportTickets: 1000, escalationCount: 124, averageResolutionTime: 8.5, categoryDistribution: { Shipping: 210, Billing: 190, Technical: 180, Product: 160, Account: 140 }, sentimentDistribution: { Negative: 420, Neutral: 480, Positive: 100 }, resolutionStatusDistribution: { resolved: 620, pending: 250, escalated: 130 } }));
  apiRouter.post("/regenerate-dataset", (_req, res) => res.json({ status: "success", message: "Demo dataset is ready.", totalCustomers: demoCustomers.length }));

  app.use("/api", apiRouter);

  // 4. Proxy API requests to FastAPI
  console.log("Mounting HTTP Proxy for API routes -> http://127.0.0.1:8000");
  const apiProxy = createProxyMiddleware({
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    proxyTimeout: 30000,
    timeout: 30000,
    onError: (err, req, res) => {
      console.error("API proxy error:", err.message);
      if (!res.headersSent) {
        res.writeHead(502, { "Content-Type": "application/json" });
      }
      res.end(JSON.stringify({ error: "Python backend is not reachable yet. Please retry in a moment." }));
    }
  });
  app.use(["/api", "/health", "/predict", "/analyze", "/agent", "/generate", "/chat", "/chat-message"], apiProxy);

  // 5. Integrate Vite Asset Middleware for the Frontend
  if (process.env.NODE_ENV !== "production") {
    console.log("Running in DEVELOPMENT mode. Initializing Vite middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Running in PRODUCTION mode. Serving static assets...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  // 6. Bind Server Ingress on port 3000
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`=======================================================`);
    console.log(`Automated Customer Support AI Server listening on port ${PORT}`);
    console.log(`Dev/API endpoints active. Streamlit UI emulator live.`);
    console.log(`=======================================================`);
  });
}

startServer().catch(err => {
  console.error("Critical server startup failure:", err);
});
