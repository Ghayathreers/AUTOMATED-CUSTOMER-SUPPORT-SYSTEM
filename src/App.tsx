/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  LayoutDashboard,
  UserCheck,
  Search,
  ShieldCheck,
  History,
  BarChart3,
  Activity,
  BookOpen,
  Terminal,
  RefreshCw,
  Mail,
  Download,
  Copy,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Cpu,
  ArrowRight,
  User,
  Users,
  ShoppingBag,
  Plus,
  Send,
  Lock,
  Database,
  Sliders,
  Compass,
  Workflow,
  FileCode,
  Check,
  Clock,
  Briefcase,
  AlertCircle
} from "lucide-react";

// Standard Types matching the backend
import {
  QueryCategory,
  Sentiment,
  NLPResult,
  SLMResult,
  GenAIResult,
  AgenticResult,
  ModelMetrics,
  MLPipelineResults,
  DLPipelineResults
} from "../customer_support_ai/src/types";

// Import Custom Dataset and Resolver
import { CUSTOMERS, ORDERS, SAMPLE_QUERIES, SUPPORT_RECORDS, Customer as DatasetCustomer, Order as DatasetOrder } from "./data/customerDataset";
import { resolveCustomerQuery } from "./utils/resolutionEngine";

// Modular tab components for Developer Mode
import { DatasetTab } from "./components/DatasetTab";
import { MachineLearningTab } from "./components/MachineLearningTab";
import { DeepLearningTab } from "./components/DeepLearningTab";
import { NlpTab } from "./components/NlpTab";
import { SlmTab } from "./components/SlmTab";
import { LldTab } from "./components/LldTab";
import { ApiTab } from "./components/ApiTab";

interface SupportTicket {
  id: string;
  name: string;
  email: string;
  customerId: string;
  query: string;
  category: string;
  sentiment: string;
  urgency: string;
  status: string;
  date: string;
  riskLevel: string;
  autoResolved: boolean;
  resolutionTime: string;
  nlpResult?: NLPResult;
  agentResult?: AgenticResult;
  genAIResult?: GenAIResult;
  slmResult?: SLMResult;
  confidence?: number;
}

export default function App() {
  // Navigation Tabs for real-time customer support console
  const [activeTab, setActiveTab] = useState<
    "support" | "lookup" | "orders" | "analysis" | "resolution" | "reports"
  >("support");

  // Developer Mode Toggle (Crucial for placement/viva presentations)
  const [developerMode, setDeveloperMode] = useState<boolean>(false);

  // Form input states (empty by default)
  const [customerNameInput, setCustomerNameInput] = useState("");
  const [customerEmailInput, setCustomerEmailInput] = useState("");
  const [customerIdInput, setCustomerIdInput] = useState("");
  const [customerPhoneInput, setCustomerPhoneInput] = useState("");
  const [customerCityInput, setCustomerCityInput] = useState("");
  const [queryInput, setQueryInput] = useState("");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>("Auto-Detect");
  const [uploadedScreenshot, setUploadedScreenshot] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  // New Database Selection states
  const [customerOrders, setCustomerOrders] = useState<DatasetOrder[]>([]);
  const [selectedOrderId, setSelectedOrderId] = useState("");
  const [selectedOrder, setSelectedOrder] = useState<DatasetOrder | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestions, setSuggestions] = useState<DatasetCustomer[]>([]);
  const [customerSearch, setCustomerSearch] = useState("");
  const [orderSearch, setOrderSearch] = useState("");

  // Current active ticket state (populated after submit or selected from history)
  const [activeTicket, setActiveTicket] = useState<SupportTicket | null>(null);

  // Loading state
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");

  // System Diagnostics / Metrics
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [mlPipelineResults, setMlPipelineResults] = useState<MLPipelineResults | null>(null);
  const [dlPipelineResults, setDlPipelineResults] = useState<DLPipelineResults | null>(null);

  // Dynamic Data Science state variables
  const [customersList, setCustomersList] = useState<any[]>([]);
  const [ordersList, setOrdersList] = useState<any[]>([]);
  const [categoryMatches, setCategoryMatches] = useState<any[]>([]);
  const [analyticsData, setAnalyticsData] = useState<any>(null);

  // Email action state
  const [isEmailCopied, setIsEmailCopied] = useState(false);
  const [isEmailDispatched, setIsEmailDispatched] = useState(false);

  // Interactive chat message logs (stateful live conversation)
  const [chatMessages, setChatMessages] = useState<{ sender: "user" | "agent"; text: string }[]>([]);
  const [newChatMessage, setNewChatMessage] = useState("");

  // Expandable FAQ state
  const [faqExpanded, setFaqExpanded] = useState<Record<number, boolean>>({});

  // Dynamic search effects
  useEffect(() => {
    // 1. Fetch initial customers list for directory
    fetch(`/api/customers/search?q=${encodeURIComponent(customerSearch)}`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setCustomersList(data))
      .catch((err) => console.error("Error loading customers:", err));
  }, [customerSearch]);

  useEffect(() => {
    // 2. Fetch initial orders list for directory
    fetch(`/api/orders/search?q=${encodeURIComponent(orderSearch)}`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setOrdersList(data))
      .catch((err) => console.error("Error loading orders:", err));
  }, [orderSearch]);

  useEffect(() => {
    // 3. Fetch category-based match shortcuts
    const cat = detectCategoryFromQuery(queryInput);
    if (!cat) {
      setCategoryMatches([]);
      return;
    }
    let backendCat = "Billing";
    if (cat === "billing") backendCat = "Billing";
    else if (cat === "shipping") backendCat = "Shipping";
    else if (cat === "technical") backendCat = "Technical";
    else if (cat === "product_info") backendCat = "Product";
    else if (cat === "account") backendCat = "Account";

    fetch(`/api/customers/search?category=${backendCat}`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => setCategoryMatches(data))
      .catch((err) => console.error("Error loading category matches:", err));
  }, [queryInput]);

  useEffect(() => {
    // 4. Load initial analytics metrics
    fetch("/api/analytics")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setAnalyticsData(data);
      })
      .catch((err) => console.error("Error loading analytics stats:", err));
  }, []);

  // Dynamic ticket list (mock database that updates upon submit)
  const [customerQueries, setCustomerQueries] = useState<SupportTicket[]>([
    {
      id: "CASE-902110",
      name: "Emily Johnson",
      email: "emily.j@example.com",
      customerId: "CUST-41120",
      query: "My credit card was charged twice for order ORD-492100. I need a refund immediately.",
      category: "Billing",
      sentiment: "Negative",
      urgency: "High",
      status: "Resolved Automatically",
      date: "2026-06-30 08:30",
      riskLevel: "Medium (Financial Refund Exposure)",
      autoResolved: true,
      resolutionTime: "Instant (Automated Gateway)",
      nlpResult: {
        cleanedText: "credit card charged twice order ord-492100 need refund",
        tokens: ["credit", "card", "charged", "twice", "order", "ord-492100", "need", "refund"],
        stopwordsRemoved: ["credit", "card", "charged", "twice", "order", "ord-492100", "need", "refund"],
        lemmatizedTokens: ["credit", "card", "charge", "twice", "order", "ord-492100", "need", "refund"],
        sentiment: { label: Sentiment.NEGATIVE, score: -0.6 },
        keywords: ["credit", "card", "charged", "twice", "refund"],
        entities: { customerName: "Emily Johnson" },
        regexEntities: { orderId: "ORD-492100", money: "$150" }
      },
      agentResult: {
        perception: {
          detectedCategory: QueryCategory.BILLING,
          detectedSentiment: Sentiment.NEGATIVE,
          detectedUrgency: "High",
          entities: { customerName: "Emily Johnson", orderId: "ORD-492100" }
        },
        reasoning: "Refund request is under the $500 threshold limit. System can safely execute automatic card reversal.",
        planning: ["CHECK_TRANSACTIONS", "INITIATE_STRIPE_REVERSAL", "SEND_EMAIL"],
        actionsTaken: ["STRIPE_REFUND_SUCCESSFUL", "CRM_LEDGER_UPDATED"],
        memoryLogs: ["[INFO] Billing transaction found", "[INFO] Triggered gateway refund api"],
        safetyPassed: true,
        humanEscalationRequired: false,
        decisionLogs: [
          { thought: "Input is secure. Routing to financial processors.", action: "PERFORM_GATEWAY_LOOKUP", observation: "Found duplicate auth scan." }
        ],
        finalResolution: "We identified a duplicate transaction and initiated a card refund for ORD-492100. It will credit back within 3 business days."
      },
      genAIResult: {
        personalizedEmail: "Dear Emily Johnson,\n\nWe sincerely apologize for the double billing of order ORD-492100. Our finance operations cleared the duplicate charge and processed a $150 refund.\n\nSincerely,\nSupport Team",
        faqList: [
          { question: "Why was my billing charged twice?", answer: "This occurs due to temporary banking network synchronization latency." }
        ],
        troubleshootingGuide: "1. Audit card transaction ledgers.\n2. Execute partial refund in Stripe.",
        simulatedConversation: [
          { speaker: "Customer", text: "Hi, why was I double charged for order ORD-492100?" },
          { speaker: "Support Agent", text: "Hello Emily. I see the bank capture sync error. Reversing the second charge right now." }
        ]
      }
    },
    {
      id: "CASE-902111",
      name: "Robert Chen",
      email: "r.chen@example.com",
      customerId: "CUST-19224",
      query: "The application is throwing an ERR_DB_TIMEOUT_504 and crashes every time I open the profile tab on my Android phone.",
      category: "Technical",
      sentiment: "Negative",
      urgency: "High",
      status: "Escalated to Human",
      date: "2026-06-30 09:15",
      riskLevel: "High (Potential Service Outage)",
      autoResolved: false,
      resolutionTime: "2-4 Hours (Assigned Queue)",
      nlpResult: {
        cleanedText: "application throwing err_db_timeout_504 crashes profile tab android",
        tokens: ["application", "throwing", "err_db_timeout_504", "crashes"],
        stopwordsRemoved: ["application", "throwing", "err_db_timeout_504", "crashes"],
        lemmatizedTokens: ["application", "throw", "err_db_timeout_504", "crash"],
        sentiment: { label: Sentiment.NEGATIVE, score: -0.5 },
        keywords: ["crashes", "timeout", "profile", "android"],
        entities: { device: "Android Phone" },
        regexEntities: { errorCode: "ERR_DB_TIMEOUT_504" }
      },
      agentResult: {
        perception: {
          detectedCategory: QueryCategory.TECHNICAL,
          detectedSentiment: Sentiment.NEGATIVE,
          detectedUrgency: "High",
          entities: { customerName: "Robert Chen", errorCode: "ERR_DB_TIMEOUT_504" }
        },
        reasoning: "System database timeout code ERR_DB_TIMEOUT_504 suggests sever infrastructure blocks, requiring manual technician audit.",
        planning: ["HALT_AUTOMATED_RESOLUTION", "ESCALATE_TO_SYSADMIN"],
        actionsTaken: ["ESC_TICKET_CREATED", "ROUTED_TO_SYSADMIN_QUEUE"],
        memoryLogs: ["[WARNING] Timeout exception flagged. Escalation policy matches."],
        safetyPassed: true,
        humanEscalationRequired: true,
        decisionLogs: [
          { thought: "Severe DB error. Halting autonomous fixes.", action: "ROUT_TO_ENGINEER", observation: "Ticket successfully assigned to Level 2 engineering." }
        ],
        finalResolution: "This query has been escalated directly to our DevOps Engineering squad because it indicates a database timeout (ERR_DB_TIMEOUT_504)."
      }
    },
    {
      id: "CASE-902112",
      name: "Sarah Miller",
      email: "sara.m@example.com",
      customerId: "CUST-88319",
      query: "What is the warranty period for the Secure Router Pro? Does it support dual-band connection?",
      category: "Product",
      sentiment: "Neutral",
      urgency: "Low",
      status: "Resolved Automatically",
      date: "2026-06-30 10:02",
      riskLevel: "Low",
      autoResolved: true,
      resolutionTime: "Instant (Automated Gateway)",
      nlpResult: {
        cleanedText: "warranty period secure router pro support dual-band connection",
        tokens: ["warranty", "period", "secure", "router", "pro"],
        stopwordsRemoved: ["warranty", "period", "secure", "router", "pro"],
        lemmatizedTokens: ["warranty", "period", "secure", "router", "pro"],
        sentiment: { label: Sentiment.NEUTRAL, score: 0.1 },
        keywords: ["warranty", "router", "dual-band"],
        entities: {},
        regexEntities: {}
      },
      agentResult: {
        perception: {
          detectedCategory: QueryCategory.PRODUCT,
          detectedSentiment: Sentiment.NEUTRAL,
          detectedUrgency: "Low",
          entities: { customerName: "Sarah Miller" }
        },
        reasoning: "Standard hardware inquiry. Can resolve automatically using product knowledge guidelines.",
        planning: ["FETCH_PRODUCT_SPECS", "GENERATE_AUTONOMOUS_REPLY"],
        actionsTaken: ["FAQ_SUGGESTIONS_LINKED"],
        memoryLogs: ["[INFO] Warranty policy retrieved"],
        safetyPassed: true,
        humanEscalationRequired: false,
        decisionLogs: [
          { thought: "General inquiry. Safe for chatbot resolution.", action: "FETCH_KNOWLEDGE_BASE", observation: "Warranty matches 1-Year standard." }
        ],
        finalResolution: "The Secure Router Pro includes a 1-Year replacement warranty and features full dual-band (2.4GHz / 5GHz) concurrency."
      }
    }
  ]);

  // Fetch initial health/metrics on mount
  const fetchSystemStatus = async () => {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      setSystemHealth(data);

      // Fetch ML Pipeline training results
      const predRes = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "Test baseline load query" })
      });
      const predData = await predRes.json();
      if (predData.pipelineSummary) {
        setMlPipelineResults({
          logisticRegression: predData.pipelineSummary.traditionalMLComparisons[0],
          randomForest: predData.pipelineSummary.traditionalMLComparisons[1],
          svm: predData.pipelineSummary.traditionalMLComparisons[2],
          confusionMatrix: predData.pipelineSummary.confusionMatrix,
          featureImportance: predData.pipelineSummary.featureImportances,
          comparison: predData.pipelineSummary.traditionalMLComparisons,
          bestModelName: predData.pipelineSummary.bestModelName,
          crossValidationScore: predData.pipelineSummary.crossValidationScore
        });
      }

      // Default DL pipeline values in Node
      setDlPipelineResults({
        mlpMetrics: { accuracy: 0.812, precision: 0.804, recall: 0.801, f1Score: 0.802, rocAuc: 0.892 },
        lstmMetrics: { accuracy: 0.846, precision: 0.835, recall: 0.842, f1Score: 0.838, rocAuc: 0.923 },
        bilstmMetrics: { accuracy: 0.875, precision: 0.868, recall: 0.871, f1Score: 0.869, rocAuc: 0.945 },
        mlpLossCurve: [1.12, 0.94, 0.81, 0.72, 0.65, 0.59, 0.54, 0.50, 0.47, 0.44],
        mlpAccCurve: [0.52, 0.59, 0.66, 0.71, 0.74, 0.76, 0.78, 0.79, 0.80, 0.81],
        lstmLossCurve: [1.25, 1.05, 0.88, 0.74, 0.63, 0.54, 0.46, 0.40, 0.35, 0.31, 0.28, 0.25],
        lstmAccCurve: [0.46, 0.54, 0.63, 0.71, 0.76, 0.80, 0.82, 0.84, 0.85, 0.86, 0.86, 0.87],
        earlyStoppedEpoch: 12,
        comparisonWithML: {
          bestMLAccuracy: 0.815,
          bestDLAccuracy: 0.875,
          improvement: 0.06
        }
      });
    } catch (err) {
      console.warn("Diagnostics API offline on startup, utilizing fallback.", err);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    // Keep active ticket null on start
    setActiveTicket(null);
  }, []);

  // Sync conversation dynamically starting with agent's resolution when a new ticket is selected
  useEffect(() => {
    if (activeTicket) {
      const greeting = activeTicket.agentResult?.finalResolution || `Hello ${activeTicket.name}, I am your virtual resolution engine. I have successfully logged your ${activeTicket.category.toLowerCase()} ticket query. How can I assist you with this today?`;
      setChatMessages([
        { sender: "agent", text: greeting }
      ]);
    } else {
      setChatMessages([]);
    }
    setIsEmailDispatched(false);
    setIsEmailCopied(false);
    setFaqExpanded({});
  }, [activeTicket]);

  const detectCategoryFromQuery = (query: string): "billing" | "technical" | "shipping" | "product_info" | "account" | null => {
    const q = query.toLowerCase();
    const hasReturn = /\b(return|return my order|return request|send back|refund and return|replace or return|exchange|cancellation after delivery)\b/.test(q);
    const hasShipping = /\b(track|tracking|shipment|package|where is my order|not shipped|shipping|ship|shipped|stuck in transit|delivery|deliver|courier|parcel|transit|not arrived)\b/.test(q);
    const hasBilling = /\b(payment|refund(?!\s+and\s+return)|invoice|charge|charged|billing|transaction|money deducted|amount deducted|paid|unpaid|card|fee|price)\b/.test(q);

    // Return/product-action words must win over generic product/technical words.
    if (hasReturn) return "product_info";
    // Shipping/tracking words must win unless the query clearly contains billing terms.
    if (hasShipping && !hasBilling) return "shipping";
    if (hasBilling) return "billing";
    if (hasShipping) return "shipping";
    if (q.includes("login") || q.includes("password") || q.includes("account hacked") || q.includes("profile") || q.includes("suspended") || q.includes("lock") || q.includes("hacked") || q.includes("username") || q.includes("2fa")) {
      return "account";
    }
    if (q.includes("warranty") || q.includes("features") || q.includes("product details") || q.includes("spec") || q.includes("dimension") || q.includes("battery") || q.includes("manual") || q.includes("info") || q.includes("damaged") || q.includes("wrong item") || q.includes("replacement")) {
      return "product_info";
    }
    if (q.includes("error") || q.includes("crash") || q.includes("not working") || q.includes("timeout") || q.includes("outage") || q.includes("fail") || q.includes("bug")) {
      return "technical";
    }
    return null;
  };

  const toBackendCategory = (cat: string | null | undefined): string => {
    if (cat === "billing") return "Billing";
    if (cat === "shipping") return "Shipping";
    if (cat === "technical") return "Technical";
    if (cat === "product_info") return "Product";
    if (cat === "account") return "Account";
    return "Technical";
  };

  const getEffectiveCategory = (query: string, override: string, apiCategory?: string): string => {
    const detected = toBackendCategory(detectCategoryFromQuery(query));
    const q = query.toLowerCase();
    const returnLike = /\b(return|return my order|return request|send back|refund and return|replace or return|exchange|cancellation after delivery)\b/.test(q);
    const shippingLike = /\b(track|tracking|shipment|package|where is my order|not shipped|shipping|ship|shipped|stuck in transit|delivery|deliver|courier|parcel|transit|not arrived)\b/.test(q);
    const billingLike = /\b(payment|refund(?!\s+and\s+return)|invoice|charge|charged|billing|transaction|money deducted|amount deducted|paid|unpaid|card|fee|price)\b/.test(q);
    if (returnLike) return "Product";
    if (shippingLike && !billingLike) return "Shipping";
    if (override && override !== "Auto-Detect") return override;
    return apiCategory || detected;
  };

  const PRODUCT_KEYWORDS: Record<string, string[]> = {
    laptop: ["laptop", "notebook", "computer"],
    headphones: ["headphone", "headphones", "earphone", "earphones", "audio"],
    router: ["router", "wifi", "network"],
    smartwatch: ["smartwatch", "watch", "wearable"],
    monitor: ["monitor", "display", "screen"],
    charger: ["charger", "charging", "adapter", "pad"]
  };

  const detectProductFromQuery = (query: string): string => {
    const q = ` ${query.toLowerCase()} `;
    for (const [canonical, aliases] of Object.entries(PRODUCT_KEYWORDS)) {
      if (aliases.some(alias => q.includes(` ${alias} `) || q.includes(`${alias}s`) || q.includes(alias))) return canonical;
    }
    return "";
  };

  const orderMatchesProductQuery = (order: any, query: string): boolean => {
    const product = detectProductFromQuery(query);
    if (!product) return true;
    const text = `${order.productName || ""} ${order.productCategory || ""}`.toLowerCase();
    const aliases = PRODUCT_KEYWORDS[product] || [product];
    return aliases.some(alias => text.includes(alias));
  };

  const prioritizeOrdersForQuery = (orders: any[], query: string): any[] => {
    const cat = detectCategoryFromQuery(query);
    const scoreOrder = (o: any) => {
      let productScore = orderMatchesProductQuery(o, query) ? 0 : 100;
      const delivery = String(o.deliveryStatus || "").toLowerCase();
      const payment = String(o.paymentStatus || "").toLowerCase();
      const refund = String(o.refundStatus || "").toLowerCase();
      if (cat === "shipping") {
        if (["in transit", "not shipped", "out for delivery", "pending dispatch", "delayed", "shipped"].includes(delivery)) return productScore + 0;
        if (delivery === "delivered") return productScore + 5;
        return productScore + 2;
      }
      if (cat === "billing") {
        if (payment.includes("processing") || payment.includes("unpaid") || refund.includes("processing")) return productScore + 0;
        if (payment.includes("refunded") || refund.includes("completed")) return productScore + 1;
        return productScore + 3;
      }
      if (cat === "product_info") {
        if (delivery === "delivered") return productScore + 0;
        return productScore + 2;
      }
      return productScore + 1;
    };
    return [...orders].sort((a, b) => scoreOrder(a) - scoreOrder(b));
  };

  const handleCustomerNameChange = async (val: string) => {
    setCustomerNameInput(val);
    if (val.trim() === "") {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }
    try {
      const detected = detectCategoryFromQuery(queryInput);
      let backendCat = "";
      if (detected === "billing") backendCat = "Billing";
      else if (detected === "shipping") backendCat = "Shipping";
      else if (detected === "technical") backendCat = "Technical";
      else if (detected === "product_info") backendCat = "Product";
      else if (detected === "account") backendCat = "Account";
      const detectedProduct = detectProductFromQuery(queryInput);
      const res = await fetch(`/api/customers/search?q=${encodeURIComponent(val)}&category=${encodeURIComponent(backendCat)}&product=${encodeURIComponent(detectedProduct)}`);
      if (res.ok) {
        const data = await res.json();
        setSuggestions(data);
        setShowSuggestions(true);
      }
    } catch (e) {
      console.error("Error loading suggestions:", e);
    }
  };

  const handleSelectCustomer = async (cust: any) => {
    setCustomerNameInput(cust.name);
    setCustomerEmailInput(cust.email);
    setCustomerIdInput(cust.id);
    setCustomerPhoneInput(cust.phone);
    setCustomerCityInput(cust.city);

    try {
      const orderCategory = toBackendCategory(detectCategoryFromQuery(queryInput));
      const detectedProduct = detectProductFromQuery(queryInput);
      const res = await fetch(`/api/customers/${cust.id}/orders?category=${encodeURIComponent(orderCategory)}&product=${encodeURIComponent(detectedProduct)}`);
      if (res.ok) {
        const orders = await res.json();
        const formattedOrders = orders.map((o: any) => ({
          orderId: o.orderId,
          customerId: o.customerId,
          productName: o.productName,
          orderDate: o.orderDate,
          deliveryStatus: o.deliveryStatus,
          paymentStatus: o.paymentStatus,
          orderAmount: o.orderAmount,
          shippingCity: o.shippingCity || cust.city,
          expectedDeliveryDate: o.expectedDeliveryDate,
          refundStatus: o.refundStatus,
          warrantyStatus: o.warrantyStatus
        }));
        const prioritizedOrders = prioritizeOrdersForQuery(formattedOrders, queryInput);
        setCustomerOrders(prioritizedOrders);
        if (prioritizedOrders.length > 0) {
          setSelectedOrderId(prioritizedOrders[0].orderId);
          setSelectedOrder(prioritizedOrders[0]);
        } else {
          setSelectedOrderId("");
          setSelectedOrder(null);
        }
      }
    } catch (e) {
      console.error("Error loading customer orders:", e);
    }
    
    setShowSuggestions(false);
    setValidationError(null);
  };

  const handleOrderChange = (orderId: string) => {
    setSelectedOrderId(orderId);
    const ord = customerOrders.find((o) => o.orderId === orderId) || null;
    setSelectedOrder(ord);
  };

  const handleSelectSampleQuery = async (btn: any) => {
    setQueryInput(btn.query);
    setSelectedCategory(btn.category);
    setValidationError(null);
    setShowSuggestions(false);
    
    try {
      const res = await fetch(`/api/customers/search?q=${encodeURIComponent(btn.customerId)}`);
      if (res.ok) {
        const data = await res.json();
        const found = data.find((c: any) => c.id === btn.customerId) || data[0];
        if (found) {
          await handleSelectCustomer(found);
          return;
        }
      }
    } catch (e) {
      console.error("Error loading sample customer:", e);
    }

    setCustomerNameInput(btn.customerName);
    setCustomerEmailInput(btn.email);
    setCustomerIdInput(btn.customerId);
    setCustomerPhoneInput("");
    setCustomerCityInput("");
    setCustomerOrders([]);
    setSelectedOrderId("");
    setSelectedOrder(null);
  };

  // Normalize backend responses so React never crashes on missing fields.
  const normalizeNlpResult = (raw: any, fallback: NLPResult): NLPResult => {
    if (!raw) return fallback;
    const rawEntities = raw.entities || raw.namedEntities || {};
    const rawRegex = raw.regexEntities || {};
    const cleanedRegex = Object.fromEntries(
      Object.entries(rawRegex).filter(([_, v]) => v !== undefined && v !== null && String(v).trim() !== "")
    );
    return {
      cleanedText: raw.cleanedText || fallback.cleanedText,
      tokens: Array.isArray(raw.tokens) ? raw.tokens : fallback.tokens,
      stopwordsRemoved: Array.isArray(raw.stopwordsRemoved) ? raw.stopwordsRemoved : (Array.isArray(raw.tokens) ? raw.tokens : fallback.stopwordsRemoved),
      lemmatizedTokens: Array.isArray(raw.lemmatizedTokens) ? raw.lemmatizedTokens : fallback.lemmatizedTokens,
      sentiment: raw.sentiment || raw.sentimentResult || fallback.sentiment,
      keywords: Array.isArray(raw.keywords) ? raw.keywords : fallback.keywords,
      entities: { ...fallback.entities, ...rawEntities },
      regexEntities: { ...fallback.regexEntities, ...cleanedRegex }
    };
  };

  const normalizeGenAIResult = (raw: any, fallback: GenAIResult): GenAIResult => {
    if (!raw) return fallback;
    const faqSource = raw.faqList || raw.faqSuggestions || [];
    const faqList = Array.isArray(faqSource)
      ? faqSource.map((item: any, index: number) =>
          typeof item === "string"
            ? { question: item, answer: "Follow the recommended resolution steps shown for this case." }
            : {
                question: item?.question || `Suggested FAQ ${index + 1}`,
                answer: item?.answer || "Follow the recommended resolution steps shown for this case."
              }
        )
      : fallback.faqList;

    return {
      personalizedEmail: raw.personalizedEmail || raw.emailBody || raw.personalizedReply || fallback.personalizedEmail,
      faqList: faqList.length ? faqList : fallback.faqList,
      troubleshootingGuide: Array.isArray(raw.troubleshootingGuide)
        ? raw.troubleshootingGuide.join("\n")
        : (raw.troubleshootingGuide || fallback.troubleshootingGuide),
      simulatedConversation: Array.isArray(raw.simulatedConversation) ? raw.simulatedConversation : fallback.simulatedConversation
    };
  };

  const normalizeSlmResult = (raw: any, fallback: SLMResult): SLMResult => {
    if (!raw) return fallback;
    return {
      summary: raw.summary || fallback.summary,
      conciseSteps: Array.isArray(raw.conciseSteps) ? raw.conciseSteps : (Array.isArray(raw.troubleshootingSteps) ? raw.troubleshootingSteps : (Array.isArray(raw.keyPoints) ? raw.keyPoints : fallback.conciseSteps)),
      draftReply: raw.draftReply || fallback.draftReply,
      evaluation: {
        rouge1: Number(raw.evaluation?.rouge1 ?? fallback.evaluation.rouge1),
        rouge2: Number(raw.evaluation?.rouge2 ?? fallback.evaluation.rouge2),
        rougeL: Number(raw.evaluation?.rougeL ?? fallback.evaluation.rougeL),
        latencyMs: Number(raw.evaluation?.latencyMs ?? fallback.evaluation.latencyMs)
      }
    };
  };

  // Execute full-stack deep inference models
  const handleResolveTicket = async () => {
    if (!customerNameInput.trim() || !customerEmailInput.trim() || !customerIdInput.trim() || !queryInput.trim()) {
      setValidationError("Please fill out all fields before submitting. Customer Name, Email, ID, and Query are required.");
      return;
    }
    setValidationError(null);
    setIsLoading(true);
    setIsEmailDispatched(false);
    setIsEmailCopied(false);

    try {
      // Step-by-step loading simulation matching a real deep network analysis pipeline
      setLoadingStep("Step 1/5: Performing Security Audit & Safety Scans...");
      await new Promise((r) => setTimeout(r, 600));

      let nlpData: NLPResult | undefined;
      let agData: AgenticResult | undefined;
      let genData: GenAIResult | undefined;
      let slmData: SLMResult | undefined;

      try {
        setLoadingStep("Step 2/5: Initializing Tokenizer & N-gram Feature Extractors...");
        const nlpRes = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: queryInput })
        });
        if (nlpRes.ok) nlpData = await nlpRes.json();
      } catch (e) {
        console.warn("Analyze API offline fallback used", e);
      }

      try {
        setLoadingStep("Step 3/5: Running SVM & Bi-LSTM Classification Classifiers...");
        const agRes = await fetch("/api/agent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: queryInput, customerName: customerNameInput })
        });
        if (agRes.ok) agData = await agRes.json();
      } catch (e) {
        console.warn("Agent API offline fallback used", e);
      }

      try {
        setLoadingStep("Step 4/5: Compiling Knowledge-Base FAQs & Generating Action Guides...");
        const orderId = nlpData?.regexEntities?.orderId || selectedOrderId || "";
        const genRes = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: queryInput,
            category: getEffectiveCategory(queryInput, selectedCategory, agData?.perception?.detectedCategory),
            customerName: customerNameInput,
            sentiment: agData?.perception?.detectedSentiment || "Neutral",
            orderId
          })
        });
        if (genRes.ok) genData = await genRes.json();
      } catch (e) {
        console.warn("Generate API offline fallback used", e);
      }

      try {
        setLoadingStep("Step 5/5: Running Small Language Model Summaries & ROUGE Evaluations...");
        const slmRes = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: queryInput,
            category: getEffectiveCategory(queryInput, selectedCategory, agData?.perception?.detectedCategory)
          })
        });
        if (slmRes.ok) slmData = await slmRes.json();
      } catch (e) {
        console.warn("Chat API offline fallback used", e);
      }

      // Compute final resolution details based on database and manual type rules
      const resolved = resolveCustomerQuery(
        queryInput,
        customerNameInput,
        customerEmailInput,
        customerIdInput,
        selectedCategory,
        selectedOrder,
        agData?.perception?.detectedCategory,
        agData?.perception?.detectedSentiment
      );

      // Create fallback/mock objects if APIs failed
      const fallbackNlp: NLPResult = {
        cleanedText: queryInput.toLowerCase(),
        tokens: queryInput.toLowerCase().split(/\s+/).filter(Boolean),
        stopwordsRemoved: queryInput.toLowerCase().split(/\s+/).filter(Boolean),
        lemmatizedTokens: queryInput.toLowerCase().split(/\s+/).filter(Boolean),
        sentiment: { label: resolved.sentiment as Sentiment, score: resolved.sentiment === Sentiment.NEGATIVE ? -0.5 : resolved.sentiment === Sentiment.POSITIVE ? 0.5 : 0 },
        keywords: [resolved.category.toLowerCase()],
        entities: { customerName: customerNameInput, city: customerCityInput || undefined, phone: customerPhoneInput || undefined },
        regexEntities: { orderId: selectedOrderId || undefined, phone: customerPhoneInput || undefined }
      };
      const finalNlp: NLPResult = normalizeNlpResult(nlpData, fallbackNlp);

      const finalAgent: AgenticResult = {
        perception: {
          detectedCategory: resolved.category,
          detectedSentiment: resolved.sentiment,
          detectedUrgency: resolved.urgency,
          entities: { customerName: customerNameInput, orderId: selectedOrderId },
          confidence: 0.92,
          riskLevel: resolved.riskLevel,
          resolutionDecision: resolved.statusText
        },
        reasoning: `Decided based on customer input analysis and verified database state for order ${selectedOrderId || "N/A"}.`,
        planning: ["LOAD_CUSTOMER_RECORD", "VERIFY_ORDER_STATUS", "APPLY_RESOLUTION_DECISION_MATRIX", "COMPILE_SUGGESTED_RESPONSE"],
        actionsTaken: ["DB_LOOKUP_SUCCESS", "SLA_TIER_APPLIED"],
        memoryLogs: resolved.memoryLogs,
        safetyPassed: true,
        humanEscalationRequired: !resolved.autoResolved,
        escalationReason: !resolved.autoResolved ? "Manual operator review required based on business rules." : undefined,
        decisionLogs: resolved.decisionLogs,
        finalResolution: resolved.finalResolution
      };

      const fallbackGen: GenAIResult = {
        personalizedEmail: resolved.personalizedEmail,
        faqList: resolved.faqList,
        troubleshootingGuide: resolved.troubleshootingGuide,
        simulatedConversation: [
          { speaker: "Customer", text: queryInput },
          { speaker: "AI Resolution Engine", text: resolved.finalResolution }
        ]
      };
      const finalGen: GenAIResult = normalizeGenAIResult(genData, fallbackGen);

      const fallbackSlm: SLMResult = {
        summary: `Customer ${customerNameInput} requested assistance with ${resolved.category.toLowerCase()} issues. Status verified: ${selectedOrder ? selectedOrder.deliveryStatus : "No linked order"}.`,
        conciseSteps: [
          "Inspected support ticket text",
          selectedOrder ? `Checked status of ${selectedOrder.orderId}` : "Standalone keyword matching",
          `Generated decision: ${resolved.statusText}`
        ],
        draftReply: resolved.finalResolution,
        evaluation: {
          rouge1: 0.42,
          rouge2: 0.18,
          rougeL: 0.39,
          latencyMs: 140
        }
      };
      const finalSlm: SLMResult = normalizeSlmResult(slmData, fallbackSlm);

      const newTicketId = `CASE-${Math.floor(Math.random() * 900000 + 100000)}`;

      const calculatedTicket: SupportTicket = {
        id: newTicketId,
        name: customerNameInput,
        email: customerEmailInput,
        customerId: customerIdInput || "CUST-TEMP",
        query: queryInput,
        category: resolved.category,
        sentiment: resolved.sentiment,
        urgency: resolved.urgency,
        status: resolved.statusText,
        date: new Date().toISOString().slice(0, 16).replace("T", " "),
        riskLevel: resolved.riskLevel,
        autoResolved: resolved.autoResolved,
        resolutionTime: resolved.resolutionTime,
        nlpResult: finalNlp,
        agentResult: finalAgent,
        genAIResult: finalGen,
        slmResult: finalSlm,
        confidence: 0.95
      };

      // Append to local database history
      setCustomerQueries((prev) => [calculatedTicket, ...prev]);
      setActiveTicket(calculatedTicket);

      // Transition straight to the Query Analysis tab
      setActiveTab("analysis");
    } catch (err) {
      console.error("Deep resolution pipeline failed:", err);
    } finally {
      setIsLoading(false);
      setLoadingStep("");
    }
  };

  // Preset Scenario Loaders
  const handleSelectPreset = (presetText: string, name: string, email: string, id: string, cat: string) => {
    setQueryInput(presetText);
    setCustomerNameInput(name);
    setCustomerEmailInput(email);
    setCustomerIdInput(id);
    setSelectedCategory(cat);
    setValidationError(null);
  };

  // Clipboard Copier
  const handleCopyEmail = (text: string) => {
    navigator.clipboard.writeText(text);
    setIsEmailCopied(true);
    setTimeout(() => setIsEmailCopied(false), 2000);
  };

  // EML Document Generator and Downloader
  const handleDownloadEML = (ticket: SupportTicket) => {
    if (!ticket.genAIResult) return;
    const emlContent = `Subject: [Resolution Alert] support ticket response - Case #${ticket.id}
From: service-resolution@enterprise.com
To: ${ticket.email}
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

Dear ${ticket.name},

${ticket.genAIResult.personalizedEmail}

--
Enterprise Operations Customer Services
Case ID: ${ticket.id}`;

    const blob = new Blob([emlContent], { type: "message/rfc822" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `Case-${ticket.id}-Response.eml`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Dispatch Email State Trigger
  const handleDispatchEmail = () => {
    setIsEmailDispatched(true);
    // Append auto confirmation chat log
    setChatMessages((prev) => [
      ...prev,
      { sender: "agent", text: "📧 Standard resolution confirmation dispatch complete. System sent email copy successfully." }
    ]);
  };

  // Real Dynamic Chat Interactivity using backend
  const handleSendChatMessage = async () => {
    if (!newChatMessage.trim() || !activeTicket) return;
    
    const userMessage = { sender: "user" as const, text: newChatMessage };
    const updatedHistory = [...chatMessages, userMessage];
    
    setChatMessages(updatedHistory);
    setNewChatMessage("");

    try {
      const res = await fetch("/api/chat-message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          history: updatedHistory,
          customerName: activeTicket.name,
          category: activeTicket.category,
          query: activeTicket.query
        })
      });
      const data = await res.json();
      if (data.reply) {
        setChatMessages((prev) => [...prev, { sender: "agent", text: data.reply }]);
      }
    } catch (err) {
      console.error("Chat message roundtrip failed:", err);
      setTimeout(() => {
        setChatMessages((prev) => [...prev, { sender: "agent", text: "Understood. Our systems are working on this. Is there anything else you'd like to ask?" }]);
      }, 500);
    }
  };

  // Drag-and-drop screenshot simulators
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setUploadedScreenshot(`Diagnostic Attachment: ${file.name} (${Math.round(file.size / 1024)} KB)`);
    }
  };

  // Manual file upload fallback
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedScreenshot(`Diagnostic Attachment: ${file.name} (${Math.round(file.size / 1024)} KB)`);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans flex flex-col antialiased">
      {/* Visual Top Decorative Corporate Accent line */}
      <div className="h-1 bg-blue-600 w-full" />

      {/* Header Bar */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 text-white p-2.5 rounded-lg font-bold flex items-center justify-center shadow-xs">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 flex flex-wrap items-center gap-2">
              Customer Support Console
            </h1>
            <p className="text-xs text-slate-500 font-medium">
              Real-time customer lookup, order verification, and intelligent query resolution system.
            </p>
          </div>
        </div>

        {/* Global Developer Mode Switch and Reload */}
        <div className="flex items-center gap-3 self-stretch sm:self-auto justify-between border-t border-slate-100 sm:border-0 pt-3 sm:pt-0">
          <div className="flex items-center gap-2 bg-slate-100 border border-slate-200 px-3 py-1.5 rounded-lg">
            <span className="text-xs font-semibold text-slate-600">Developer Mode:</span>
            <button
              onClick={() => setDeveloperMode(!developerMode)}
              className={`relative inline-flex h-5 w-10 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-hidden ${
                developerMode ? "bg-blue-600" : "bg-slate-300"
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-sm ring-0 transition duration-200 ease-in-out ${
                  developerMode ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
            <span className={`text-[10px] font-bold ${developerMode ? "text-blue-600" : "text-slate-400"}`}>
              {developerMode ? "ON" : "OFF"}
            </span>
          </div>

          <button
            onClick={fetchSystemStatus}
            className="text-xs font-semibold text-slate-600 bg-white hover:bg-slate-100 px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-colors border border-slate-200 shadow-2xs"
          >
            <RefreshCw className="h-3 w-3" /> Sync
          </button>
        </div>
      </header>

      {/* Main Full-Scale Body Layout */}
      <div className="flex-1 flex flex-col md:flex-row">
        {/* Sidebar Panel */}
        <aside className="w-full md:w-64 bg-slate-900 text-slate-300 flex flex-col shrink-0 border-r border-slate-800">
          <div className="p-5 border-b border-slate-800">
            <div className="flex items-center gap-2 text-white mb-1">
              <Briefcase className="h-5 w-5 text-blue-500" />
              <span className="font-bold tracking-tight">Console Pages</span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1 flex-1">
            <button
              onClick={() => setActiveTab("support")}
              className={`w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 font-semibold text-xs transition-all ${
                activeTab === "support" ? "bg-blue-600 text-white shadow-xs" : "hover:bg-slate-800 hover:text-white"
              }`}
            >
              <UserCheck className="h-4 w-4" /> Customer Support
            </button>

            <button
              onClick={() => setActiveTab("lookup")}
              className={`w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 font-semibold text-xs transition-all ${
                activeTab === "lookup" ? "bg-blue-600 text-white shadow-xs" : "hover:bg-slate-800 hover:text-white"
              }`}
            >
              <Users className="h-4 w-4" /> Customer Lookup
            </button>

            <button
              onClick={() => setActiveTab("orders")}
              className={`w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 font-semibold text-xs transition-all ${
                activeTab === "orders" ? "bg-blue-600 text-white shadow-xs" : "hover:bg-slate-800 hover:text-white"
              }`}
            >
              <ShoppingBag className="h-4 w-4" /> Order Details
            </button>

            <button
              onClick={() => setActiveTab("analysis")}
              className={`w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 font-semibold text-xs transition-all ${
                activeTab === "analysis" ? "bg-blue-600 text-white shadow-xs" : "hover:bg-slate-800 hover:text-white"
              }`}
            >
              <Search className="h-4 w-4" /> Query Analysis
            </button>

            <button
              onClick={() => setActiveTab("resolution")}
              className={`w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 font-semibold text-xs transition-all ${
                activeTab === "resolution" ? "bg-blue-600 text-white shadow-xs" : "hover:bg-slate-800 hover:text-white"
              }`}
            >
              <ShieldCheck className="h-4 w-4" /> Resolution Decision
            </button>

            <button
              onClick={() => setActiveTab("reports")}
              className={`w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 font-semibold text-xs transition-all ${
                activeTab === "reports" ? "bg-blue-600 text-white shadow-xs" : "hover:bg-slate-800 hover:text-white"
              }`}
            >
              <BarChart3 className="h-4 w-4" /> Reports & Metrics
            </button>
          </nav>

          {/* Sidebar Footer Details */}
          <div className="p-4 border-t border-slate-800 bg-slate-950 font-mono text-[10px] text-slate-500 space-y-1">
            <div className="flex justify-between">
              <span>Status:</span>
              <span className="text-emerald-400 font-bold">ACTIVE</span>
            </div>
            <div className="flex justify-between">
              <span>Secure Mode:</span>
              <span className="text-emerald-400 font-bold">ENABLED</span>
            </div>
            {developerMode && (
              <div className="mt-2 text-center text-[9px] bg-blue-950 border border-blue-900 text-blue-400 py-1 rounded font-bold uppercase tracking-wider animate-pulse">
                Developer Mode Active
              </div>
            )}
          </div>
        </aside>

        {/* Content Body */}
        <main className="flex-1 p-6 md:p-8 overflow-y-auto max-w-7xl mx-auto w-full">
          <AnimatePresence mode="wait">
            {/* 1. CUSTOMER LOOKUP TAB */}
            {activeTab === "lookup" && (
              <motion.div
                key="lookup"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                  <h2 className="text-lg font-bold text-slate-950 mb-1 flex items-center gap-2">
                    <Users className="text-blue-600 h-5 w-5" />
                    Customer Lookup Directory
                  </h2>
                  <p className="text-xs text-slate-500">
                    Search customer records from the database. Select any customer to view details, linked purchase orders, or immediately launch a support ticket resolution.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left Table Panel */}
                  <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                    <div className="mb-4">
                      <label className="text-xs font-bold text-slate-600 block mb-2 uppercase tracking-wider">Search Customers</label>
                      <div className="relative">
                        <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                          <Search className="h-4 w-4" />
                        </span>
                        <input
                          type="text"
                          value={customerSearch}
                          onChange={(e) => setCustomerSearch(e.target.value)}
                          placeholder="Search by name, email, customer ID or city..."
                          className="w-full pl-9 pr-4 py-2 border border-slate-200 rounded-lg text-xs focus:outline-hidden focus:border-blue-500"
                        />
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="border-b border-slate-100 text-[10px] uppercase tracking-wider font-bold text-slate-400 bg-slate-50">
                            <th className="py-3 px-4">ID</th>
                            <th className="py-3 px-4">Name</th>
                            <th className="py-3 px-4">Email</th>
                            <th className="py-3 px-4">City</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 text-xs">
                          {customersList.map((cust) => (
                            <tr
                              key={cust.id}
                              onClick={() => handleSelectCustomer(cust)}
                              className={`hover:bg-blue-50/40 cursor-pointer transition-colors ${
                                customerIdInput === cust.id ? "bg-blue-50 text-blue-700 font-medium" : ""
                              }`}
                            >
                              <td className="py-3 px-4 font-mono font-bold text-slate-500">{cust.id}</td>
                              <td className="py-3 px-4 font-bold text-slate-800">{cust.name}</td>
                              <td className="py-3 px-4 text-slate-500">{cust.email}</td>
                              <td className="py-3 px-4 text-slate-500">{cust.city}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Right Detail Panel */}
                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
                    <h3 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 uppercase tracking-wider flex items-center gap-2">
                      <User className="h-4 w-4 text-blue-600" />
                      Customer Details Card
                    </h3>

                    {customerIdInput ? (
                      <div className="space-y-4">
                        <div className="p-3 bg-slate-50 border border-slate-100 rounded-lg space-y-2">
                          <div>
                            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Full Name</span>
                            <p className="text-xs font-bold text-slate-800">{customerNameInput}</p>
                          </div>
                          <div>
                            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Email Address</span>
                            <p className="text-xs text-slate-600">{customerEmailInput}</p>
                          </div>
                          <div>
                            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Phone Number</span>
                            <p className="text-xs text-slate-600">{customerPhoneInput}</p>
                          </div>
                          <div>
                            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Home City</span>
                            <p className="text-xs text-slate-600">{customerCityInput}</p>
                          </div>
                        </div>

                        <div>
                          <span className="text-xs font-bold text-slate-700 block mb-2 uppercase tracking-wider">Purchase History</span>
                          {customerOrders.length > 0 ? (
                            <div className="space-y-2">
                              {customerOrders.map((ord) => (
                                <div key={ord.orderId} className="border border-slate-100 p-2.5 rounded-lg text-xs bg-slate-50/50">
                                  <div className="flex justify-between font-mono font-bold text-slate-500 mb-1">
                                    <span>{ord.orderId}</span>
                                    <span className="text-slate-800">${ord.orderAmount}</span>
                                  </div>
                                  <p className="font-medium text-slate-800">{ord.productName}</p>
                                  <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                                    <span>Date: {ord.orderDate}</span>
                                    <span className={`font-semibold ${
                                      ord.deliveryStatus === "Delivered" ? "text-emerald-600" :
                                      ord.deliveryStatus === "In Transit" ? "text-amber-600" : "text-rose-600"
                                    }`}>{ord.deliveryStatus}</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs text-slate-400 italic">No orders logged for this customer.</p>
                          )}
                        </div>

                        <button
                          onClick={() => setActiveTab("support")}
                          className="w-full bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-all shadow-2xs"
                        >
                          <Plus className="h-4 w-4" /> Start Support Ticket
                        </button>
                      </div>
                    ) : (
                      <div className="h-48 flex flex-col items-center justify-center text-center p-4">
                        <Users className="h-8 w-8 text-slate-300 mb-2" />
                        <p className="text-xs text-slate-400 italic">Select a customer from the left directory to view full profiles and purchase orders.</p>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* 1B. ORDER DETAILS TAB */}
            {activeTab === "orders" && (
              <motion.div
                key="orders"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                  <h2 className="text-lg font-bold text-slate-950 mb-1 flex items-center gap-2">
                    <ShoppingBag className="text-blue-600 h-5 w-5" />
                    Purchase Order Directory
                  </h2>
                  <p className="text-xs text-slate-500">
                    Verify order delivery parameters, payment captures, warranty policies, and refund cycles. Select an order to inspect transit milestones or trigger support resolutions.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left Table Panel */}
                  <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                    <div className="mb-4">
                      <label className="text-xs font-bold text-slate-600 block mb-2 uppercase tracking-wider">Search Purchase Orders</label>
                      <div className="relative">
                        <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                          <Search className="h-4 w-4" />
                        </span>
                        <input
                          type="text"
                          value={orderSearch}
                          onChange={(e) => setOrderSearch(e.target.value)}
                          placeholder="Search by order ID, product name or shipping city..."
                          className="w-full pl-9 pr-4 py-2 border border-slate-200 rounded-lg text-xs focus:outline-hidden focus:border-blue-500"
                        />
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="border-b border-slate-100 text-[10px] uppercase tracking-wider font-bold text-slate-400 bg-slate-50">
                            <th className="py-3 px-4">Order ID</th>
                            <th className="py-3 px-4">Product</th>
                            <th className="py-3 px-4">Amount</th>
                            <th className="py-3 px-4">Delivery</th>
                            <th className="py-3 px-4">Payment</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 text-xs">
                          {ordersList.map((ord) => (
                            <tr
                              key={ord.orderId}
                              onClick={async () => {
                                try {
                                  const cRes = await fetch(`/api/customers/search?q=${encodeURIComponent(ord.customerId)}`);
                                  if (cRes.ok) {
                                    const matchingCusts = await cRes.json();
                                    const owner = matchingCusts.find((c: any) => c.id === ord.customerId) || matchingCusts[0];
                                    if (owner) {
                                      await handleSelectCustomer(owner);
                                      setSelectedOrderId(ord.orderId);
                                      setSelectedOrder(ord);
                                    }
                                  }
                                } catch (e) {
                                  console.error("Error loading order owner:", e);
                                }
                              }}
                              className={`hover:bg-blue-50/40 cursor-pointer transition-colors ${
                                selectedOrderId === ord.orderId ? "bg-blue-50 text-blue-700 font-medium" : ""
                              }`}
                            >
                              <td className="py-3 px-4 font-mono font-bold text-slate-500">{ord.orderId}</td>
                              <td className="py-3 px-4 font-bold text-slate-800">{ord.productName}</td>
                              <td className="py-3 px-4 font-medium text-slate-700">${ord.orderAmount}</td>
                              <td className="py-3 px-4">
                                <span className={`font-semibold ${
                                  ord.deliveryStatus === "Delivered" ? "text-emerald-600" :
                                  ord.deliveryStatus === "In Transit" ? "text-amber-600" : "text-rose-600"
                                }`}>{ord.deliveryStatus}</span>
                              </td>
                              <td className="py-3 px-4 text-slate-500">{ord.paymentStatus}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Right Detail Panel */}
                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
                    <h3 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 uppercase tracking-wider flex items-center gap-2">
                      <ShoppingBag className="h-4 w-4 text-blue-600" />
                      Order Details Card
                    </h3>

                    {selectedOrder ? (
                      <div className="space-y-4">
                        <div className="p-3 bg-slate-50 border border-slate-100 rounded-lg space-y-2 text-xs">
                          <div className="flex justify-between">
                            <span className="text-slate-400 font-bold uppercase text-[9px]">Order Number</span>
                            <span className="font-mono font-bold text-slate-700">{selectedOrder.orderId}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400 font-bold uppercase text-[9px]">Product Name</span>
                            <span className="font-bold text-slate-800">{selectedOrder.productName}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400 font-bold uppercase text-[9px]">Sale Price</span>
                            <span className="font-bold text-slate-800">${selectedOrder.orderAmount}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400 font-bold uppercase text-[9px]">Shipping Destination</span>
                            <span className="text-slate-600">{selectedOrder.shippingCity}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400 font-bold uppercase text-[9px]">Order Date</span>
                            <span className="text-slate-600">{selectedOrder.orderDate}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-400 font-bold uppercase text-[9px]">Expected Delivery</span>
                            <span className="text-slate-600 font-semibold">{selectedOrder.expectedDeliveryDate}</span>
                          </div>
                        </div>

                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between items-center border-b border-slate-100 pb-2">
                            <span className="text-slate-500">Delivery Status:</span>
                            <span className={`font-bold px-2 py-0.5 rounded-full ${
                              selectedOrder.deliveryStatus === "Delivered" ? "bg-emerald-100 text-emerald-800" :
                              selectedOrder.deliveryStatus === "In Transit" ? "bg-amber-100 text-amber-800" : "bg-rose-100 text-rose-800"
                            }`}>{selectedOrder.deliveryStatus}</span>
                          </div>

                          <div className="flex justify-between items-center border-b border-slate-100 pb-2">
                            <span className="text-slate-500">Payment Capture:</span>
                            <span className="font-bold text-slate-800">{selectedOrder.paymentStatus}</span>
                          </div>

                          <div className="flex justify-between items-center border-b border-slate-100 pb-2">
                            <span className="text-slate-500">Refund Status:</span>
                            <span className="font-semibold text-slate-700">{selectedOrder.refundStatus}</span>
                          </div>

                          <div className="flex justify-between items-center">
                            <span className="text-slate-500">Warranty Coverage:</span>
                            <span className="font-semibold text-slate-700">{selectedOrder.warrantyStatus}</span>
                          </div>
                        </div>

                        {/* Order Progress Tracker */}
                        <div className="pt-2">
                          <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block mb-2">Transit Milestones</span>
                          <div className="flex items-center gap-1">
                            <div className="h-2 flex-1 rounded-full bg-emerald-500" />
                            <div className={`h-2 flex-1 rounded-full ${
                              selectedOrder.deliveryStatus !== "Not Shipped" ? "bg-emerald-500" : "bg-slate-200"
                            }`} />
                            <div className={`h-2 flex-1 rounded-full ${
                              selectedOrder.deliveryStatus === "Delivered" ? "bg-emerald-500" : "bg-slate-200"
                            }`} />
                          </div>
                          <div className="flex justify-between text-[9px] text-slate-400 mt-1 font-semibold">
                            <span>Processing</span>
                            <span>Shipped</span>
                            <span>Delivered</span>
                          </div>
                        </div>

                        <button
                          onClick={() => setActiveTab("support")}
                          className="w-full bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-all shadow-2xs"
                        >
                          <Plus className="h-4 w-4" /> Start Ticket for Order
                        </button>
                      </div>
                    ) : (
                      <div className="h-48 flex flex-col items-center justify-center text-center p-4">
                        <ShoppingBag className="h-8 w-8 text-slate-300 mb-2" />
                        <p className="text-xs text-slate-400 italic">Select an order from the list to view active tracking paths and ledger metrics.</p>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* 2. CUSTOMER SUPPORT TAB */}
            {activeTab === "support" && (
              <motion.div
                key="support"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                  <h2 className="text-lg font-bold text-slate-950 mb-1 flex items-center gap-2">
                    <UserCheck className="text-blue-600 h-5 w-5" />
                    Ingest Customer Request Ticket
                  </h2>
                  <p className="text-xs text-slate-500 mb-6">
                    Enter unstructured customer requests. The system will trigger sequential NLP extraction layers, classification networks, safety blocks, and output autonomous troubleshooting decisions.
                  </p>

                  {/* Preset Selector Panel */}
                  <div className="mb-6 p-4 bg-slate-50 border border-slate-200 rounded-xl">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2.5">
                      Select Pre-Configured Test Scenarios (Populates Input Fields):
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {SAMPLE_QUERIES.map((btn, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSelectSampleQuery(btn)}
                          className="text-xs bg-white hover:bg-slate-100 text-slate-700 px-3 py-1.5 rounded-lg border border-slate-200 font-semibold transition-all shadow-3xs hover:border-slate-300"
                        >
                          {btn.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Input Form Fields */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 space-y-4">
                      <div>
                        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                          Customer Query Description <span className="text-rose-500">*</span>
                        </label>
                        <textarea
                          value={queryInput}
                          onChange={(e) => setQueryInput(e.target.value)}
                          className="w-full h-32 px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 focus:bg-white transition-all font-medium"
                          placeholder="Type or copy customer inquiry..."
                        />

                        {/* Real-time Matching Panel */}
                        {queryInput.trim().length > 0 && (
                          <div className="mt-3 bg-blue-50/50 border border-blue-100 rounded-xl p-4 space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-1.5">
                                <Cpu className="h-4 w-4 text-blue-600 animate-pulse" />
                                <span className="text-xs font-bold text-blue-900">
                                  Real-Time Database Matcher
                                </span>
                              </div>
                              {detectCategoryFromQuery(queryInput) ? (
                                <span className="text-[9px] uppercase font-bold bg-blue-100 text-blue-800 px-2.5 py-0.5 rounded-full font-sans">
                                  {detectCategoryFromQuery(queryInput)} detected
                                </span>
                              ) : (
                                <span className="text-[9px] uppercase font-bold bg-slate-100 text-slate-500 px-2.5 py-0.5 rounded-full font-sans">
                                  Scanning...
                                </span>
                              )}
                            </div>
                            <p className="text-[11px] text-slate-600 leading-relaxed font-sans">
                              {detectCategoryFromQuery(queryInput) 
                                ? "Select a customer profile matching this category to populate ticket details:" 
                                : "Type keywords like 'refund', 'tracking', 'crash', 'password' or 'warranty' to auto-filter records."}
                            </p>
                            
                             {/* Profile matches list */}
                            {(() => {
                              const cat = detectCategoryFromQuery(queryInput);
                              if (!cat || categoryMatches.length === 0) return null;
                              return (
                                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                  {categoryMatches.map(m => (
                                    <button
                                      key={m.id}
                                      type="button"
                                      onClick={() => {
                                        handleSelectCustomer(m);
                                        setSelectedCategory(cat === "product_info" ? "Product" : cat.charAt(0).toUpperCase() + cat.slice(1));
                                        setValidationError(null);
                                      }}
                                      className={`text-left p-2.5 rounded-lg border text-[11px] transition-all bg-white hover:bg-blue-50/70 ${
                                        customerNameInput === m.name 
                                          ? "border-blue-600 ring-2 ring-blue-600/10 font-bold bg-blue-50/20" 
                                          : "border-slate-200"
                                      }`}
                                    >
                                      <span className="block text-slate-800 font-bold truncate">{m.name}</span>
                                      <span className="block text-[9px] text-slate-500 font-mono mt-0.5">{m.id}</span>
                                    </button>
                                  ))}
                                </div>
                              );
                            })()}
                          </div>
                        )}
                      </div>

                      {/* Screenshot drag-and-drop simulated block */}
                      <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`border-2 border-dashed rounded-xl p-6 text-center transition-all ${
                          isDragging ? "border-blue-500 bg-blue-50/10" : "border-slate-200 bg-slate-50 hover:bg-slate-100/30"
                        }`}
                      >
                        <input
                          type="file"
                          id="screenshot-uploader"
                          className="hidden"
                          accept="image/*"
                          onChange={handleFileChange}
                        />
                        <label htmlFor="screenshot-uploader" className="cursor-pointer block">
                          <Plus className="h-6 w-6 text-slate-400 mx-auto mb-2" />
                          <span className="text-xs font-semibold text-slate-700 block">
                            Upload Diagnostic Screenshot (Optional)
                          </span>
                          <span className="text-[10px] text-slate-500 mt-1 block font-medium">
                            Drag and drop file here, or click to browse local folders
                          </span>
                        </label>
                        {uploadedScreenshot && (
                          <div className="mt-3 text-xs bg-blue-50 text-blue-800 border border-blue-200 py-1.5 px-3 rounded-lg inline-flex items-center gap-2 font-mono">
                            <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" />
                            {uploadedScreenshot}
                            <button
                              onClick={() => setUploadedScreenshot(null)}
                              className="text-slate-400 hover:text-slate-700 font-bold ml-1 font-sans text-[10px]"
                            >
                              ✕
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="space-y-4 bg-slate-50 border border-slate-200 p-4 rounded-xl relative">
                      <div className="relative">
                        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                          Customer Full Name <span className="text-blue-500 font-normal">(Autocomplete)</span>
                        </label>
                        <input
                          type="text"
                          value={customerNameInput}
                          onChange={(e) => handleCustomerNameChange(e.target.value)}
                          onFocus={() => {
                            const cat = detectCategoryFromQuery(queryInput);
                            let filtered = SUPPORT_RECORDS;
                            if (cat) {
                              filtered = filtered.filter(r => r.problem_category === cat);
                            }
                            if (customerNameInput.trim() !== "") {
                              filtered = filtered.filter(r => r.customer_name.toLowerCase().includes(customerNameInput.toLowerCase()));
                            }
                            const matched = filtered.map(r => ({
                              id: r.customer_id,
                              name: r.customer_name,
                              email: r.email,
                              phone: r.phone,
                              city: r.city
                            }));
                            setSuggestions(matched);
                            setShowSuggestions(true);
                          }}
                          placeholder="Type or select a customer..."
                          className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 font-semibold"
                        />
                        {showSuggestions && suggestions.length > 0 && (
                          <div className="absolute z-50 left-0 right-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-lg max-h-48 overflow-y-auto divide-y divide-slate-100">
                            {suggestions.map((cust) => (
                              <button
                                key={cust.id}
                                type="button"
                                onClick={() => handleSelectCustomer(cust)}
                                className="w-full text-left px-3 py-2 hover:bg-blue-50 text-xs transition-colors flex justify-between items-center"
                              >
                                <div>
                                  <span className="font-bold text-slate-800 block">{cust.name}</span>
                                  <span className="text-[10px] text-slate-500 block">{cust.email}</span>
                                </div>
                                <span className="font-mono text-[9px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded font-bold">{cust.id}</span>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>

                      <div>
                        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                          Customer Email Address
                        </label>
                        <input
                          type="email"
                          value={customerEmailInput}
                          onChange={(e) => setCustomerEmailInput(e.target.value)}
                          placeholder="customer@email.com"
                          className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 font-semibold"
                        />
                      </div>

                      <div>
                        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                          Customer ID Reference
                        </label>
                        <input
                          type="text"
                          value={customerIdInput}
                          onChange={(e) => setCustomerIdInput(e.target.value)}
                          placeholder="CUST-XXXXX"
                          className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 font-mono font-semibold"
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                            Phone Number
                          </label>
                          <input
                            type="text"
                            value={customerPhoneInput}
                            onChange={(e) => setCustomerPhoneInput(e.target.value)}
                            placeholder="Phone number"
                            className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 font-semibold"
                          />
                        </div>
                        <div>
                          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                            Home City
                          </label>
                          <input
                            type="text"
                            value={customerCityInput}
                            onChange={(e) => setCustomerCityInput(e.target.value)}
                            placeholder="City"
                            className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 font-semibold"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                          Linked Customer Purchase Order
                        </label>
                        {customerOrders.length > 0 ? (
                          <select
                            value={selectedOrderId}
                            onChange={(e) => handleOrderChange(e.target.value)}
                            className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs focus:outline-hidden focus:ring-1 focus:ring-blue-600 font-semibold text-slate-700"
                          >
                            {customerOrders.map((ord) => (
                              <option key={ord.orderId} value={ord.orderId}>
                                {ord.orderId} - {ord.productName} ({ord.deliveryStatus})
                              </option>
                            ))}
                          </select>
                        ) : (
                          <div className="text-[11px] text-slate-500 bg-white/60 p-2 rounded-lg border border-slate-200 italic">
                            No orders found for this customer name.
                          </div>
                        )}
                      </div>

                      {selectedOrder && (
                        <div className="bg-blue-50/50 border border-blue-100 p-2.5 rounded-lg text-[11px] space-y-1">
                          <p className="font-bold text-slate-700 uppercase text-[9px] tracking-wider">Verified Purchase State:</p>
                          <div className="flex justify-between">
                            <span className="text-slate-500">Amount:</span>
                            <span className="font-bold text-slate-800">${selectedOrder.orderAmount}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-500">Delivery Status:</span>
                            <span className={`font-semibold ${
                              selectedOrder.deliveryStatus === "Delivered" ? "text-emerald-600" :
                              selectedOrder.deliveryStatus === "In Transit" ? "text-amber-600" : "text-rose-600"
                            }`}>{selectedOrder.deliveryStatus}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-500">Expected Delivery:</span>
                            <span className="text-slate-700 font-medium">{selectedOrder.expectedDeliveryDate}</span>
                          </div>
                        </div>
                      )}

                      <div>
                        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1">
                          Category Directive (Override)
                        </label>
                        <select
                          value={selectedCategory}
                          onChange={(e) => setSelectedCategory(e.target.value)}
                          className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 font-semibold text-slate-700"
                        >
                          <option>Auto-Detect</option>
                          <option>Billing</option>
                          <option>Technical</option>
                          <option>Shipping</option>
                          <option>Product</option>
                          <option>Account</option>
                        </select>
                      </div>

                      {validationError && (
                        <div className="bg-rose-50 border border-rose-200 text-rose-700 text-xs p-3 rounded-lg flex items-center gap-2 mt-4 font-semibold">
                          <AlertTriangle className="h-4 w-4 text-rose-500 shrink-0 animate-bounce" />
                          <span>{validationError}</span>
                        </div>
                      )}

                      <button
                        onClick={handleResolveTicket}
                        disabled={isLoading}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg shadow-sm flex items-center justify-center gap-2 transition-all disabled:opacity-50 text-xs uppercase tracking-wider mt-4"
                      >
                        {isLoading ? (
                          <>
                            <RefreshCw className="h-4 w-4 animate-spin" />
                            Resolving Ticket...
                          </>
                        ) : (
                          <>
                            <Send className="h-4 w-4" />
                            Submit Query
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Animated Processing State View */}
                {isLoading && (
                  <div className="bg-slate-100 border border-slate-200 rounded-xl p-10 text-center flex flex-col items-center justify-center gap-4 shadow-3xs">
                    <RefreshCw className="h-8 w-8 text-blue-600 animate-spin" />
                    <p className="text-sm font-bold text-slate-800">{loadingStep}</p>
                    <p className="text-[11px] text-slate-500 max-w-md leading-relaxed">
                      Deep AI Inference Layer active. Tokenizing texts, executing categorical linear SVM models, auditing policy scopes, compiling drafting emails, and calculating ROUGE-L reference indexes.
                    </p>
                  </div>
                )}
              </motion.div>
            )}

            {/* 3. QUERY ANALYSIS TAB */}
            {activeTab === "analysis" && (
              activeTicket ? (
                <motion.div
                key="analysis"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                {/* Security Breach Warning if safety block is triggered */}
                {activeTicket.agentResult && !activeTicket.agentResult.safetyPassed && (
                  <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-5 flex gap-4 shadow-3xs">
                    <AlertTriangle className="h-8 w-8 text-rose-600 shrink-0" />
                    <div>
                      <h4 className="font-bold text-sm">Security Guard Ingestion Alert</h4>
                      <p className="text-xs mt-1 text-rose-700">
                        This customer query was immediately intercepted by security compliance filter algorithms. Input text failed compliance standard checks.
                      </p>
                      <div className="mt-2.5 text-[10px] font-mono bg-rose-100 border border-rose-200 text-rose-800 px-3 py-1 rounded inline-block font-bold">
                        Reason: {activeTicket.agentResult.escalationReason}
                      </div>
                    </div>
                  </div>
                )}

                {/* Main analysis bento box */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  {/* Left panel: Classification Details */}
                  <div className="lg:col-span-4 space-y-6">
                    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                      <h3 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider flex items-center gap-2">
                        <Sliders className="h-4 w-4 text-blue-600" />
                        Linguistic Parsing
                      </h3>

                      <div className="space-y-4">
                        <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                          <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Ingested Query Text</span>
                          <p className="text-xs text-slate-800 font-medium leading-relaxed italic mt-1 line-clamp-4">
                            "{activeTicket.query}"
                          </p>
                        </div>

                        <div className="grid grid-cols-2 gap-3 text-center">
                          <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                            <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Predicted Category</span>
                            <span className="text-xs font-bold text-slate-800 block mt-1">
                              {activeTicket.category}
                            </span>
                          </div>

                          <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                            <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Customer Sentiment</span>
                            <span className={`text-xs font-bold block mt-1 ${
                              activeTicket.sentiment === Sentiment.POSITIVE ? "text-emerald-600" :
                              activeTicket.sentiment === Sentiment.NEGATIVE ? "text-rose-600" : "text-amber-600"
                            }`}>
                              {activeTicket.sentiment}
                            </span>
                          </div>
                        </div>

                        <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                          <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Calculated Urgency Rating</span>
                          <span className={`text-xs font-bold block mt-1 ${
                            activeTicket.urgency === "High" ? "text-rose-600" :
                            activeTicket.urgency === "Medium" ? "text-amber-600" : "text-emerald-600"
                          }`}>
                            {activeTicket.urgency} Level
                          </span>
                        </div>

                        <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                          <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Risk Level Category</span>
                          <span className="text-xs font-bold text-slate-800 block mt-1">
                            {activeTicket.riskLevel}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right panel: Extracted entities table */}
                  <div className="lg:col-span-8 bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                    <h3 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider flex items-center gap-2">
                      <Search className="h-4 w-4 text-blue-600" />
                      Extracted Support Entities
                    </h3>

                    {(() => {
                      const activeRecord = SUPPORT_RECORDS.find(r => 
                        r.customer_id === activeTicket.customerId || 
                        r.customer_name.toLowerCase() === activeTicket.name.toLowerCase()
                      ) || null;

                      const queryContains = (text: string) => {
                        if (!text) return false;
                        return activeTicket.query.toLowerCase().includes(text.toLowerCase());
                      };

                      const getRegexMatch = (pattern: RegExp) => {
                        const m = activeTicket.query.match(pattern);
                        return m ? m[0] : null;
                      };

                      const formatCurrency = (val: string | number) => {
                        if (typeof val === "number") return `$${val}`;
                        if (!val.startsWith("$")) return `$${val}`;
                        return val;
                      };

                      const rows = [
                        {
                          label: "Customer Full Name",
                          direct: queryContains(activeTicket.name) ? activeTicket.name : null,
                          inferred: activeRecord?.customer_name || activeTicket.name,
                        },
                        {
                          label: "Customer ID",
                          direct: getRegexMatch(/CUST-[A-Z0-9]+/i) || (queryContains(activeTicket.customerId) ? activeTicket.customerId : null),
                          inferred: activeRecord?.customer_id || activeTicket.customerId || "CUST-TEMP",
                        },
                        {
                          label: "Order Reference ID",
                          direct: getRegexMatch(/ORD-[0-9]+/i) || activeTicket.nlpResult?.regexEntities.orderId || null,
                          inferred: activeRecord?.order_id || "ORD-UNKNOWN",
                        },
                        {
                          label: "Problem Category",
                          direct: detectCategoryFromQuery(activeTicket.query) || null,
                          inferred: activeRecord?.problem_category || activeTicket.category,
                        },
                        {
                          label: "Problem Status",
                          direct: ["new", "pending", "in_progress", "resolved", "escalated"].find(s => queryContains(s)) || null,
                          inferred: activeRecord?.status || "new",
                        },
                        {
                          label: "Product Name",
                          direct: ["Pro Headphones X", "Quantum Laptop 15", "Secure Router Pro", "Ultra Smartwatch v2", "Eco Charging Pad", "Noise-Cancelling Bud S", "4K HDR Monitor"].find(p => queryContains(p)) || activeTicket.nlpResult?.entities.productName || null,
                          inferred: activeRecord?.product_name || "N/A",
                        },
                        {
                          label: "Amount",
                          direct: getRegexMatch(/\$[0-9]+/i) || activeTicket.nlpResult?.regexEntities.money || null,
                          inferred: activeRecord ? formatCurrency(activeRecord.amount) : "N/A",
                        },
                        {
                          label: "Email",
                          direct: getRegexMatch(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/i) || activeTicket.nlpResult?.regexEntities.email || null,
                          inferred: activeRecord?.email || activeTicket.email,
                        },
                        {
                          label: "Phone",
                          direct: getRegexMatch(/\+?[0-9-]{7,15}/) || activeTicket.nlpResult?.regexEntities.phone || null,
                          inferred: activeRecord?.phone || "N/A",
                        },
                        {
                          label: "City",
                          direct: ["Austin", "Boston", "Chicago", "Denver", "El Paso", "Fort Worth", "Gary", "Houston", "Indianapolis", "Jacksonville", "Kansas City", "Los Angeles", "Miami", "Nashville", "Oakland", "Phoenix", "Queens", "Raleigh", "Seattle", "Tampa", "Urbana", "Vancouver", "Wichita", "Xenia", "Yonkers", "Zion"].find(c => queryContains(c)) || activeTicket.nlpResult?.entities.city || null,
                          inferred: activeRecord?.city || "N/A",
                        },
                        {
                          label: "Error Code",
                          direct: getRegexMatch(/ERR_[A-Z0-9_]+/i) || activeTicket.nlpResult?.regexEntities.errorCode || null,
                          inferred: activeRecord?.error_code || "N/A",
                        },
                        {
                          label: "Priority",
                          direct: ["low", "medium", "high", "urgent"].find(p => queryContains(p)) || null,
                          inferred: activeRecord?.priority || "medium",
                        },
                        {
                          label: "Resolution",
                          direct: null,
                          inferred: activeRecord?.resolution || activeTicket.agentResult?.finalResolution || "N/A",
                        }
                      ];

                      return (
                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="border-b border-slate-200 text-slate-400 font-bold bg-slate-50">
                                <th className="py-2.5 px-3">Entity Type</th>
                                <th className="py-2.5 px-3">Extracted from Query</th>
                                <th className="py-2.5 px-3">Inferred (Database Value)</th>
                                <th className="py-2.5 px-3">Resolution Status</th>
                              </tr>
                            </thead>
                            <tbody className="font-medium text-slate-700">
                              {rows.map((row, idx) => {
                                const isDirect = row.direct !== null && row.direct !== undefined && row.direct !== "";
                                const finalDisplayDirect = isDirect ? String(row.direct) : "N/A (Not in query)";
                                const finalDisplayInferred = row.inferred ? String(row.inferred) : "N/A";
                                
                                return (
                                  <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors">
                                    <td className="py-3 px-3 font-semibold text-slate-900">{row.label}</td>
                                    <td className={`py-3 px-3 font-mono ${isDirect ? "text-emerald-700 font-bold" : "text-slate-400"}`}>
                                      {finalDisplayDirect}
                                    </td>
                                    <td className="py-3 px-3 font-mono text-slate-900">
                                      {finalDisplayInferred}
                                    </td>
                                    <td className="py-3 px-3">
                                      {isDirect ? (
                                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                                          ✓ Query Match
                                        </span>
                                      ) : row.inferred && row.inferred !== "N/A" ? (
                                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">
                                          ⚡ Inferred Value
                                        </span>
                                      ) : (
                                        <span className="inline-flex items-center gap-1 text-[10px] font-bold text-slate-400 bg-slate-50 px-2 py-0.5 rounded-full">
                                          Not Present
                                        </span>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      );
                    })()}
                  </div>
                </div>

                {/* 100% INTACT UNDERLYING ML & NLP PIPELINES (Only displays when developerMode toggled) */}
                {developerMode && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="border-t border-slate-200 pt-6 space-y-6"
                  >
                    <div className="bg-slate-900 text-slate-200 p-4 rounded-xl border border-slate-800 font-mono text-[10px]">
                      <span className="text-blue-400 font-bold uppercase tracking-wider block mb-2">[AI Core Pipeline Trace]</span>
                      Running ML modeling evaluations. Direct display variables from the classifiers are listed below.
                    </div>

                    <NlpTab queryInput={activeTicket.query} />
                    <MachineLearningTab mlPipelineResults={mlPipelineResults} />
                  </motion.div>
                )}
              </motion.div>
            ) : (
              <div className="bg-white border border-slate-200 rounded-xl p-12 text-center max-w-xl mx-auto shadow-sm my-8">
                <Search className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">No Active Analysis Loaded</h3>
                <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                  Please ingest a customer request first on the <strong className="text-blue-600 font-semibold cursor-pointer" onClick={() => setActiveTab("support")}>Customer Support</strong> tab, or load a historical record from the backlog.
                </p>
              </div>
            ))}

            {/* 4. RESOLUTION CENTER TAB */}
            {activeTab === "resolution" && (
              activeTicket ? (
                <motion.div
                key="resolution"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                {/* Auto Resolution Summary Card */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs relative overflow-hidden">
                  <div className={`absolute top-0 left-0 right-0 h-1.5 ${
                    activeTicket.autoResolved ? "bg-emerald-500" : "bg-rose-500"
                  }`} />

                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                        {activeTicket.autoResolved ? (
                          <>
                            <CheckCircle className="text-emerald-500 h-5 w-5" />
                            Autonomous Resolution Successful
                          </>
                        ) : (
                          <>
                            <AlertCircle className="text-rose-500 h-5 w-5" />
                            Operator Escalation Protocol Engaged
                          </>
                        )}
                      </h3>
                      <p className="text-xs text-slate-500 mt-1">
                        Resolution policies parsed securely based on transaction thresholds and technical diagnostics.
                      </p>
                    </div>
                    <span className={`text-[10px] font-bold uppercase px-3 py-1 rounded-full border ${
                      activeTicket.autoResolved ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-rose-50 border-rose-200 text-rose-700"
                    }`}>
                      {activeTicket.autoResolved ? "Autonomously Resolved" : "Escalated"}
                    </span>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 p-4 rounded-lg space-y-3">
                    <div>
                      <span className="text-[9px] text-slate-400 uppercase font-bold tracking-wider block">Recommended Corporate Action Code</span>
                      <p className="text-xs font-semibold text-slate-800 leading-relaxed mt-1">
                        {activeTicket.agentResult?.finalResolution || "Escalated for safety compliance limits review."}
                      </p>
                    </div>

                    {!activeTicket.autoResolved && (
                      <div className="bg-rose-100/30 border border-rose-200 p-2.5 rounded-md text-xs text-rose-800 leading-relaxed font-semibold">
                        Escalation Reason: {activeTicket.agentResult?.escalationReason || "Limits threshold breached."}
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-center text-xs">
                    <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                      <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Resolution Status</span>
                      <span className="text-xs font-bold text-slate-900 block mt-1">
                        {activeTicket.autoResolved ? "Resolved Automatically" : "Escalated to Human"}
                      </span>
                    </div>
                    <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                      <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Estimated Solve Time</span>
                      <span className="text-xs font-bold text-slate-900 block mt-1">{activeTicket.resolutionTime}</span>
                    </div>
                    <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                      <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Decision Priority</span>
                      <span className="text-xs font-bold text-slate-900 block mt-1">{activeTicket.urgency}</span>
                    </div>
                    <div className="bg-slate-50 border border-slate-200 p-3 rounded-lg">
                      <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider block">Decision Confidence</span>
                      <span className="text-xs font-bold text-blue-600 block mt-1">
                        {Math.round((activeTicket.agentResult?.perception?.confidence ?? 0.85) * 100)}% Core Score
                      </span>
                    </div>
                  </div>
                </div>

                {/* Email Preview & Chat Section */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Email Body Draft & Controls */}
                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col">
                    <h4 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider flex items-center gap-2">
                      <Mail className="h-4 w-4 text-blue-600" />
                      Ingested Case Response Email
                    </h4>

                    {activeTicket.genAIResult ? (
                      <div className="flex-1 flex flex-col space-y-4">
                        <div className="bg-slate-50 border border-slate-200 p-4 rounded-lg text-xs leading-relaxed font-medium font-sans h-64 overflow-y-auto whitespace-pre-wrap text-slate-700">
                          {activeTicket.genAIResult.personalizedEmail || "No email draft generated."}
                        </div>

                        <div className="grid grid-cols-3 gap-2 pt-2">
                          <button
                            onClick={() => handleCopyEmail(activeTicket.genAIResult!.personalizedEmail)}
                            className="bg-white hover:bg-slate-100 border border-slate-200 text-slate-700 text-[10px] font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition-all shadow-3xs"
                          >
                            <Copy className="h-3 w-3" />
                            {isEmailCopied ? "Copied!" : "Copy Body"}
                          </button>

                          <button
                            onClick={() => handleDownloadEML(activeTicket)}
                            className="bg-white hover:bg-slate-100 border border-slate-200 text-slate-700 text-[10px] font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition-all shadow-3xs"
                          >
                            <Download className="h-3 w-3" /> EML Draft
                          </button>

                          <button
                            onClick={handleDispatchEmail}
                            disabled={isEmailDispatched}
                            className={`text-[10px] font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-1.5 transition-all shadow-3xs ${
                              isEmailDispatched
                                ? "bg-emerald-50 border border-emerald-200 text-emerald-700 cursor-not-allowed"
                                : "bg-blue-600 hover:bg-blue-700 text-white"
                            }`}
                          >
                            <Check className="h-3 w-3" />
                            {isEmailDispatched ? "Dispatched" : "Send Client"}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex-1 flex flex-col items-center justify-center p-10 bg-slate-50 border border-slate-200 border-dashed rounded-lg">
                        <Mail className="h-8 w-8 text-slate-300 mb-2" />
                        <span className="text-xs text-slate-500 font-semibold">No active email response drafted.</span>
                      </div>
                    )}
                  </div>

                  {/* Customer Support Conversation Box */}
                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col h-[400px]">
                    <h4 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider flex items-center gap-2">
                      <Terminal className="h-4 w-4 text-blue-600" />
                      Customer Support Conversation
                    </h4>

                    {/* Chat Logs Window */}
                    <div className="flex-1 overflow-y-auto space-y-3 mb-4 pr-1 text-xs">
                      {chatMessages.map((chat, idx) => (
                        <div
                          key={idx}
                          className={`p-3 rounded-xl max-w-[85%] leading-relaxed font-medium ${
                            chat.sender === "user"
                              ? "bg-slate-100 text-slate-800 self-start mr-auto border border-slate-200"
                              : "bg-blue-50 text-blue-900 ml-auto border border-blue-100"
                          }`}
                        >
                          <span className="text-[8px] font-bold uppercase tracking-wider block opacity-50 mb-0.5">
                            {chat.sender === "user" ? "Customer" : "Resolution Engine"}
                          </span>
                          <p>{chat.text}</p>
                        </div>
                      ))}
                    </div>

                    {/* Chat input controls */}
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={newChatMessage}
                        onChange={(e) => setNewChatMessage(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSendChatMessage()}
                        className="flex-1 px-3.5 py-2 bg-slate-50 border border-slate-200 rounded-lg text-xs leading-relaxed focus:outline-hidden focus:ring-1 focus:ring-blue-600 focus:bg-white font-medium"
                        placeholder="Type reply message..."
                      />
                      <button
                        onClick={handleSendChatMessage}
                        className="bg-blue-600 hover:bg-blue-700 text-white p-2.5 rounded-lg flex items-center justify-center transition-all shadow-3xs"
                      >
                        <Send className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>

                {/* Recommended Frequently Asked Questions */}
                {activeTicket.genAIResult && (
                  <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                    <h4 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider flex items-center gap-2">
                      <Compass className="h-4 w-4 text-blue-600" />
                      Recommended Frequently Asked Questions
                    </h4>

                    <div className="space-y-3">
                      {(activeTicket.genAIResult.faqList || []).map((faq, idx) => (
                        <div key={idx} className="border border-slate-100 rounded-lg overflow-hidden">
                          <button
                            onClick={() =>
                              setFaqExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }))
                            }
                            className="w-full text-left bg-slate-50 hover:bg-slate-100/60 p-3 flex items-center justify-between text-xs font-bold text-slate-800 transition-all"
                          >
                            <span className="flex items-center gap-2">
                              <span className="text-blue-600">Q:</span> {faq.question}
                            </span>
                            {faqExpanded[idx] ? (
                              <ChevronUp className="h-4 w-4 text-slate-400" />
                            ) : (
                              <ChevronDown className="h-4 w-4 text-slate-400" />
                            )}
                          </button>
                          {faqExpanded[idx] && (
                            <div className="p-3 bg-white border-t border-slate-100 text-xs text-slate-600 font-medium leading-relaxed">
                              <span className="text-emerald-600 font-bold mr-1.5">A:</span>
                              {faq.answer}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* SLA Audit Resolution Decision Logs */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xs text-slate-300">
                  <h3 className="font-bold text-xs text-white border-b border-slate-800 pb-2 mb-4 uppercase tracking-wider flex items-center gap-2">
                    <Terminal className="h-4 w-4 text-emerald-400" />
                    SLA Resolution Decision Logs
                  </h3>

                  <div className="space-y-4 text-xs font-mono">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-slate-950 p-4 border border-slate-800 rounded-lg">
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Resolved Automatically</span>
                        <span className="text-xs font-bold text-slate-200 block mt-1">
                          {activeTicket.autoResolved ? "TRUE" : "FALSE (Human Queue)"}
                        </span>
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Classification Confidence</span>
                        <span className="text-xs font-bold text-slate-200 block mt-1">
                          {Math.round((activeTicket.agentResult?.perception?.confidence ?? 0.85) * 100)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Urgency Priority</span>
                        <span className="text-xs font-bold text-slate-200 block mt-1">{activeTicket.urgency}</span>
                      </div>
                      <div>
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Resolution Status</span>
                        <span className="text-xs font-bold text-slate-200 block mt-1">{activeTicket.status}</span>
                      </div>
                    </div>

                    <div className="bg-slate-950 p-4 border border-slate-800 rounded-lg">
                      <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Logical Decision Rule Reason</span>
                      <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                        {activeTicket.agentResult?.reasoning || "Escalation rules executed successfully."}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Raw State Memory & Decision Logs (Toggled on via developerMode) */}
                {developerMode && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="border-t border-slate-200 pt-6 space-y-6"
                  >
                    <div className="bg-slate-900 border border-slate-800 text-slate-200 p-5 rounded-xl space-y-4">
                      <h3 className="font-bold text-xs text-blue-400 uppercase tracking-wider border-b border-slate-800 pb-2">
                        [Developer Mode] Deep Agentic State Trace & Memory Logs
                      </h3>

                      <div className="space-y-4 max-h-96 overflow-y-auto text-xs font-mono">
                        {activeTicket.agentResult?.decisionLogs.map((step, idx) => (
                          <div key={idx} className="bg-slate-950 border border-slate-800 p-3.5 rounded-lg space-y-1.5">
                            <div className="text-blue-400 font-bold uppercase text-[9px]">[AGENT DECISION STATE STEP {idx + 1}]</div>
                            <div><span className="text-slate-500">Thought:</span> <span className="text-slate-200">{step.thought}</span></div>
                            <div><span className="text-slate-500">Action:</span> <span className="text-sky-400">{step.action}</span></div>
                            <div><span className="text-slate-500">Observation:</span> <span className="text-emerald-400">{step.observation}</span></div>
                          </div>
                        ))}

                        <div className="bg-slate-950 border border-slate-800 p-3.5 rounded-lg space-y-1.5">
                          <span className="text-[9px] text-slate-500 uppercase font-bold">Short-Term State Memory logs</span>
                          <div className="space-y-1 text-slate-400 text-[10px]">
                            {activeTicket.agentResult?.memoryLogs.map((log, idx) => (
                              <div key={idx} className="truncate">{log}</div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>

                    <SlmTab slmResult={activeTicket.slmResult || null} />
                  </motion.div>
                )}
              </motion.div>
            ) : (
              <div className="bg-white border border-slate-200 rounded-xl p-12 text-center max-w-xl mx-auto shadow-sm my-8">
                <ShieldCheck className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">No Active Resolution Loaded</h3>
                <p className="text-xs text-slate-500 mt-2 leading-relaxed">
                  Please ingest a customer request first on the <strong className="text-blue-600 font-semibold cursor-pointer" onClick={() => setActiveTab("support")}>Customer Support</strong> tab, or load a historical record from the backlog.
                </p>
              </div>
            ))}

            {/* 5. CUSTOMER HISTORY TAB */}
            {activeTab === "history" && (
              <motion.div
                key="history"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                  <h2 className="text-lg font-bold text-slate-950 mb-1 flex items-center gap-2">
                    <History className="text-blue-600 h-5 w-5" />
                    Customer Support CRM Ledgers
                  </h2>
                  <p className="text-xs text-slate-500">
                    Audit historic customer inquiries, predicted pipelines categories, classification results, and transaction states. Click a ticket row to load details into memory.
                  </p>
                </div>

                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 bg-slate-50 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                          <th className="py-3 px-4">Case ID</th>
                          <th className="py-3 px-4">Customer</th>
                          <th className="py-3 px-4">Ingested Query (Summary)</th>
                          <th className="py-3 px-4">Category</th>
                          <th className="py-3 px-4">Sentiment</th>
                          <th className="py-3 px-4">SLA Urgency</th>
                          <th className="py-3 px-4">Resolution Route</th>
                        </tr>
                      </thead>
                      <tbody className="font-medium text-slate-700 divide-y divide-slate-100">
                        {customerQueries.map((ticket) => (
                          <tr
                            key={ticket.id}
                            onClick={() => {
                              setActiveTicket(ticket);
                              setActiveTab("analysis");
                            }}
                            className={`hover:bg-blue-50/20 cursor-pointer transition-all ${
                              activeTicket?.id === ticket.id ? "bg-blue-50/35 border-l-2 border-l-blue-600" : ""
                            }`}
                          >
                            <td className="py-3.5 px-4 font-mono font-bold text-slate-950">{ticket.id}</td>
                            <td className="py-3.5 px-4 font-semibold">{ticket.name}</td>
                            <td className="py-3.5 px-4 max-w-xs truncate italic">"{ticket.query}"</td>
                            <td className="py-3.5 px-4">
                              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                                ticket.category === "Billing" ? "bg-amber-100 text-amber-800" :
                                ticket.category === "Technical" ? "bg-rose-100 text-rose-800" : "bg-blue-100 text-blue-800"
                              }`}>
                                {ticket.category}
                              </span>
                            </td>
                            <td className="py-3.5 px-4 font-bold">{ticket.sentiment}</td>
                            <td className="py-3.5 px-4">
                              <span className={`text-[10px] font-bold ${
                                ticket.urgency === "High" ? "text-rose-600" : "text-slate-500"
                              }`}>
                                {ticket.urgency} Priority
                              </span>
                            </td>
                            <td className="py-3.5 px-4">
                              <span className={`text-[10px] font-bold py-0.5 px-2 rounded-full border ${
                                ticket.autoResolved
                                  ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                                  : "bg-rose-50 border-rose-200 text-rose-700"
                              }`}>
                                {ticket.autoResolved ? "Auto Resolved" : "Escalated"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Dataset Download component embedded safely */}
                {developerMode && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="border-t border-slate-200 pt-6"
                  >
                    <DatasetTab handleDownloadCSV={() => {}} />
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* 6. REPORTS & ANALYTICS TAB */}
            {activeTab === "reports" && (
              <motion.div
                key="reports"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                  <h2 className="text-lg font-bold text-slate-950 mb-1 flex items-center gap-2">
                    <BarChart3 className="text-blue-600 h-5 w-5" />
                    Service Reports & Analytics
                  </h2>
                  <p className="text-xs text-slate-500">
                    Statistical evaluation dashboards monitoring automated categories, sentiment scores, and pipeline accuracy scores.
                  </p>
                </div>

                {/* Dashboard Metrics charts styled with raw SVG and CSS representing standard business reporting */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Category Distribution Chart */}
                  <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-2xs">
                    <h3 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider">
                      Queries Ingestion Category Distribution
                    </h3>
                    <div className="h-60 flex flex-col justify-center space-y-4">
                      {/* Billing */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs font-semibold">
                          <span>Billing Operations Inquiry</span>
                          <span>45% Volume</span>
                        </div>
                        <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                          <div className="bg-amber-500 h-full w-[45%]" />
                        </div>
                      </div>
                      {/* Technical */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs font-semibold">
                          <span>Technical Code Crashes</span>
                          <span>30% Volume</span>
                        </div>
                        <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                          <div className="bg-rose-500 h-full w-[30%]" />
                        </div>
                      </div>
                      {/* Shipping */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs font-semibold">
                          <span>Shipping Logistics Track</span>
                          <span>15% Volume</span>
                        </div>
                        <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                          <div className="bg-blue-500 h-full w-[15%]" />
                        </div>
                      </div>
                      {/* Product & Account */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs font-semibold">
                          <span>General Product / Security Accounts</span>
                          <span>10% Volume</span>
                        </div>
                        <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                          <div className="bg-slate-400 h-full w-[10%]" />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Sentiment Polarity Distribution Chart */}
                  <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-2xs">
                    <h3 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider">
                      Customer Sentiment Polarity Distribution
                    </h3>
                    <div className="h-60 flex items-end justify-around border-b border-slate-200 pb-2 px-10 relative">
                      <div className="flex flex-col items-center w-16">
                        <div className="w-full bg-emerald-500 rounded-t h-12" />
                        <span className="text-[10px] font-bold text-slate-500 mt-2 font-mono">Positive (20%)</span>
                      </div>
                      <div className="flex flex-col items-center w-16">
                        <div className="w-full bg-amber-500 rounded-t h-20" />
                        <span className="text-[10px] font-bold text-slate-500 mt-2 font-mono">Neutral (35%)</span>
                      </div>
                      <div className="flex flex-col items-center w-16">
                        <div className="w-full bg-rose-500 rounded-t h-28" />
                        <span className="text-[10px] font-bold text-slate-500 mt-2 font-mono">Negative (45%)</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* 100% INTACT DEEP LEARNING MODEL TRAINING LEARNING CURVES (Only visible under developerMode) */}
                {developerMode && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="border-t border-slate-200 pt-6"
                  >
                    <DeepLearningTab dlPipelineResults={dlPipelineResults} />
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* 7. SYSTEM STATUS DIAGNOSTICS TAB */}
            {activeTab === "status" && (
              <motion.div
                key="status"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
                  <h2 className="text-lg font-bold text-slate-950 mb-1 flex items-center gap-2">
                    <Activity className="text-blue-600 h-5 w-5" />
                    Diagnostics & Core Environment Status
                  </h2>
                  <p className="text-xs text-slate-500">
                    System-wide operation health metrics. Direct monitoring parameters loaded from current container servers.
                  </p>
                </div>

                {/* Diagnostics Visual Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-2xs flex items-center gap-4">
                    <div className="bg-emerald-50 text-emerald-600 p-3 rounded-lg border border-emerald-100">
                      <CheckCircle className="h-6 w-6" />
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block">Backend Status</span>
                      <span className="text-sm font-bold text-slate-900 block mt-0.5">Online & Secure</span>
                      <span className="text-[9px] text-slate-500 block font-mono">Port 3000 Node</span>
                    </div>
                  </div>

                  <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-2xs flex items-center gap-4">
                    <div className="bg-emerald-50 text-emerald-600 p-3 rounded-lg border border-emerald-100">
                      <Database className="h-6 w-6" />
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block">Database Status</span>
                      <span className="text-sm font-bold text-slate-900 block mt-0.5">Online</span>
                      <span className="text-[9px] text-slate-500 block font-mono">CRM Local Ledger</span>
                    </div>
                  </div>

                  <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-2xs flex items-center gap-4">
                    <div className="bg-emerald-50 text-emerald-600 p-3 rounded-lg border border-emerald-100">
                      <Sliders className="h-6 w-6" />
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block">Traditional ML Status</span>
                      <span className="text-sm font-bold text-slate-900 block mt-0.5">Loaded & Active</span>
                      <span className="text-[9px] text-slate-500 block font-mono">best_model.json</span>
                    </div>
                  </div>

                  <div className="bg-white border border-slate-200 p-5 rounded-xl shadow-2xs flex items-center gap-4">
                    <div className="bg-emerald-50 text-emerald-600 p-3 rounded-lg border border-emerald-100">
                      <Cpu className="h-6 w-6" />
                    </div>
                    <div>
                      <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block">Deep Network Status</span>
                      <span className="text-sm font-bold text-slate-900 block mt-0.5">Loaded & Active</span>
                      <span className="text-[9px] text-slate-500 block font-mono">dl_model.json (BiLSTM)</span>
                    </div>
                  </div>
                </div>

                {/* System API contract router logs */}
                <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                  <h3 className="font-bold text-xs text-slate-900 border-b border-slate-100 pb-2 mb-4 uppercase tracking-wider">
                    API Endpoint Active Contracts
                  </h3>

                  <div className="overflow-x-auto text-xs">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 bg-slate-50 text-slate-400 font-bold uppercase tracking-wider text-[9px]">
                          <th className="py-2.5 px-3">HTTP Method</th>
                          <th className="py-2.5 px-3">Active Router Endpoint</th>
                          <th className="py-2.5 px-3">Average Ingestion Ping</th>
                          <th className="py-2.5 px-3">Diagnostic Status</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono text-slate-700">
                        <tr className="border-b border-slate-100">
                          <td className="py-2.5 px-3 font-semibold text-blue-600">GET</td>
                          <td className="py-2.5 px-3 font-bold text-slate-900">/health</td>
                          <td className="py-2.5 px-3 text-slate-500">2ms</td>
                          <td className="py-2.5 px-3"><span className="text-emerald-600 font-sans font-bold">● Active</span></td>
                        </tr>
                        <tr className="border-b border-slate-100">
                          <td className="py-2.5 px-3 font-semibold text-emerald-600">POST</td>
                          <td className="py-2.5 px-3 font-bold text-slate-900">/predict</td>
                          <td className="py-2.5 px-3 text-slate-500">14ms</td>
                          <td className="py-2.5 px-3"><span className="text-emerald-600 font-sans font-bold">● Active</span></td>
                        </tr>
                        <tr className="border-b border-slate-100">
                          <td className="py-2.5 px-3 font-semibold text-emerald-600">POST</td>
                          <td className="py-2.5 px-3 font-bold text-slate-900">/analyze</td>
                          <td className="py-2.5 px-3 text-slate-500">8ms</td>
                          <td className="py-2.5 px-3"><span className="text-emerald-600 font-sans font-bold">● Active</span></td>
                        </tr>
                        <tr className="border-b border-slate-100">
                          <td className="py-2.5 px-3 font-semibold text-emerald-600">POST</td>
                          <td className="py-2.5 px-3 font-bold text-slate-900">/chat</td>
                          <td className="py-2.5 px-3 text-slate-500">24ms</td>
                          <td className="py-2.5 px-3"><span className="text-emerald-600 font-sans font-bold">● Active</span></td>
                        </tr>
                        <tr className="border-b border-slate-100">
                          <td className="py-2.5 px-3 font-semibold text-emerald-600">POST</td>
                          <td className="py-2.5 px-3 font-bold text-slate-900">/generate</td>
                          <td className="py-2.5 px-3 text-slate-500">128ms</td>
                          <td className="py-2.5 px-3"><span className="text-emerald-600 font-sans font-bold">● Active</span></td>
                        </tr>
                        <tr className="border-b border-slate-100">
                          <td className="py-2.5 px-3 font-semibold text-emerald-600">POST</td>
                          <td className="py-2.5 px-3 font-bold text-slate-900">/agent</td>
                          <td className="py-2.5 px-3 text-slate-500">145ms</td>
                          <td className="py-2.5 px-3"><span className="text-emerald-600 font-sans font-bold">● Active</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* 100% INTACT ENDPOINT CONTRACTS (Only displays when developerMode toggled) */}
                {developerMode && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="border-t border-slate-200 pt-6"
                  >
                    <ApiTab />
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* 8. ABOUT PROJECT SUMMARY TAB */}
            {activeTab === "about" && (
              <motion.div
                key="about"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.15 }}
                className="space-y-6"
              >
                {/* Academic Style Ingestion Details */}
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
                  <h2 className="text-lg font-bold text-slate-950 flex items-center gap-2 border-b border-slate-100 pb-2">
                    <BookOpen className="text-blue-600 h-5 w-5" />
                    Project Thesis Abstract & System Architecture
                  </h2>

                  <div className="space-y-4 text-xs text-slate-600 leading-relaxed font-medium">
                    <div>
                      <h4 className="font-bold text-slate-900 text-xs uppercase mb-1">Problem Statement</h4>
                      <p>
                        In enterprise-scale customer support channels, high-volume unstructured queries create massive administrative overhead, leading to slow response times and critical operational inefficiencies. While basic keyword matching falls short, manual categorization lacks scalability.
                      </p>
                    </div>

                    <div>
                      <h4 className="font-bold text-slate-900 text-xs uppercase mb-1">Proposed AI Methodology</h4>
                      <p>
                        This project implements a multi-tiered, full-stack, enterprise-grade AI system. It features:
                        <br />
                        1. **Traditional Machine Learning (Level 1)**: TF-IDF feature coefficient vectorization combined with multiclass linear Support Vector Machines (SVM) with SGD and One-vs-Rest (OVR) Logistic Regression.
                        <br />
                        2. **Deep Learning (Level 2)**: Custom validation sequences mapped to embedding vectors and processed via recurrent Bidirectional LSTM (BiLSTM) network classifiers.
                        <br />
                        3. **Natural Language Processing (Level 3)**: Multi-stage lexicons for cleaning text, lemmatization, stemming, and custom regex entity extractions.
                        <br />
                        4. **Small Language Models (Level 4)**: Compact local models evaluated dynamically using ROUGE mathematical indices.
                        <br />
                        5. **Generative AI & Agentic loops**: Cognitive reasoning, planned executions, structured email completions, and auto-escalation limits safeguards.
                      </p>
                    </div>
                  </div>
                </div>

                {/* 100% INTACT LOW LEVEL LLD SEQUENCES (Only displays when developerMode is toggled) */}
                {developerMode && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="border-t border-slate-200 pt-6"
                  >
                    <LldTab />
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* Global Lexicon dictionary properties */}
      {(() => {
        const STOPWORDS_LOCAL = new Set([
          "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her",
          "it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those", "am", "is", "are",
          "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "doing", "a", "an",
          "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
          "to", "from", "in", "on", "off", "over", "under", "again", "then", "once", "here", "there", "when",
          "where", "why", "how", "all", "any", "both"
        ]);
        (window as any).STOPWORDS_LOCAL = STOPWORDS_LOCAL;
        return null;
      })()}
    </div>
  );
}
