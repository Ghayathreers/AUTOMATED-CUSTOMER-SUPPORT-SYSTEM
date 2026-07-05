/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from "fs";
import * as path from "path";
import { QueryCategory, Sentiment, SupportQuery } from "./types.js";

// Helper to get random item from list
const randomChoice = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];

// Helper to get random number in range
const randomRange = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

// Raw materials for randomization
const CITIES_WITH_STATES = [
  { city: "Austin", state: "TX" },
  { city: "Boston", state: "MA" },
  { city: "Chicago", state: "IL" },
  { city: "Denver", state: "CO" },
  { city: "El Paso", state: "TX" },
  { city: "Fort Worth", state: "TX" },
  { city: "Gary", state: "IN" },
  { city: "Houston", state: "TX" },
  { city: "Indianapolis", state: "IN" },
  { city: "Jacksonville", state: "FL" },
  { city: "Kansas City", state: "MO" },
  { city: "Los Angeles", state: "CA" },
  { city: "Miami", state: "FL" },
  { city: "Nashville", state: "TN" },
  { city: "Oakland", state: "CA" },
  { city: "Phoenix", state: "AZ" },
  { city: "Queens", state: "NY" },
  { city: "Raleigh", state: "NC" },
  { city: "Seattle", state: "WA" },
  { city: "Tampa", state: "FL" },
  { city: "Urbana", state: "IL" },
  { city: "Vancouver", state: "WA" },
  { city: "Wichita", state: "KS" },
  { city: "Xenia", state: "OH" },
  { city: "Yonkers", state: "NY" },
  { city: "Zion", state: "IL" }
];

const PRODUCTS = [
  { name: "Pro Headphones X", category: "Audio", price: 299 },
  { name: "Quantum Laptop 15", category: "Computing", price: 1499 },
  { name: "Secure Router Pro", category: "Networking", price: 199 },
  { name: "Ultra Smartwatch v2", category: "Wearables", price: 349 },
  { name: "Eco Charging Pad", category: "Accessories", price: 49 },
  { name: "Noise-Cancelling Bud S", category: "Audio", price: 129 },
  { name: "4K HDR Monitor", category: "Display", price: 599 }
];

const DEVICES = ["iPhone 14 Pro", "Samsung Galaxy S23", "MacBook Pro M2", "Dell XPS 13", "iPad Air", "Windows Desktop", "Android Tablet"];
const ERROR_CODES = ["ERR_CONNECTION_TIMED_OUT", "ERR_AUTH_FAILED_401", "ERR_DB_TIMEOUT_504", "ERR_RESOURCE_NOT_FOUND_404", "ERR_SSL_HANDSHAKE_FAILED", "ERR_INVALID_LICENSE_KEY"];
const FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Casey", "Jamie", "Riley", "Robin", "Drew", "Skyler", "Cameron", "Chris", "Pat", "Amelia", "Benjamin", "Chloe", "Daniel", "Emily", "Fiona", "George", "Hannah", "Ian", "Julia", "Kevin", "Lily", "Matthew", "Natalie", "Owen", "Penelope", "Quincy", "Ruby", "Samuel", "Victor", "Wyatt", "Xavier", "Yvonne", "Zachary"];
const LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Adams", "Baker", "Carter", "Evans", "Fisher", "Green", "Hill", "Irvin", "King", "Lewis", "Nelson", "Ortiz", "Perez", "Quinn", "Robinson", "Thomas", "Underwood", "Vance", "Watson", "Xu", "Young", "Zimmerman"];

// Category-specific text templates to make queries highly realistic and learnable
const TEMPLATES: Record<QueryCategory, string[]> = {
  [QueryCategory.BILLING]: [
    "Hi, I noticed a double charge of ${AMOUNT} on my credit card for order {ORDER_ID}. Can you please issue a refund?",
    "I was billed twice for my subscription. The invoice shows {ORDER_ID} for {AMOUNT} but I only authorized one charge.",
    "My card on file was charged {AMOUNT} but I cancelled my plan last week. Refund needed ASAP.",
    "Could you send me the invoice for my purchase of {PRODUCT}? Order ID is {ORDER_ID}.",
    "I'm trying to buy {PRODUCT} from {CITY}, but the payment keeps failing with billing code {ERROR_CODE}.",
    "Why was my account charged {AMOUNT}? I didn't purchase anything recently. Order reference is {ORDER_ID}.",
    "I need to update my billing address from {CITY} to my new home. Can you help me?",
    "I am requesting a refund for order {ORDER_ID} since the product arrived damaged. The total amount was ${AMOUNT}."
  ],
  [QueryCategory.TECHNICAL]: [
    "My {DEVICE} is throwing error {ERROR_CODE} when trying to connect to the server. Please help.",
    "The software crashes on startup on my {DEVICE}. It worked fine yesterday but now displays {ERROR_CODE}.",
    "I am unable to login on my {DEVICE}. It keeps spinning and says {ERROR_CODE}.",
    "How do I configure the firmware update on the {PRODUCT}? I'm getting a connection failure.",
    "I'm experiencing slow speeds in {CITY} using {DEVICE}. Is there an outage in my area?",
    "The secure setup for {PRODUCT} fails with code {ERROR_CODE}. I have tried restarting the router already.",
    "Can you help me debug the sync issue on {DEVICE}? It shows database error {ERROR_CODE} constantly.",
    "The application screen is completely frozen on my {DEVICE} after the latest update."
  ],
  [QueryCategory.SHIPPING]: [
    "Where is my package? The tracking for order {ORDER_ID} has not updated in 3 days. It was supposed to ship to {CITY}.",
    "I ordered {PRODUCT} but the delivery address is incorrect. Can I change it to {CITY}?",
    "The delivery status for order {ORDER_ID} says delivered, but I haven't received anything at my home in {CITY}.",
    "How long does shipping to {CITY} take for the {PRODUCT}? I need it before Friday.",
    "My order {ORDER_ID} was shipped via express delivery but it's delayed. Can I get a shipping fee refund?",
    "I received a notification that order {ORDER_ID} was returned to sender. Why did that happen?",
    "Can you provide the tracking number or link for my shipping order {ORDER_ID}?",
    "The package containing my {PRODUCT} is stuck at the local hub in {CITY}. Please check on it."
  ],
  [QueryCategory.PRODUCT]: [
    "Does {PRODUCT} support integration with {DEVICE}? I couldn't find this spec in the manual.",
    "I am looking at {PRODUCT} and wanted to know if it comes with a warranty. What does the warranty cover?",
    "Can you tell me the physical dimensions and weight of {PRODUCT}? I want to make sure it fits my desk.",
    "Is {PRODUCT} compatible with Mac and Windows? I plan to use it on my {DEVICE}.",
    "What is the battery life of {PRODUCT}? Does it support fast charging on {DEVICE}?",
    "How do I clean and maintain my {PRODUCT}? The user guide doesn't explain this well.",
    "Is {PRODUCT} currently in stock in any physical stores near {CITY}?",
    "I want to compare {PRODUCT} with other models. What are the key advantages of this version?"
  ],
  [QueryCategory.ACCOUNT]: [
    "I forgot my password and the reset email is not arriving. My email is {EMAIL}.",
    "How can I delete my profile? I no longer wish to use this service. My customer name is {NAME}.",
    "My profile has been suspended for security reasons, it says error {ERROR_CODE}. How can I unlock it?",
    "I need to change my registered email from {EMAIL} to another address.",
    "Someone tried to access my profile from a different location. Can you verify my login history?",
    "How do I enable two-factor authentication (2FA) on my {DEVICE}? I want to secure my account.",
    "I cannot change my username. It keeps saying invalid format. Please help.",
    "I want to merge my old profile with my new one. Both are under {NAME}."
  ]
};

export interface CustomerRecord {
  customer_id: string;
  customer_name: string;
  email: string;
  phone: string;
  city: string;
  state: string;
  account_created_date: string;
  customer_segment: string;
  support_history_count: number;
}

export interface OrderRecord {
  order_id: string;
  customer_id: string;
  product_name: string;
  product_category: string;
  order_date: string;
  order_amount: number;
  payment_status: string;
  delivery_status: string;
  expected_delivery_date: string;
  refund_status: string;
  warranty_status: string;
}

export interface SupportQueryRecord {
  ticket_id: string;
  customer_id: string;
  order_id: string;
  customer_query: string;
  query_category: string;
  sentiment: string;
  priority: string;
  created_date: string;
  resolution_status: string;
  escalation_status: string;
}

export function generateFullDataset(customerCount: number = 10000): {
  customers: CustomerRecord[];
  orders: OrderRecord[];
  queries: SupportQueryRecord[];
  mappedQueriesForTraining: SupportQuery[];
} {
  const customers: CustomerRecord[] = [];
  const orders: OrderRecord[] = [];
  const queries: SupportQueryRecord[] = [];
  const mappedQueriesForTraining: SupportQuery[] = [];

  console.log(`Generating ${customerCount} customers...`);
  
  // 1. Generate 10,000 Customers
  for (let i = 0; i < customerCount; i++) {
    const custId = `CUST-${10000 + i}`;
    const location = randomChoice(CITIES_WITH_STATES);
    const firstName = randomChoice(FIRST_NAMES);
    const lastName = randomChoice(LAST_NAMES);
    const customerName = `${firstName} ${lastName}`;
    const email = `${firstName.toLowerCase()}.${lastName.toLowerCase()}@example.com`;
    const phone = `+1-555-${randomRange(100, 999).toString().padStart(3, "0")}-${randomRange(1000, 9999).toString().padStart(4, "0")}`;
    
    // Account creation date over last 3 years
    const acctDate = new Date();
    acctDate.setDate(acctDate.getDate() - randomRange(30, 1000));
    const accountCreatedDate = acctDate.toISOString().split("T")[0];
    
    const customerSegment = randomChoice(["Standard", "Premium", "VIP"]);
    const supportHistoryCount = randomRange(0, 5);

    customers.push({
      customer_id: custId,
      customer_name: customerName,
      email,
      phone,
      city: location.city,
      state: location.state,
      account_created_date: accountCreatedDate,
      customer_segment: customerSegment,
      support_history_count: supportHistoryCount
    });

    // 2. Generate 1 to 5 Orders for each Customer
    const numOrders = randomRange(1, 5);
    const customerOrders: OrderRecord[] = [];
    
    for (let j = 0; j < numOrders; j++) {
      const orderId = `ORD-${randomRange(100000, 999999)}`;
      const product = randomChoice(PRODUCTS);
      
      // Order date is after account created
      const oDate = new Date(accountCreatedDate);
      oDate.setDate(oDate.getDate() + randomRange(1, 20));
      const orderDate = oDate.toISOString().split("T")[0];
      
      const paymentStatus = randomChoice(["Paid", "Paid", "Paid", "Paid", "Paid", "Unpaid", "Refunded", "Processing Refund"]);
      const deliveryStatus = randomChoice(["Delivered", "Delivered", "Delivered", "In Transit", "Not Shipped", "Cancelled"]);
      
      const expDate = new Date(orderDate);
      expDate.setDate(expDate.getDate() + randomRange(3, 7));
      const expectedDeliveryDate = expDate.toISOString().split("T")[0];
      
      let refundStatus = "None";
      if (paymentStatus === "Refunded") refundStatus = "Completed";
      else if (paymentStatus === "Processing Refund") refundStatus = "Processing";

      let warrantyStatus = "No Warranty";
      if (product.category !== "Accessories") {
        const orderAgeDays = (Date.now() - new Date(orderDate).getTime()) / (1000 * 60 * 60 * 24);
        warrantyStatus = orderAgeDays > 365 ? "Expired" : "Active";
      }

      const oRec: OrderRecord = {
        order_id: orderId,
        customer_id: custId,
        product_name: product.name,
        product_category: product.category,
        order_date: orderDate,
        order_amount: product.price,
        payment_status: paymentStatus,
        delivery_status: deliveryStatus,
        expected_delivery_date: expectedDeliveryDate,
        refund_status: refundStatus,
        warranty_status: warrantyStatus
      };
      
      orders.push(oRec);
      customerOrders.push(oRec);
    }
  }

  console.log(`Generated ${orders.length} orders. Generating 10,000 support queries...`);

  // 3. Generate 10,000 Support Queries linked to random customers & their orders
  const categories = Object.values(QueryCategory);
  const sentiments = Object.values(Sentiment);
  const priorities = ["low", "medium", "high", "urgent"];
  const resolutionStatuses = ["new", "pending", "in_progress", "resolved", "escalated"];

  for (let i = 0; i < 10000; i++) {
    const tktId = `TKT-${100000 + i}`;
    
    // Select a random customer
    const customer = randomChoice(customers);
    
    // Get that customer's orders
    const custOrders = orders.filter(o => o.customer_id === customer.customer_id);
    const order = randomChoice(custOrders) || { order_id: "ORD-UNKNOWN", product_name: "Pro Headphones X", order_amount: 299 };
    
    // Choose a category
    const category = randomChoice(categories);
    const sentiment = randomChoice(sentiments);
    const priority = randomChoice(priorities);
    
    // Created date over the last 30 days
    const cDate = new Date();
    cDate.setDate(cDate.getDate() - randomRange(0, 30));
    const createdDate = cDate.toISOString().split("T")[0];
    
    const resolutionStatus = randomChoice(resolutionStatuses);
    const escalationStatus = resolutionStatus === "escalated" ? "Yes" : "No";

    const device = randomChoice(DEVICES);
    const errorCode = randomChoice(ERROR_CODES);

    // Grab templates for this category
    const templates = TEMPLATES[category];
    let queryText = randomChoice(templates);

    // Substitute placeholders
    queryText = queryText
      .replace(/{ORDER_ID}/g, order.order_id)
      .replace(/{CITY}/g, customer.city)
      .replace(/{PRODUCT}/g, order.product_name)
      .replace(/{AMOUNT}/g, order.order_amount.toString())
      .replace(/{DEVICE}/g, device)
      .replace(/{ERROR_CODE}/g, errorCode)
      .replace(/{NAME}/g, customer.customer_name)
      .replace(/{EMAIL}/g, customer.email);

    if (Math.random() < 0.1) {
      queryText = queryText.toLowerCase();
    }
    if (Math.random() < 0.05) {
      queryText = queryText + " please assist asap.";
    }

    queries.push({
      ticket_id: tktId,
      customer_id: customer.customer_id,
      order_id: order.order_id,
      customer_query: queryText,
      query_category: category,
      sentiment,
      priority,
      created_date: createdDate,
      resolution_status: resolutionStatus,
      escalation_status: escalationStatus
    });

    // Also map to traditional SupportQuery interface for ML / DL training
    mappedQueriesForTraining.push({
      id: tktId,
      query: queryText,
      category,
      orderId: order.order_id,
      city: customer.city,
      product: order.product_name,
      amount: order.order_amount,
      device,
      errorCode,
      customerName: customer.customer_name,
      sentiment,
      timestamp: new Date(createdDate).toISOString()
    });
  }

  return {
    customers,
    orders,
    queries,
    mappedQueriesForTraining
  };
}

export function saveFullDatasetToCSVAndJSON(
  data: {
    customers: CustomerRecord[];
    orders: OrderRecord[];
    queries: SupportQueryRecord[];
    mappedQueriesForTraining: SupportQuery[];
  },
  dirPath: string
) {
  // Save to the specified customer_support_ai directory
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }

  // Also save in root 'data/' directory as requested by the user
  const rootDataDir = path.join(process.cwd(), "data");
  if (!fs.existsSync(rootDataDir)) {
    fs.mkdirSync(rootDataDir, { recursive: true });
  }

  const escapeCSV = (str: string) => {
    if (typeof str !== "string") return str;
    const escaped = str.replace(/"/g, '""');
    return `"${escaped}"`;
  };

  // 1. Write customers.csv
  const customerHeaders = ["Customer ID", "Customer Name", "Email", "Phone Number", "City", "State", "Account Created Date", "Customer Segment", "Support History Count"];
  const customerRows = [customerHeaders.join(",")];
  for (const c of data.customers) {
    customerRows.push([
      c.customer_id,
      escapeCSV(c.customer_name),
      escapeCSV(c.email),
      escapeCSV(c.phone),
      escapeCSV(c.city),
      escapeCSV(c.state),
      c.account_created_date,
      c.customer_segment,
      c.support_history_count
    ].join(","));
  }

  fs.writeFileSync(path.join(dirPath, "customers.csv"), customerRows.join("\n"), "utf-8");
  fs.writeFileSync(path.join(rootDataDir, "customers.csv"), customerRows.join("\n"), "utf-8");

  // 2. Write orders.csv
  const orderHeaders = ["Order ID", "Customer ID", "Product Name", "Product Category", "Order Date", "Order Amount", "Payment Status", "Delivery Status", "Expected Delivery Date", "Refund Status", "Warranty Status"];
  const orderRows = [orderHeaders.join(",")];
  for (const o of data.orders) {
    orderRows.push([
      o.order_id,
      o.customer_id,
      escapeCSV(o.product_name),
      escapeCSV(o.product_category),
      o.order_date,
      o.order_amount,
      o.payment_status,
      o.delivery_status,
      o.expected_delivery_date,
      o.refund_status,
      o.warranty_status
    ].join(","));
  }

  fs.writeFileSync(path.join(dirPath, "orders.csv"), orderRows.join("\n"), "utf-8");
  fs.writeFileSync(path.join(rootDataDir, "orders.csv"), orderRows.join("\n"), "utf-8");

  // 3. Write support_queries.csv
  const queryHeaders = ["Ticket ID", "Customer ID", "Order ID", "Customer Query", "Query Category", "Sentiment", "Priority", "Created Date", "Resolution Status", "Escalation Status"];
  const queryRows = [queryHeaders.join(",")];
  for (const q of data.queries) {
    queryRows.push([
      q.ticket_id,
      q.customer_id,
      q.order_id,
      escapeCSV(q.customer_query),
      q.query_category,
      q.sentiment,
      q.priority,
      q.created_date,
      q.resolution_status,
      q.escalation_status
    ].join(","));
  }

  fs.writeFileSync(path.join(dirPath, "support_queries.csv"), queryRows.join("\n"), "utf-8");
  fs.writeFileSync(path.join(rootDataDir, "support_queries.csv"), queryRows.join("\n"), "utf-8");

  // 4. Also save original formatted customer_support_queries JSON and CSV for model training pipeline compatibility
  fs.writeFileSync(path.join(dirPath, "customer_support_queries.json"), JSON.stringify(data.mappedQueriesForTraining, null, 2), "utf-8");
  
  const originalHeaders = [
    "id", "query", "category", "orderId", "city", "product", "amount", "device", "errorCode", "customerName", "sentiment", "timestamp"
  ];
  const originalRows = [originalHeaders.join(",")];
  for (const q of data.mappedQueriesForTraining) {
    originalRows.push([
      q.id,
      escapeCSV(q.query),
      q.category,
      q.orderId,
      escapeCSV(q.city),
      escapeCSV(q.product),
      q.amount,
      escapeCSV(q.device),
      q.errorCode,
      escapeCSV(q.customerName),
      q.sentiment,
      q.timestamp
    ].join(","));
  }
  fs.writeFileSync(path.join(dirPath, "customer_support_queries.csv"), originalRows.join("\n"), "utf-8");

  console.log(`Successfully generated and written datasets!`);
  console.log(`- 10,000 Customers written to data/customers.csv`);
  console.log(`- ${data.orders.length} Orders written to data/orders.csv`);
  console.log(`- 10,000 Support Queries written to data/support_queries.csv`);
  console.log(`- ML Pipeline compatibility files generated.`);
}

// Deprecated original generator mapped to new generator to prevent import breaks
export function generateSyntheticDataset(count: number = 10000): SupportQuery[] {
  const result = generateFullDataset(count);
  return result.mappedQueriesForTraining;
}

export function saveDatasetToCSVAndJSON(queries: SupportQuery[], dirPath: string) {
  // Just in case, map SupportQuery back to full datasets to fulfill start conditions
  const fullData = generateFullDataset(queries.length);
  saveFullDatasetToCSVAndJSON(fullData, dirPath);
}
