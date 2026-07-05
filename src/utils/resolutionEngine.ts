import { Customer, Order } from "../data/customerDataset";
import { QueryCategory, Sentiment, NLPResult, AgenticResult, GenAIResult, SLMResult } from "../../customer_support_ai/src/types";

export interface ResolutionEngineOutput {
  category: QueryCategory;
  sentiment: Sentiment;
  urgency: "Low" | "Medium" | "High";
  riskLevel: string;
  autoResolved: boolean;
  resolutionTime: string;
  statusText: string;
  finalResolution: string;
  personalizedEmail: string;
  faqList: { question: string; answer: string }[];
  troubleshootingGuide: string;
  decisionLogs: any[];
  memoryLogs: string[];
}

export function resolveCustomerQuery(
  query: string,
  customerName: string,
  customerEmail: string,
  customerId: string,
  categoryOverride: string,
  selectedOrder: Order | null,
  apiCategory?: string,
  apiSentiment?: string
): ResolutionEngineOutput {
  // 1. Detect Category
  let category = QueryCategory.TECHNICAL;
  const qLower = query.toLowerCase();

  const hasReturn = /\b(return|return my order|return request|send back|refund and return|replace or return|exchange|cancellation after delivery)\b/.test(qLower);
  const hasShipping = /\b(track|tracking|shipment|package|where is my order|not shipped|shipping|ship|shipped|stuck in transit|delivery|deliver|courier|parcel|transit|not arrived)\b/.test(qLower);
  const hasBilling = /\b(payment|refund(?!\s+and\s+return)|invoice|charge|charged|billing|transaction|money deducted|amount deducted|paid|unpaid|card|fee|price)\b/.test(qLower);

  // Return requests must win over generic product/technical words.
  // Shipping/tracking issues must never be converted into Billing just because
  // the category dropdown was stale or the selected order has unpaid status.
  if (hasReturn) {
    category = QueryCategory.PRODUCT;
  } else if (hasShipping && !hasBilling) {
    category = QueryCategory.SHIPPING;
  } else if (categoryOverride && categoryOverride !== "Auto-Detect") {
    category = categoryOverride as QueryCategory;
  } else if (apiCategory) {
    category = apiCategory as QueryCategory;
  } else {
    // Basic heuristic fallback
    if (hasReturn) {
      category = QueryCategory.PRODUCT;
    } else if (hasBilling) {
      category = QueryCategory.BILLING;
    } else if (hasShipping) {
      category = QueryCategory.SHIPPING;
    } else if (qLower.includes("warranty") || qLower.includes("dimension") || qLower.includes("battery") || qLower.includes("spec") || qLower.includes("features") || qLower.includes("manual") || qLower.includes("damaged") || qLower.includes("wrong item") || qLower.includes("return") || qLower.includes("replacement")) {
      category = QueryCategory.PRODUCT;
    } else if (qLower.includes("password") || qLower.includes("login") || qLower.includes("profile") || qLower.includes("account") || qLower.includes("email") || qLower.includes("username")) {
      category = QueryCategory.ACCOUNT;
    } else if (qLower.includes("error") || qLower.includes("crash") || qLower.includes("bug") || qLower.includes("not working") || qLower.includes("server") || qLower.includes("slow")) {
      category = QueryCategory.TECHNICAL;
    } else {
      category = QueryCategory.TECHNICAL;
    }
  }

  // 2. Detect Sentiment
  let sentiment = Sentiment.NEUTRAL;
  if (apiSentiment) {
    sentiment = apiSentiment as Sentiment;
  } else {
    if (qLower.includes("error") || qLower.includes("crash") || qLower.includes("bad") || qLower.includes("double charge") || qLower.includes("frustrated") || qLower.includes("delay") || qLower.includes("immediately") || qLower.includes("outage") || qLower.includes("stolen")) {
      sentiment = Sentiment.NEGATIVE;
    } else if (qLower.includes("thanks") || qLower.includes("thank you") || qLower.includes("great") || qLower.includes("perfect") || qLower.includes("excellent")) {
      sentiment = Sentiment.POSITIVE;
    }
  }

  // 3. Define Urgency and Risk
  let urgency: "Low" | "Medium" | "High" = "Medium";
  let riskLevel = "Low";
  
  if (category === QueryCategory.BILLING) {
    urgency = "High";
    riskLevel = "Medium (Financial Refund Exposure)";
  } else if (category === QueryCategory.TECHNICAL && (qLower.includes("crash") || qLower.includes("timeout") || qLower.includes("outage"))) {
    urgency = "High";
    riskLevel = "High (Potential Service Outage)";
  } else if (category === QueryCategory.SHIPPING) {
    urgency = "Medium";
    riskLevel = "Medium (Delivery / Tracking Support)";
  } else if (hasReturn && category === QueryCategory.PRODUCT) {
    urgency = sentiment === Sentiment.NEGATIVE ? "Medium" : "Low";
    riskLevel = "Medium (Return / Customer Satisfaction Issue)";
  } else {
    urgency = "Low";
    riskLevel = "Low";
  }

  // 4. Core Decision & Resolution Logic using selectedOrder
  let finalResolution = "";
  let autoResolved = true;
  let statusText = "Resolved Automatically";
  let resolutionTime = "Instant (Automated Gateway)";
  let decisionLogs: any[] = [];
  let memoryLogs: string[] = [];

  memoryLogs.push(`[DATABASE] Customer ID query: ${customerId || "N/A"}`);
  if (selectedOrder) {
    memoryLogs.push(`[DATABASE] Located order reference: ${selectedOrder.orderId}`);
    memoryLogs.push(`[DATABASE] Order deliveryStatus: ${selectedOrder.deliveryStatus}, paymentStatus: ${selectedOrder.paymentStatus}, refundStatus: ${selectedOrder.refundStatus}`);
  } else {
    memoryLogs.push(`[DATABASE] No order linked. Performing standalone query text prediction.`);
  }

  // Check Category Specific Rules
  if (hasReturn && category === QueryCategory.PRODUCT) {
    resolutionTime = selectedOrder ? "Return workflow initiated" : "Needs order selection";
    statusText = "Return Support";
    autoResolved = true;

    if (!selectedOrder) {
      autoResolved = false;
      statusText = "Needs More Information";
      resolutionTime = "N/A";
      finalResolution = "Your return request has been received. Please select the related order so we can verify return eligibility and initiate the return workflow.";
      decisionLogs.push({
        thought: "Return request received, but no order was selected.",
        action: "PROMPT_FOR_ORDER_SELECTION",
        observation: "Customer needs to choose the order to return."
      });
    } else if (selectedOrder.deliveryStatus === "Delivered") {
      finalResolution = `Your return request for order ${selectedOrder.orderId} (${selectedOrder.productName}) has been initiated. Please verify the product condition, keep the original packaging if available, and confirm return eligibility. Our support team will share the return pickup or drop-off instructions.`;
      decisionLogs.push({
        thought: "Return request matched to a delivered order. Start return workflow instead of technical troubleshooting.",
        action: "INITIATE_RETURN_WORKFLOW",
        observation: `Return workflow initiated for ${selectedOrder.orderId}.`
      });
    } else if (["In Transit", "Not Shipped", "Pending Dispatch", "Shipped"].includes(selectedOrder.deliveryStatus)) {
      autoResolved = false;
      statusText = "Return Pending Delivery Completion";
      resolutionTime = "After delivery / agent review";
      finalResolution = `Your return request for order ${selectedOrder.orderId} (${selectedOrder.productName}) has been noted. The order is currently ${selectedOrder.deliveryStatus}, so return pickup can be initiated after delivery or after cancellation eligibility is verified.`;
      decisionLogs.push({
        thought: "Return requested before delivery is completed. Cancellation/return eligibility should be checked.",
        action: "CHECK_CANCELLATION_OR_RETURN_ELIGIBILITY",
        observation: "Order is not yet delivered."
      });
    } else {
      finalResolution = `Your return request for order ${selectedOrder.orderId} has been initiated. We will verify eligibility and share the next return steps.`;
      decisionLogs.push({
        thought: "Return request with non-standard order status. General return workflow selected.",
        action: "VERIFY_RETURN_ELIGIBILITY",
        observation: "Return eligibility review prepared."
      });
    }
  } else if (category === QueryCategory.SHIPPING) {
    resolutionTime = "Same day tracking verification";
    statusText = "Tracking / Delivery Support";
    autoResolved = true;

    if (!selectedOrder) {
      autoResolved = false;
      statusText = "Needs More Information";
      resolutionTime = "N/A";
      finalResolution = "Your order tracking status is being verified. Please provide or select the related order so we can check courier updates and notify you with the latest delivery status.";
      decisionLogs.push({
        thought: "Tracking/delivery query received, but no order record was selected or linked.",
        action: "PROMPT_FOR_ORDER_SELECTION",
        observation: "Requested customer to select a valid purchase order."
      });
    } else if (["Not Shipped", "Pending Dispatch"].includes(selectedOrder.deliveryStatus)) {
      autoResolved = false;
      statusText = "Warehouse Follow-up Required";
      resolutionTime = "24-48 Hours";
      finalResolution = `Your order tracking status is being verified. Order ${selectedOrder.orderId} for "${selectedOrder.productName}" has not shipped yet, so we are raising a fulfillment follow-up and will notify you with the latest dispatch update.`;
      decisionLogs.push({
        thought: "Tracking issue with order status not shipped. Warehouse follow-up needed.",
        action: "RAISE_FULFILLMENT_FOLLOWUP",
        observation: "Fulfillment team notified for dispatch verification."
      });
    } else if (["In Transit", "Out for Delivery", "Shipped", "Delayed"].includes(selectedOrder.deliveryStatus)) {
      finalResolution = `Your order tracking status is being verified. Order ${selectedOrder.orderId} for "${selectedOrder.productName}" is currently ${selectedOrder.deliveryStatus} to ${selectedOrder.shippingCity}. We will check courier updates and notify you with the latest delivery status. Expected delivery: ${selectedOrder.expectedDeliveryDate}.`;
      decisionLogs.push({
        thought: "Tracking issue matched to a non-delivered order. Courier status should be verified.",
        action: "VERIFY_COURIER_TRACKING",
        observation: "Tracking verification prepared for order: " + selectedOrder.orderId
      });
    } else if (selectedOrder.deliveryStatus === "Delivered") {
      statusText = "Delivery Confirmation Required";
      autoResolved = false;
      resolutionTime = "12-24 Hours";
      finalResolution = `Your order tracking status is being verified. Our records show order ${selectedOrder.orderId} for "${selectedOrder.productName}" is marked Delivered, but since you are unable to track/confirm it, we will verify the courier delivery proof and notify you with the latest status.`;
      decisionLogs.push({
        thought: "Tracking issue linked to a delivered order. Do not route to billing. Verify courier proof instead.",
        action: "VERIFY_DELIVERY_PROOF",
        observation: "Delivery confirmation review required."
      });
    } else {
      finalResolution = `Your order tracking status is being verified. We will check courier updates for order ${selectedOrder.orderId} and notify you with the latest delivery status.`;
      decisionLogs.push({
        thought: "Shipping issue with unusual order status. General tracking verification selected.",
        action: "VERIFY_TRACKING_STATUS",
        observation: "Tracking support workflow selected."
      });
    }
  } else if (category === QueryCategory.BILLING && selectedOrder) {
    if (selectedOrder.refundStatus === "Processing") {
      finalResolution = `Your refund of $${selectedOrder.orderAmount} for order ${selectedOrder.orderId} is already being processed. It is currently in the clearing gateway and should be back on your original payment method shortly.`;
      
      decisionLogs.push({
        thought: "Query is billing-related. Refund status is 'Processing'. Reassure customer and monitor clearing progress.",
        action: "FETCH_GATEWAY_REFUND_TX",
        observation: "Gateway shows status: processing_auth_reversal."
      });
    } else if (selectedOrder.paymentStatus === "Paid") {
      if (qLower.includes("double charge") || qLower.includes("charged twice") || qLower.includes("duplicate")) {
        autoResolved = false;
        statusText = "Escalated to Human Agent";
        resolutionTime = "2-4 Hours (Assigned Queue)";
        finalResolution = `Your payment of $${selectedOrder.orderAmount} for order ${selectedOrder.orderId} is currently listed as paid. Since you reported a double charge, we have escalated this transaction to our Billing Team to check for double auth holds.`;
        
        decisionLogs.push({
          thought: "Double charge claim found on paid transaction. Escalating to human finance audits.",
          action: "QUEUE_FINANCE_RECONCILIATION",
          observation: "Reconciliation ticket created in Finance Queue."
        });
      } else {
        finalResolution = `Your payment of $${selectedOrder.orderAmount} for order ${selectedOrder.orderId} was successfully processed and is currently marked as Paid. No active refund has been initiated.`;
        
        decisionLogs.push({
          thought: "Billing query on paid transaction. No refund requested in records.",
          action: "DISPLAY_BILLING_SUMMARY",
          observation: "Invoice verified. Paid on: " + selectedOrder.orderDate
        });
      }
    } else if (selectedOrder.paymentStatus === "Refunded" || selectedOrder.refundStatus === "Completed") {
      finalResolution = `A full refund of $${selectedOrder.orderAmount} for order ${selectedOrder.orderId} has been successfully completed and credited back to your account.`;
      
      decisionLogs.push({
        thought: "Billing query on a refunded transaction.",
        action: "DISPLAY_REFUND_LEDGER",
        observation: "Refund transaction confirmed completed."
      });
    } else {
      finalResolution = `The payment status for order ${selectedOrder.orderId} is Unpaid. Please update your payment credentials to complete the purchase.`;
      decisionLogs.push({
        thought: "Billing query on unpaid transaction.",
        action: "PROMPT_PAYMENT_RETRY",
        observation: "No successful captured hold found."
      });
    }
  } else if (qLower.includes("refund") && selectedOrder && selectedOrder.refundStatus === "Processing") {
    finalResolution = `Your refund for order ${selectedOrder.orderId} is already being processed. It is currently in the clearing gateway and should credit back within 2-4 business days.`;
    
    decisionLogs.push({
      thought: "Refund-related query. Database lookup confirmed refundStatus is Processing.",
      action: "FETCH_GATEWAY_REFUND_TX",
      observation: "Gateway status: processing_auth_reversal."
    });
  } else if (category === QueryCategory.TECHNICAL) {
    const productItem = selectedOrder ? selectedOrder.productName : "your device";
    finalResolution = `Regarding your technical query about "${productItem}", please try the following troubleshooting sequence:
1. Power cycle and reset the hardware device.
2. Verify that network access is stable and router connections are solid.
3. Check for any pending system firmware or app updates in settings.
If the interface still displays errors, please reply with the error code so our technician squad can inspect.`;

    decisionLogs.push({
      thought: "Technical issue flagged. Generating custom hardware sequence for: " + productItem,
      action: "GENERATE_TROUBLESHOOTING",
      observation: "Diagnostics steps generated successfully."
    });
  } else {
    // Default / Standalone prediction resolution
    if ((category as any) === QueryCategory.BILLING) {
      finalResolution = `We received your billing inquiry. We will check our accounting ledgers for any duplicate authorizations or refund holds and update you shortly.`;
    } else if ((category as any) === QueryCategory.SHIPPING) {
      finalResolution = `Your order tracking status is being verified. We will check courier updates and notify you with the latest delivery status.`;
    } else if ((category as any) === QueryCategory.PRODUCT) {
      finalResolution = `We received your product inquiry. Our sales and specifications desk will verify the dimensions and features and respond via email.`;
    } else if ((category as any) === QueryCategory.ACCOUNT) {
      finalResolution = `We received your account inquiry. Please verify your email or click "Reset Password" to trigger a secure reset link.`;
    } else {
      finalResolution = `We received your technical query. Our technical engineering squad has logged your request and will assist you shortly.`;
    }

    decisionLogs.push({
      thought: "No specific order linked. Formulated general response based on predicted category: " + category,
      action: "COMPILE_GENERIC_RESOLUTION",
      observation: "Standard SLA response generated."
    });
  }

  // 5. Generate suggested response email
  const personalizedEmail = `Dear ${customerName || "Customer"},\n\nThank you for reaching out to customer support regarding your query.\n\n${finalResolution}\n\nWe appreciate your patience. If you have any further questions, please let us know.\n\nSincerely,\nAI Resolution Engine\nRef: ${selectedOrder ? selectedOrder.orderId : "N/A"}`;

  // 6. Generate troubleshooting guide
  const productLabel = selectedOrder ? selectedOrder.productName : "Universal Hardware";
  const troubleshootingGuide = `### Troubleshooting Guide for ${productLabel}
1. **Power Cycle**: Turn off the device, unplug it for 30 seconds, and restart.
2. **Network Reset**: Restart your internet router and check signal indicators.
3. **Firmware Update**: Sync with our server to verify that firmware is updated.
4. **Diagnostic Logs**: If error persists, export device logs and email to technical support.`;

  return {
    category,
    sentiment,
    urgency,
    riskLevel,
    autoResolved,
    resolutionTime,
    statusText,
    finalResolution,
    personalizedEmail,
    faqList: [
      { question: `What is the delivery status of my ${productLabel}?`, answer: selectedOrder ? `The current status is ${selectedOrder.deliveryStatus}.` : "Please link a valid order ID to inspect live transit coordinates." },
      { question: "How long do refunds take to process?", answer: "Once initiated, standard banking gateway clearance takes 3 to 5 business days." }
    ],
    troubleshootingGuide,
    decisionLogs,
    memoryLogs
  };
}
