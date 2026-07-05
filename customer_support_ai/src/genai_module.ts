/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from "@google/genai";
import { GenAIResult, QueryCategory, Sentiment } from "./types.js";

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

// -------------------------------------------------------------
// LOCAL GENERATION RULES (Robust fallback engine)
// -------------------------------------------------------------
function generateLocalGenAI(
  queryText: string,
  category: QueryCategory,
  customerName: string,
  sentiment: Sentiment,
  orderId?: string
): GenAIResult {
  const name = customerName || "Customer";
  const ord = orderId || "ORD-000000";

  let personalizedEmail = "";
  let faqList: { question: string; answer: string }[] = [];
  let troubleshootingGuide = "";
  let simulatedConversation: { speaker: string; text: string }[] = [];

  const sentimentIntro = sentiment === Sentiment.NEGATIVE 
    ? "I want to sincerely apologize for the frustration this has caused. We completely understand your concern and are committed to resolving this immediately."
    : "Thank you for reaching out to us. We are pleased to assist you with your inquiry.";

  switch (category) {
    case QueryCategory.BILLING:
      personalizedEmail = `Dear ${name},

${sentimentIntro}

Regarding your order reference ${ord}, we have investigated the billing history. We identified a duplicate capture transaction and have flagged this transaction for immediate reversal. 

We have initiated a refund of the duplicate amount. This should reflect back in your original payment method in 3-5 business days. No further action is required from your end.

Sincerely,
Automated Finance Support Team`;

      faqList = [
        {
          question: `Why was my order ${ord} billed twice?`,
          answer: "This occasionally occurs if the transaction checkout is clicked multiple times before receiving a response, or if the merchant bank gateway undergoes a slight sync delay."
        },
        {
          question: "How long do refunds take to credit back?",
          answer: "Our system processes refunds instantly, but your card issuer may take 3 to 10 banking days to clear and post the credit to your statement."
        },
        {
          question: "Can I update my billing statement details after checkout?",
          answer: "No, once an invoice is finalized and locked, we cannot modify the billing address on that specific order. However, we can update your account settings for all future transactions."
        }
      ];

      troubleshootingGuide = `### BILLING RESOLUTION PROTOCOL
1. **Audit Logs Lookup**: Check Stripe/Merchant ledger with Order ID ${ord}.
2. **Identify Gateway Errors**: Ensure transaction shows double auth status.
3. **Execute Safe Reverse**: Trigger standard partial refund on the duplicate ID.
4. **Verify Settlement**: Check if gateway reports refund pending confirmation.
5. **Lock Account Records**: Store updated refund token in user ledger database.`;

      simulatedConversation = [
        { speaker: "Customer", text: `Hi, I see two identical charges on my card for order ${ord}.` },
        { speaker: "Support Agent", text: `Hello ${name}. I am very sorry to hear about the duplicate charge. Let me pull up your transaction records for order ${ord}.` },
        { speaker: "Customer", text: "Yes, please, it has locked up a lot of credit on my card." },
        { speaker: "Support Agent", text: "I completely understand your frustration. I see the duplicate capture here. I am reversing the second charge right now and issuing a full refund. You should receive a receipt email in a few minutes." },
        { speaker: "Customer", text: "Thank you so much! That was extremely quick." }
      ];
      break;

    case QueryCategory.TECHNICAL:
      personalizedEmail = `Dear ${name},

${sentimentIntro}

Our engineering team has analyzed the diagnostic details and error reports regarding your device setup. We noticed a system mismatch that is causing the application client to throw errors.

To address this technical issue, we recommend updating your software version or clearing your client-side cached databases as outlined in the troubleshooting guide below.

Best regards,
Automated Technical Ops`;

      faqList = [
        {
          question: "What causes error logs on my client device?",
          answer: "Most connection or startup errors occur due to stale local cache files, invalid authorization tokens, or momentary server latency."
        },
        {
          question: "How do I clear the cached database on my system?",
          answer: "Navigate to Settings > Application Management > Clear Storage and Cache, then restart your device."
        },
        {
          question: "Is there an outage affecting our services?",
          answer: "All core database nodes are currently running normally. You can view real-time operations on our server status dashboards."
        }
      ];

      troubleshootingGuide = `### TECHNICAL DIAGNOSTIC GUIDE
1. **Identify OS Platform**: Determine if system is on mobile or desktop client.
2. **Clear Application Cache**: Go to system settings and clear temporary cache data.
3. **Flush Local DNS**: Open terminal and execute 'ipconfig /flushdns' (Windows) or terminal command.
4. **Verify License Active**: Re-enter auth credentials in settings.
5. **Cold Reboot**: Power off the hardware device for 30 seconds and reboot.`;

      simulatedConversation = [
        { speaker: "Customer", text: "My application keeps throwing error codes and freezing on my device." },
        { speaker: "Support Agent", text: `I apologize for the technical issue, ${name}. Let's run a diagnostic. What operating version is your device running?` },
        { speaker: "Customer", text: "I am running the latest firmware update on my device." },
        { speaker: "Support Agent", text: "Got it. This is a known cache mismatch. Please navigate to Application settings, click 'Clear Application Cache', and perform a cold reboot. This should clear the database error." },
        { speaker: "Customer", text: "Ah, that worked perfectly! The application is open and running smoothly now." }
      ];
      break;

    case QueryCategory.SHIPPING:
      personalizedEmail = `Dear ${name},

${sentimentIntro}

Regarding your order shipping tracker ${ord}, we checked our logistics database. Your package is currently processed at the local hub and is in transit.

We have compiled shipping updates and tracking links below so you can monitor the progress of your shipment directly.

Sincerely,
Automated Logistics Team`;

      faqList = [
        {
          question: `Where can I find the tracking number for ${ord}?`,
          answer: "Tracking numbers are emailed immediately upon shipment dispatch, and can also be retrieved in the order history section of your user profile."
        },
        {
          question: "What happens if my package is delayed?",
          answer: "If a package is held in transit beyond the estimated delivery date, we will initiate an investigation with the carrier and keep you informed."
        },
        {
          question: "Can I update my delivery address post-dispatch?",
          answer: "Once a package leaves our warehouse, we are unable to update the routing. However, you can contact the carrier directly with your tracking number to request a hub pickup."
        }
      ];

      troubleshootingGuide = `### LOGISTICS TRACKING PROTOCOL
1. **Query Carrier Registry**: Check tracking code in carrier shipment log database.
2. **Confirm Hub Status**: Verify if package is scanned at sorting hubs.
3. **Review Address Routing**: Validate that city information matches user invoice address.
4. **Check Regional Latency**: Look up any carrier-reported storms, sorting backlogs, or sorting delays.
5. **Issue Status Alert**: Send automatic tracking update email directly to the customer.`;

      simulatedConversation = [
        { speaker: "Customer", text: `Hi, where is my package for order ${ord}? It has been stuck at the sorting hub.` },
        { speaker: "Support Agent", text: `Hello ${name}. Let's check on your shipment tracker for order ${ord}.` },
        { speaker: "Customer", text: "It was supposed to arrive yesterday and I really need it." },
        { speaker: "Support Agent", text: "I understand. I see the carrier scanning logs. It was delayed due to a regional weather backup at the hub. It is now out for delivery today and should be at your doorstep by 5 PM." },
        { speaker: "Customer", text: "Great, thanks for looking into that for me." }
      ];
      break;

    default:
      personalizedEmail = `Dear ${name},

${sentimentIntro}

We have received your account inquiry and would love to assist you. Our support database has flagged your request, and we have drafted immediate resolution details.

Please refer to the detailed guide below.

Sincerely,
Customer Relations Team`;

      faqList = [
        {
          question: "How do I secure my login credentials?",
          answer: "We strongly recommend enabling Two-Factor Authentication (2FA) via an authenticator app in your profile security dashboard."
        },
        {
          question: "How do I cancel my subscription billing?",
          answer: "Go to Profile > Account Settings > Subscription and click 'Cancel Subscription'."
        },
        {
          question: "Are there warranties on physical items?",
          answer: "All our standard items carry a 1-year hardware replacement warranty from the original date of purchase."
        }
      ];

      troubleshootingGuide = `### ACCOUNT SECURITY STANDARD
1. **Verify Customer Profile**: Validate identity using verification links.
2. **Review Access Logs**: Inspect login histories, IP addresses, and browsers.
3. **Enable 2FA Options**: Guide user to scan the secure authenticator barcode.
4. **Update Register Details**: Save changes safely in the encrypted user accounts database.`;

      simulatedConversation = [
        { speaker: "Customer", text: "I am trying to update my profile details and set up some security protocols." },
        { speaker: "Support Agent", text: `I would be happy to guide you through that, ${name}. Let's get your profile secured.` },
        { speaker: "Customer", text: "Do you support authenticator apps for 2FA?" },
        { speaker: "Support Agent", text: "Absolutely. Log into your dashboard, go to Security, and click 'Configure Authenticator App'. A barcode will appear for your mobile app to scan." },
        { speaker: "Customer", text: "Done! Security settings are successfully enabled now. Thanks." }
      ];
      break;
  }

  return { personalizedEmail, faqList, troubleshootingGuide, simulatedConversation };
}

// -------------------------------------------------------------
// GENAI EXECUTION ENGINE
// -------------------------------------------------------------
export async function runGenAITasks(
  queryText: string,
  category: QueryCategory,
  customerName: string,
  sentiment: Sentiment,
  orderId?: string
): Promise<GenAIResult> {
  if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== "MY_GEMINI_API_KEY") {
    try {
      const ai = getGenAIClient();
      const systemPrompt = `You are an expert Generative AI system specialized in customer service automation.
Analyze the customer query and details, and output a strict JSON payload adhering to this format:
{
  "personalizedEmail": "Dear [Name], \\n\\n[A beautifully formatted, highly personalized email addressing their query context, order ID if present, sentiment, and outlining the clear resolution paths...]\\n\\nSincerely,\\nSupport Team",
  "faqList": [
    { "question": "Question 1 relevant to query", "answer": "Answer 1" },
    { "question": "Question 2 relevant to query", "answer": "Answer 2" },
    { "question": "Question 3 relevant to query", "answer": "Answer 3" }
  ],
  "troubleshootingGuide": "A detailed step-by-step troubleshooting manual for this issue with headers, numbers, and technical instructions.",
  "simulatedConversation": [
    { "speaker": "Customer", "text": "Customer opening statement" },
    { "speaker": "Support Agent", "text": "Empathetic agent response" },
    { "speaker": "Customer", "text": "Customer follow up" },
    { "speaker": "Support Agent", "text": "Agent final resolution" }
  ]
}

No extra words, no markdown blocks. Output raw JSON only.`;

      const userPrompt = `Customer Name: "${customerName}"
Query: "${queryText}"
Category: ${category}
Sentiment: ${sentiment}
Order ID: "${orderId || 'N/A'}"`;

      const response = await callWithRetry(() => ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: [
          { text: systemPrompt },
          { text: userPrompt }
        ],
        config: {
          responseMimeType: "application/json",
          temperature: 0.3
        }
      }));

      const text = response.text || "{}";
      const data = JSON.parse(text);

      return {
        personalizedEmail: data.personalizedEmail || "",
        faqList: data.faqList || [],
        troubleshootingGuide: data.troubleshootingGuide || "",
        simulatedConversation: data.simulatedConversation || []
      };
    } catch (err: any) {
      console.log(`Generative AI Gemini tasks failed gracefully, falling back to local engine. Error: ${err?.message || err}`);
      return generateLocalGenAI(queryText, category, customerName, sentiment, orderId);
    }
  } else {
    // Graceful offline fallback
    return generateLocalGenAI(queryText, category, customerName, sentiment, orderId);
  }
}
