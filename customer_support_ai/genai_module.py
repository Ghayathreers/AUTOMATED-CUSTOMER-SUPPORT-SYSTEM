# -*- coding: utf-8 -*-
"""
Generative AI Module for Automated Customer Support AI.
Generates personalized response emails, related FAQs, structured troubleshooting manuals,
and multi-turn customer-agent conversational simulations using Gemini or rule-based templates.
"""

import os
import json
import random

# Reuse local rule generation structure
def generate_local_gen_ai(
    query_text: str,
    category: str,
    customer_name: str,
    sentiment: str,
    order_id: str = ""
) -> dict:
    name = customer_name if customer_name else "Customer"
    ord_id = order_id if order_id else "ORD-000000"
    sentiment_str = sentiment if sentiment else "Neutral"
    cat_key = category.title() if category else "Technical"

    sentiment_intro = (
        "I want to sincerely apologize for the frustration this has caused. We completely understand your concern and are committed to resolving this immediately."
        if sentiment_str.upper() == "NEGATIVE" else
        "Thank you for reaching out to us. We are pleased to assist you with your inquiry."
    )

    personalized_email = ""
    faq_list = []
    troubleshooting_guide = ""
    simulated_conversation = []

    if cat_key == "Billing":
        personalized_email = f"""Dear {name},

{sentiment_intro}

Regarding your order reference {ord_id}, we have investigated the billing history. We identified a duplicate capture transaction and have flagged this transaction for immediate reversal. 

We have initiated a refund of the duplicate amount. This should reflect back in your original payment method in 3-5 business days. No further action is required from your end.

Sincerely,
Automated Finance Support Team"""

        faq_list = [
            {
                "question": f"Why was my order {ord_id} billed twice?",
                "answer": "This occasionally occurs if the transaction checkout is clicked multiple times before receiving a response, or if the merchant bank gateway undergoes a slight sync delay."
            },
            {
                "question": "How long do refunds take to credit back?",
                "answer": "Our system processes refunds instantly, but your card issuer may take 3 to 10 banking days to clear and post the credit to your statement."
            },
            {
                "question": "Can I update my billing statement details after checkout?",
                "answer": "No, once an invoice is finalized and locked, we cannot modify the billing address on that specific order. However, we can update your account settings for all future transactions."
            }
        ]

        troubleshooting_guide = f"""### BILLING RESOLUTION PROTOCOL
1. **Audit Logs Lookup**: Check Stripe/Merchant ledger with Order ID {ord_id}.
2. **Identify Gateway Errors**: Ensure transaction shows double auth status.
3. **Execute Safe Reverse**: Trigger standard partial refund on the duplicate ID.
4. **Verify Settlement**: Check if gateway reports refund pending confirmation.
5. **Lock Account Records**: Store updated refund token in user ledger database."""

        simulated_conversation = [
            {"speaker": "Customer", "text": f"Hi, I see two identical charges on my card for order {ord_id}."},
            {"speaker": "Support Agent", "text": f"Hello {name}. I am very sorry to hear about the duplicate charge. Let me pull up your transaction records for order {ord_id}."},
            {"speaker": "Customer", "text": "Yes, please, it has locked up a lot of credit on my card."},
            {"speaker": "Support Agent", "text": "I completely understand your frustration. I see the duplicate capture here. I am reversing the second charge right now and issuing a full refund. You should receive a receipt email in a few minutes."},
            {"speaker": "Customer", "text": "Thank you so much! That was extremely quick."}
        ]

    elif cat_key == "Technical":
        personalized_email = f"""Dear {name},

{sentiment_intro}

Our engineering team has analyzed the diagnostic details and error reports regarding your device setup. We noticed a system mismatch that is causing the application client to throw errors.

To address this technical issue, we recommend updating your software version or clearing your client-side cached databases as outlined in the troubleshooting guide below.

Best regards,
Automated Technical Ops"""

        faq_list = [
            {
                "question": "What causes error logs on my client device?",
                "answer": "Most connection or startup errors occur due to stale local cache files, invalid authorization tokens, or momentary server latency."
            },
            {
                "question": "How do I clear the cached database on my system?",
                "answer": "Navigate to Settings > Application Management > Clear Storage and Cache, then restart your device."
            },
            {
                "question": "Is there an outage affecting our services?",
                "answer": "All core database nodes are currently running normally. You can view real-time operations on our server status dashboards."
            }
        ]

        troubleshooting_guide = """### TECHNICAL DIAGNOSTIC GUIDE
1. **Identify OS Platform**: Determine if system is on mobile or desktop client.
2. **Clear Application Cache**: Go to system settings and clear temporary cache data.
3. **Flush Local DNS**: Open terminal and execute 'ipconfig /flushdns' (Windows) or terminal command.
4. **Verify License Active**: Re-enter auth credentials in settings.
5. **Cold Reboot**: Power off the hardware device for 30 seconds and reboot."""

        simulated_conversation = [
            {"speaker": "Customer", "text": "My application keeps throwing error codes and freezing on my device."},
            {"speaker": "Support Agent", "text": f"I apologize for the technical issue, {name}. Let's run a diagnostic. What operating version is your device running?"},
            {"speaker": "Customer", "text": "I am running the latest firmware update on my device."},
            {"speaker": "Support Agent", "text": "Got it. This is a known cache mismatch. Please navigate to Application settings, click 'Clear Application Cache', and perform a cold reboot. This should clear the database error."},
            {"speaker": "Customer", "text": "Ah, that worked perfectly! The application is open and running smoothly now."}
        ]

    elif cat_key == "Shipping":
        personalized_email = f"""Dear {name},

{sentiment_intro}

Regarding your order shipping tracker {ord_id}, we checked our logistics database. Your package is currently processed at the local hub and is in transit.

We have compiled shipping updates and tracking links below so you can monitor the progress of your shipment directly.

Sincerely,
Automated Logistics Team"""

        faq_list = [
            {
                "question": f"Where can I find the tracking number for {ord_id}?",
                "answer": "Tracking numbers are emailed immediately upon shipment dispatch, and can also be retrieved in the order history section of your user profile."
            },
            {
                "question": "What happens if my package is delayed?",
                "answer": "If a package is held in transit beyond the estimated delivery date, we will initiate an investigation with the carrier and keep you informed."
            },
            {
                "question": "Can I update my delivery address post-dispatch?",
                "answer": "Once a package leaves our warehouse, we are unable to update the routing. However, you can contact the carrier directly with your tracking number to request a hub pickup."
            }
        ]

        troubleshooting_guide = f"""### LOGISTICS TRACKING PROTOCOL
1. **Query Carrier Registry**: Check tracking code in carrier shipment log database.
2. **Confirm Hub Status**: Verify if package is scanned at sorting hubs.
3. **Review Address Routing**: Validate that city information matches user invoice address.
4. **Check Regional Latency**: Look up any carrier-reported storms, sorting backlogs, or sorting delays.
5. **Issue Status Alert**: Send automatic tracking update email directly to the customer."""

        simulated_conversation = [
            {"speaker": "Customer", "text": f"Hi, where is my package for order {ord_id}? It has been stuck at the sorting hub."},
            {"speaker": "Support Agent", "text": f"Hello {name}. Let's check on your shipment tracker for order {ord_id}."},
            {"speaker": "Customer", "text": "It was supposed to arrive yesterday and I really need it."},
            {"speaker": "Support Agent", "text": "I understand. I see the carrier scanning logs. It was delayed due to a regional weather backup at the hub. It is now out for delivery today and should be at your doorstep by 5 PM."},
            {"speaker": "Customer", "text": "Great, thanks for looking into that for me."}
        ]

    else:
        personalized_email = f"""Dear {name},

{sentiment_intro}

We have received your account inquiry and would love to assist you. Our support database has flagged your request, and we have drafted immediate resolution details.

Please refer to the detailed guide below.

Sincerely,
Customer Relations Team"""

        faq_list = [
            {
                "question": "How do I secure my login credentials?",
                "answer": "We strongly recommend enabling Two-Factor Authentication (2FA) via an authenticator app in your profile security dashboard."
            },
            {
                "question": "How do I cancel my subscription billing?",
                "answer": "Go to Profile > Account Settings > Subscription and click 'Cancel Subscription'."
            },
            {
                "question": "Are there warranties on physical items?",
                "answer": "All our standard items carry a 1-year hardware replacement warranty from the original date of purchase."
            }
        ]

        troubleshooting_guide = """### ACCOUNT SECURITY STANDARD
1. **Verify Customer Profile**: Validate identity using verification links.
2. **Review Access Logs**: Inspect login histories, IP addresses, and browsers.
3. **Enable 2FA Options**: Guide user to scan the secure authenticator barcode.
4. **Update Register Details**: Save changes safely in the encrypted user accounts database."""

        simulated_conversation = [
            {"speaker": "Customer", "text": "I am trying to update my profile details and set up some security protocols."},
            {"speaker": "Support Agent", "text": f"I would be happy to guide you through that, {name}. Let's get your profile secured."},
            {"speaker": "Customer", "text": "Do you support authenticator apps for 2FA?"},
            {"speaker": "Support Agent", "text": "Absolutely. Log into your dashboard, go to Security, and click 'Configure Authenticator App'. A barcode will appear for your mobile app to scan."},
            {"speaker": "Customer", "text": "Done! Security settings are successfully enabled now. Thanks."}
        ]

    return {
        "personalizedEmail": personalized_email,
        "faqList": faq_list,
        "troubleshootingGuide": troubleshooting_guide,
        "simulatedConversation": simulated_conversation
    }

async def run_genai_tasks(
    query_text: str,
    category: str,
    customer_name: str,
    sentiment: str,
    order_id: str = ""
) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if api_key and api_key != "MY_GEMINI_API_KEY" and api_key != "":
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            system_prompt = (
                "You are an expert Generative AI system specialized in customer service automation.\n"
                "Analyze the customer query and details, and output a strict JSON payload adhering to this format:\n"
                "{\n"
                "  \"personalizedEmail\": \"Dear [Name],\\n\\n[A beautifully formatted email addressing their query context, order ID, sentiment, and outlining clear resolution paths...]\\n\\nSincerely,\\nSupport Team\",\n"
                "  \"faqList\": [\n"
                "    { \"question\": \"Question 1 relevant to query\", \"answer\": \"Answer 1\" },\n"
                "    { \"question\": \"Question 2 relevant to query\", \"answer\": \"Answer 2\" },\n"
                "    { \"question\": \"Question 3 relevant to query\", \"answer\": \"Answer 3\" }\n"
                "  ],\n"
                "  \"troubleshootingGuide\": \"A detailed step-by-step troubleshooting manual for this issue with headers, numbers, and technical instructions.\",\n"
                "  \"simulatedConversation\": [\n"
                "    { \"speaker\": \"Customer\", \"text\": \"Customer opening statement\" },\n"
                "    { \"speaker\": \"Support Agent\", \"text\": \"Empathetic agent response\" },\n"
                "    { \"speaker\": \"Customer\", \"text\": \"Customer follow up\" },\n"
                "    { \"speaker\": \"Support Agent\", \"text\": \"Agent final resolution\" }\n"
                "  ]\n"
                "}\n"
                "No extra words, no markdown blocks. Output raw JSON only."
            )

            user_prompt = (
                f"Customer Name: \"{customer_name}\"\n"
                f"Query: \"{query_text}\"\n"
                f"Category: {category}\n"
                f"Sentiment: {sentiment}\n"
                f"Order ID: \"{order_id or 'N/A'}\""
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )

            data = json.loads(response.text)
            return {
                "personalizedEmail": data.get("personalizedEmail", ""),
                "faqList": data.get("faqList", []),
                "troubleshootingGuide": data.get("troubleshootingGuide", ""),
                "simulatedConversation": data.get("simulatedConversation", [])
            }

        except Exception as e:
            print(f"GenAI Gemini task execution failed, falling back to rule engine. Error: {e}")
            return generate_local_gen_ai(query_text, category, customer_name, sentiment, order_id)
    else:
        return generate_local_gen_ai(query_text, category, customer_name, sentiment, order_id)

if __name__ == "__main__":
    import asyncio
    res = asyncio.run(run_genai_tasks("I forgot my password.", "Account", "Emily", "Negative"))
    print(json.dumps(res, indent=2))
