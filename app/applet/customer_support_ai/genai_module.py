# -*- coding: utf-8 -*-
"""
Generative AI module for Customer Support AI.
Drafts personalized emails, structures FAQ templates, and emulates interactive dialogs.
"""

import os
import json

# --- LOCAL RULE-BASED GENERATOR ---
def generate_local_genai(query_text: str, category: str, name: str, sentiment: str, order_id: str) -> dict:
    order_ref = order_id or "ORD-XXXXXX"
    cust_name = name or "Customer"

    # 1. Draft Email
    email_subject = f"Support Follow-up: Re: {category} Inquiry ({order_ref})"
    email_body = (
        f"Subject: {email_subject}\n\n"
        f"Dear {cust_name},\n\n"
        f"Thank you for contacting our {category} team. We have received your query regarding:\n"
        f"\"{query_text}\"\n\n"
        f"We understand this matter is important and we are actively tracking order {order_ref}. "
        f"Based on our initial assessment (Sentiment: {sentiment}), we have prioritized your ticket. "
        "An automated diagnostic pipeline has been allocated, and we expect to resolve this issue within 24 hours.\n\n"
        "Should you have any further questions in the meantime, please feel free to reply to this thread.\n\n"
        "Sincerely,\n"
        "Automated Support AI Agent"
    )

    # 2. FAQs
    faqs = [
        {
            "question": f"How do I track order {order_ref}?",
            "answer": "You can search for your order ID directly on our dashboard or click the live tracking link sent via email."
        },
        {
            "question": f"What is the average resolution time for {category} inquiries?",
            "answer": f"Standard {category} issues are resolved automatically within 12 hours, while escalated cases are processed by senior agents in under 24 hours."
        }
    ]

    # 3. Troubleshooting Guide
    guide_steps = [
        "Ensure all network connection cables are tightly secured.",
        "Clear your application session cache and restart your browser client.",
        "Verify your payment status or account credentials under settings."
    ]
    guide = (
        f"# Automated Troubleshooting Guide: {category}\n\n"
        "To resolve your issue immediately, please complete these steps:\n\n" +
        "\n".join([f"{idx+1}. **{s}**" for idx, s in enumerate(guide_steps)]) +
        "\n\nIf the issue persists, our support lines remain open 24/7."
    )

    # 4. Interactive Dialog Simulation
    dialog = [
        {"speaker": "Customer", "text": f"Hi, I need help with my {category} issue."},
        {"speaker": "Agent", "text": f"Hello {cust_name}! I can certainly help. I see this relates to order {order_ref}. Can you describe what occurred?"},
        {"speaker": "Customer", "text": query_text},
        {"speaker": "Agent", "text": "I've logged those details. Let me run a diagnostic script and draft a ticket. We will keep you updated!"}
    ]

    return {
        "emailDraft": email_body,
        "faqs": faqs,
        "troubleshootingGuide": guide,
        "dialogSimulator": dialog
    }

# --- DUAL GENAI PIPELINE ---
async def run_genai_tasks(query_text: str, category: str = "Technical", name: str = "Customer", sentiment: str = "Neutral", order_id: str = "") -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key or api_key == "MY_GEMINI_API_KEY" or api_key == "":
        return generate_local_genai(query_text, category, name, sentiment, order_id)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        prompt = (
            "You are an advanced Generative AI customer support copywriter.\n"
            f"Analyze this ticket details:\n"
            f"- Customer Query: \"{query_text}\"\n"
            f"- Category: {category}\n"
            f"- Customer Name: {name}\n"
            f"- Query Sentiment: {sentiment}\n"
            f"- Associated Order ID: {order_id}\n\n"
            "Generate a structured JSON containing four items:\n"
            "1. \"emailDraft\": A complete, beautifully structured personalized support email follow-up including subject and body.\n"
            "2. \"faqs\": An array of 2 frequently asked questions and answer objects tailored to this issue.\n"
            "3. \"troubleshootingGuide\": A markdown formatted troubleshooting guide with 3 actionable bold steps.\n"
            "4. \"dialogSimulator\": A 4-turn realistic chat transcript array between Customer and Agent.\n\n"
            "Return ONLY a clean JSON object fitting this schema:\n"
            "{\n"
            "  \"emailDraft\": \"Subject: ...\\n\\nDear ...\",\n"
            "  \"faqs\": [{\"question\": \"...\", \"answer\": \"...\"}, ...],\n"
            "  \"troubleshootingGuide\": \"# Title\\n\\n...\",\n"
            "  \"dialogSimulator\": [{\"speaker\": \"Customer\", \"text\": \"...\"}, {\"speaker\": \"Agent\", \"text\": \"...\"}, ...]\n"
            "}\n"
            "Output ONLY raw JSON, do not include markdown backticks."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )

        res_text = response.text.strip()
        if res_text.startswith("```json"):
            res_text = res_text[7:]
        if res_text.endswith("```"):
            res_text = res_text[:-3]
        res_text = res_text.strip()

        return json.loads(res_text)

    except Exception as e:
        print(f"Gemini GenAI operations failed: {e}. Falling back to Rule Engine.")
        return generate_local_genai(query_text, category, name, sentiment, order_id)

if __name__ == "__main__":
    import asyncio
    q = "Secure Router Pro firmware failed on installation."
    res = asyncio.run(run_genai_tasks(q, "Technical", "Taylor Swift", "Negative", "ORD-991201"))
    print(res)
