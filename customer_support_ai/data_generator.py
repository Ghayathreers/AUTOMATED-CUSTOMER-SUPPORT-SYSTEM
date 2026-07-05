# -*- coding: utf-8 -*-
"""
Data Generator for Automated Customer Support AI.
Generates 10,000 realistic customer profiles, orders, and support tickets,
saving them to customers.csv, orders.csv, and support_queries.csv.
"""

import os
import random
import csv
import json
from datetime import datetime, timedelta

# Constants & templates
CITIES_WITH_STATES = [
    {"city": "Austin", "state": "TX"},
    {"city": "Boston", "state": "MA"},
    {"city": "Chicago", "state": "IL"},
    {"city": "Denver", "state": "CO"},
    {"city": "El Paso", "state": "TX"},
    {"city": "Fort Worth", "state": "TX"},
    {"city": "Gary", "state": "IN"},
    {"city": "Houston", "state": "TX"},
    {"city": "Indianapolis", "state": "IN"},
    {"city": "Jacksonville", "state": "FL"},
    {"city": "Kansas City", "state": "MO"},
    {"city": "Los Angeles", "state": "CA"},
    {"city": "Miami", "state": "FL"},
    {"city": "Nashville", "state": "TN"},
    {"city": "Oakland", "state": "CA"},
    {"city": "Phoenix", "state": "AZ"},
    {"city": "Queens", "state": "NY"},
    {"city": "Raleigh", "state": "NC"},
    {"city": "Seattle", "state": "WA"},
    {"city": "Tampa", "state": "FL"},
    {"city": "Urbana", "state": "IL"},
    {"city": "Vancouver", "state": "WA"},
    {"city": "Wichita", "state": "KS"},
    {"city": "Xenia", "state": "OH"},
    {"city": "Yonkers", "state": "NY"},
    {"city": "Zion", "state": "IL"}
]

PRODUCTS = [
    {"name": "Pro Headphones X", "category": "Audio", "price": 299},
    {"name": "Quantum Laptop 15", "category": "Computing", "price": 1499},
    {"name": "Secure Router Pro", "category": "Networking", "price": 199},
    {"name": "Ultra Smartwatch v2", "category": "Wearables", "price": 349},
    {"name": "Eco Charging Pad", "category": "Accessories", "price": 49},
    {"name": "Noise-Cancelling Bud S", "category": "Audio", "price": 129},
    {"name": "4K HDR Monitor", "category": "Display", "price": 599}
]

DEVICES = ["iPhone 14 Pro", "Samsung Galaxy S23", "MacBook Pro M2", "Dell XPS 13", "iPad Air", "Windows Desktop", "Android Tablet"]
ERROR_CODES = ["ERR_CONNECTION_TIMED_OUT", "ERR_AUTH_FAILED_401", "ERR_DB_TIMEOUT_504", "ERR_RESOURCE_NOT_FOUND_404", "ERR_SSL_HANDSHAKE_FAILED", "ERR_INVALID_LICENSE_KEY"]
FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Casey", "Jamie", "Riley", "Robin", "Drew", "Skyler", "Cameron", "Chris", "Pat", "Amelia", "Benjamin", "Chloe", "Daniel", "Emily", "Fiona", "George", "Hannah", "Ian", "Julia", "Kevin", "Lily", "Matthew", "Natalie", "Owen", "Penelope", "Quincy", "Ruby", "Samuel", "Victor", "Wyatt", "Xavier", "Yvonne", "Zachary"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Adams", "Baker", "Carter", "Evans", "Fisher", "Green", "Hill", "Irvin", "King", "Lewis", "Nelson", "Ortiz", "Perez", "Quinn", "Robinson", "Thomas", "Underwood", "Vance", "Watson", "Xu", "Young", "Zimmerman"]

TEMPLATES = {
    "Billing": [
        "Hi, I noticed a double charge of ${AMOUNT} on my credit card for order {ORDER_ID}. Can you please issue a refund?",
        "I was billed twice for my subscription. The invoice shows {ORDER_ID} for {AMOUNT} but I only authorized one charge.",
        "My card on file was charged {AMOUNT} but I cancelled my plan last week. Refund needed ASAP.",
        "Could you send me the invoice for my purchase of {PRODUCT}? Order ID is {ORDER_ID}.",
        "I'm trying to buy {PRODUCT} from {CITY}, but the payment keeps failing with billing code {ERROR_CODE}.",
        "Why was my account charged {AMOUNT}? I didn't purchase anything recently. Order reference is {ORDER_ID}.",
        "I need to update my billing address from {CITY} to my new home. Can you help me?",
        "I am requesting a refund for order {ORDER_ID} since the product arrived damaged. The total amount was ${AMOUNT}."
    ],
    "Technical": [
        "My {DEVICE} is throwing error {ERROR_CODE} when trying to connect to the server. Please help.",
        "The software crashes on startup on my {DEVICE}. It worked fine yesterday but now displays {ERROR_CODE}.",
        "I am unable to login on my {DEVICE}. It keeps spinning and says {ERROR_CODE}.",
        "How do I configure the firmware update on the {PRODUCT}? I'm getting a connection failure.",
        "I'm experiencing slow speeds in {CITY} using {DEVICE}. Is there an outage in my area?",
        "The secure setup for {PRODUCT} fails with code {ERROR_CODE}. I have tried restarting the router already.",
        "Can you help me debug the sync issue on {DEVICE}? It shows database error {ERROR_CODE} constantly.",
        "The application screen is completely frozen on my {DEVICE} after the latest update."
    ],
    "Shipping": [
        "Where is my package? The tracking for order {ORDER_ID} has not updated in 3 days. It was supposed to ship to {CITY}.",
        "I ordered {PRODUCT} but the delivery address is incorrect. Can I change it to {CITY}?",
        "The delivery status for order {ORDER_ID} says delivered, but I haven't received anything at my home in {CITY}.",
        "How long does shipping to {CITY} take for the {PRODUCT}? I need it before Friday.",
        "My order {ORDER_ID} was shipped via express delivery but it's delayed. Can I get a shipping fee refund?",
        "I received a notification that order {ORDER_ID} was returned to sender. Why did that happen?",
        "Can you provide the tracking number or link for my shipping order {ORDER_ID}?",
        "The package containing my {PRODUCT} is stuck at the local hub in {CITY}. Please check on it."
    ],
    "Product": [
        "Does {PRODUCT} support integration with {DEVICE}? I couldn't find this spec in the manual.",
        "I am looking at {PRODUCT} and wanted to know if it comes with a warranty. What does the warranty cover?",
        "Can you tell me the physical dimensions and weight of {PRODUCT}? I want to make sure it fits my desk.",
        "Is {PRODUCT} compatible with Mac and Windows? I plan to use it on my {DEVICE}.",
        "What is the battery life of {PRODUCT}? Does it support fast charging on {DEVICE}?",
        "How do I clean and maintain my {PRODUCT}? The user guide doesn't explain this well.",
        "Is {PRODUCT} currently in stock in any physical stores near {CITY}?",
        "I want to compare {PRODUCT} with other models. What are the key advantages of this version?"
    ],
    "Account": [
        "I forgot my password and the reset email is not arriving. My email is {EMAIL}.",
        "How can I delete my profile? I no longer wish to use this service. My customer name is {NAME}.",
        "My profile has been suspended for security reasons, it says error {ERROR_CODE}. How can I unlock it?",
        "I need to change my registered email from {EMAIL} to another address.",
        "Someone tried to access my profile from a different location. Can you verify my login history?",
        "How do I enable two-factor authentication (2FA) on my {DEVICE}? I want to secure my account.",
        "I cannot change my username. It keeps saying invalid format. Please help.",
        "I want to merge my old profile with my new one. Both are under {NAME}."
    ]
}

def generate_full_dataset(customer_count=10000):
    customers = []
    orders = []
    queries = []
    mapped_queries_for_training = []
    
    print(f"Generating {customer_count} customers in Python...")
    
    # 1. Customers
    for i in range(customer_count):
        cust_id = f"CUST-{10000 + i}"
        loc = random.choice(CITIES_WITH_STATES)
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@example.com"
        phone = f"+1-555-{random.randint(100, 999):03d}-{random.randint(1000, 9999):04d}"
        
        # Account creation date over last 3 years
        days_ago = random.randint(30, 1000)
        acct_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        segment = random.choice(["Standard", "Premium", "VIP"])
        history_count = random.randint(0, 5)
        
        customers.append({
            "customer_id": cust_id,
            "customer_name": name,
            "email": email,
            "phone": phone,
            "city": loc["city"],
            "state": loc["state"],
            "account_created_date": acct_date,
            "customer_segment": segment,
            "support_history_count": history_count
        })
        
        # 2. Orders for this customer
        num_orders = random.randint(1, 5)
        for _ in range(num_orders):
            order_id = f"ORD-{random.randint(100000, 999999)}"
            prod = random.choice(PRODUCTS)
            
            # Order date is after account creation
            o_days_after = random.randint(1, 20)
            order_date = (datetime.strptime(acct_date, "%Y-%m-%d") + timedelta(days=o_days_after)).strftime("%Y-%m-%d")
            
            payment_status = random.choice(["Paid", "Paid", "Paid", "Paid", "Paid", "Unpaid", "Refunded", "Processing Refund"])
            delivery_status = random.choice(["Delivered", "Delivered", "Delivered", "In Transit", "Not Shipped", "Cancelled"])
            
            expected_delivery_date = (datetime.strptime(order_date, "%Y-%m-%d") + timedelta(days=random.randint(3, 7))).strftime("%Y-%m-%d")
            
            refund_status = "None"
            if payment_status == "Refunded":
                refund_status = "Completed"
            elif payment_status == "Processing Refund":
                refund_status = "Processing"
                
            warranty_status = "No Warranty"
            if prod["category"] != "Accessories":
                order_age = (datetime.now() - datetime.strptime(order_date, "%Y-%m-%d")).days
                warranty_status = "Expired" if order_age > 365 else "Active"
                
            orders.append({
                "order_id": order_id,
                "customer_id": cust_id,
                "product_name": prod["name"],
                "product_category": prod["category"],
                "order_date": order_date,
                "order_amount": prod["price"],
                "payment_status": payment_status,
                "delivery_status": delivery_status,
                "expected_delivery_date": expected_delivery_date,
                "refund_status": refund_status,
                "warranty_status": warranty_status
            })
            
    print(f"Generated {len(orders)} orders. Generating 10,000 support queries...")
    
    # 3. Support Queries
    categories = list(TEMPLATES.keys())
    sentiments = ["Positive", "Neutral", "Negative"]
    priorities = ["low", "medium", "high", "urgent"]
    resolution_statuses = ["new", "pending", "in_progress", "resolved", "escalated"]
    
    # Pre-index orders by customer_id for fast lookup
    orders_by_customer = {}
    for o in orders:
        c_id = o["customer_id"]
        if c_id not in orders_by_customer:
            orders_by_customer[c_id] = []
        orders_by_customer[c_id].append(o)
        
    for i in range(10000):
        tkt_id = f"TKT-{100000 + i}"
        customer = random.choice(customers)
        cust_id = customer["customer_id"]
        
        cust_orders = orders_by_customer.get(cust_id, [])
        if cust_orders:
            order = random.choice(cust_orders)
        else:
            order = {"order_id": "ORD-UNKNOWN", "product_name": "Pro Headphones X", "order_amount": 299}
            
        category = random.choice(categories)
        sentiment = random.choice(sentiments)
        priority = random.choice(priorities)
        
        # Created date over last 30 days
        created_date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        resolution_status = random.choice(resolution_statuses)
        escalation_status = "Yes" if resolution_status == "escalated" else "No"
        
        device = random.choice(DEVICES)
        error_code = random.choice(ERROR_CODES)
        
        templates = TEMPLATES[category]
        query_text = random.choice(templates)
        
        # Replace template placeholders
        query_text = (query_text
                      .replace("{ORDER_ID}", order["order_id"])
                      .replace("{CITY}", customer["city"])
                      .replace("{PRODUCT}", order["product_name"])
                      .replace("{AMOUNT}", str(order["order_amount"]))
                      .replace("{DEVICE}", device)
                      .replace("{ERROR_CODE}", error_code)
                      .replace("{NAME}", customer["customer_name"])
                      .replace("{EMAIL}", customer["email"]))
        
        if random.random() < 0.1:
            query_text = query_text.lower()
        if random.random() < 0.05:
            query_text += " please assist asap."
            
        queries.append({
            "ticket_id": tkt_id,
            "customer_id": cust_id,
            "order_id": order["order_id"],
            "customer_query": query_text,
            "query_category": category,
            "sentiment": sentiment,
            "priority": priority,
            "created_date": created_date,
            "resolution_status": resolution_status,
            "escalation_status": escalation_status
        })
        
        # Map for training pipeline compatibility
        mapped_queries_for_training.append({
            "id": tkt_id,
            "query": query_text,
            "category": category,
            "orderId": order["order_id"],
            "city": customer["city"],
            "product": order["product_name"],
            "amount": order["order_amount"],
            "device": device,
            "errorCode": error_code,
            "customerName": customer["customer_name"],
            "sentiment": sentiment,
            "timestamp": datetime.strptime(created_date, "%Y-%m-%d").isoformat() + "Z"
        })
        
    return {
        "customers": customers,
        "orders": orders,
        "queries": queries,
        "mapped_queries_for_training": mapped_queries_for_training
    }

def save_full_dataset_to_csv_and_json(data, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    root_data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(root_data_dir, exist_ok=True)
    
    # 1. Customers.csv
    cust_fields = ["Customer ID", "Customer Name", "Email", "Phone Number", "City", "State", "Account Created Date", "Customer Segment", "Support History Count"]
    for d in [target_dir, root_data_dir]:
        with open(os.path.join(d, "customers.csv"), mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cust_fields)
            for c in data["customers"]:
                writer.writerow([
                    c["customer_id"],
                    c["customer_name"],
                    c["email"],
                    c["phone"],
                    c["city"],
                    c["state"],
                    c["account_created_date"],
                    c["customer_segment"],
                    c["support_history_count"]
                ])
                
    # 2. Orders.csv
    order_fields = ["Order ID", "Customer ID", "Product Name", "Product Category", "Order Date", "Order Amount", "Payment Status", "Delivery Status", "Expected Delivery Date", "Refund Status", "Warranty Status"]
    for d in [target_dir, root_data_dir]:
        with open(os.path.join(d, "orders.csv"), mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(order_fields)
            for o in data["orders"]:
                writer.writerow([
                    o["order_id"],
                    o["customer_id"],
                    o["product_name"],
                    o["product_category"],
                    o["order_date"],
                    o["order_amount"],
                    o["payment_status"],
                    o["delivery_status"],
                    o["expected_delivery_date"],
                    o["refund_status"],
                    o["warranty_status"]
                ])
                
    # 3. Support_queries.csv
    query_fields = ["Ticket ID", "Customer ID", "Order ID", "Customer Query", "Query Category", "Sentiment", "Priority", "Created Date", "Resolution Status", "Escalation Status"]
    for d in [target_dir, root_data_dir]:
        with open(os.path.join(d, "support_queries.csv"), mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(query_fields)
            for q in data["queries"]:
                writer.writerow([
                    q["ticket_id"],
                    q["customer_id"],
                    q["order_id"],
                    q["customer_query"],
                    q["query_category"],
                    q["sentiment"],
                    q["priority"],
                    q["created_date"],
                    q["resolution_status"],
                    q["escalation_status"]
                ])
                
    # 4. JSON & CSV mapping for pipeline compatibility
    with open(os.path.join(target_dir, "customer_support_queries.json"), mode="w", encoding="utf-8") as f:
        json.dump(data["mapped_queries_for_training"], f, indent=2)
        
    compat_fields = ["id", "query", "category", "orderId", "city", "product", "amount", "device", "errorCode", "customerName", "sentiment", "timestamp"]
    with open(os.path.join(target_dir, "customer_support_queries.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(compat_fields)
        for q in data["mapped_queries_for_training"]:
            writer.writerow([
                q["id"],
                q["query"],
                q["category"],
                q["orderId"],
                q["city"],
                q["product"],
                q["amount"],
                q["device"],
                q["errorCode"],
                q["customerName"],
                q["sentiment"],
                q["timestamp"]
            ])
            
    print(f"Data generation complete! Written successfully in Python.")

if __name__ == "__main__":
    from utils import DATA_DIR
    dataset = generate_full_dataset(10000)
    save_full_dataset_to_csv_and_json(dataset, DATA_DIR)
