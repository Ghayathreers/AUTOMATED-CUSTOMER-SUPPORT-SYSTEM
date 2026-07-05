# -*- coding: utf-8 -*-
"""
High-Performance programmatically synthetic dataset generator for Customer Support AI.
Generates 10,000 highly structured, cohesive customer profiles, matching orders, and tickets.
"""

import os
import csv
import json
import random
from datetime import datetime, timedelta

from utils import DATA_DIR, WORKSPACE_DATA_DIR

# --- MOCK DATA POOLS ---
FIRST_NAMES = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
               "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
               "Christopher", "Nancy", "Daniel", "Lisa", "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson",
              "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White",
              "Lopez", "Lee", "Gonzalez", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Perez", "Hall"]

CITIES = [
    ("Austin", "TX"), ("San Francisco", "CA"), ("New York", "NY"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Miami", "FL"), ("Chicago", "IL"), ("Boston", "MA"),
    ("Portland", "OR"), ("Los Angeles", "CA"), ("Atlanta", "GA"), ("Phoenix", "AZ"),
    ("Dallas", "TX"), ("San Diego", "CA"), ("Salt Lake City", "UT"), ("Nashville", "TN")
]

SEGMENTS = ["Retail Customer", "Premium VIP", "Enterprise Corporate", "SME Business"]

PRODUCTS = [
    ("Secure Router Pro", "Hardware", 199.99),
    ("Mesh Extender Dual-Band", "Hardware", 89.99),
    ("Gigabit Switch 8-Port", "Hardware", 49.99),
    ("SaaS Cloud Firewall Plus", "Subscription", 29.99),
    ("Enterprise VPN Client license", "Subscription", 14.99),
    ("Premium Support SLA Ticket", "Subscription", 9.99)
]

QUERY_TEMPLATES = [
    # Billing templates
    ("I noticed a double charge of {amount} on my credit card for {product}. Please issue a refund.", "Billing", "Negative", "High", "No"),
    ("Can you please send me the tax invoice for order {order_id}? I need it for my expense claim.", "Billing", "Neutral", "Medium", "No"),
    ("Why was my credit card charged {amount} without authorization? I want a refund immediately.", "Billing", "Negative", "High", "No"),
    ("My billing subscription renewal failed with error {error_code}. Can you assist with the payment?", "Billing", "Negative", "Medium", "No"),
    ("I was overcharged for {product}. The catalog says it is on discount but I paid full price.", "Billing", "Negative", "Medium", "No"),
    
    # Technical templates
    ("My {product} is crashing constantly. The system log shows error code {error_code}.", "Technical", "Negative", "High", "Yes"),
    ("Can you provide instructions on how to install the firmware updates for my secure router pro?", "Technical", "Neutral", "Medium", "No"),
    ("I am getting a timeout exception ERR_DB_TIMEOUT_504 when attempting to login to my account profile.", "Technical", "Negative", "High", "Yes"),
    ("The mobile application is freezing on the splash screen after the latest software update.", "Technical", "Negative", "Medium", "No"),
    ("I need help with port forwarding settings on my dual-band mesh extender.", "Technical", "Neutral", "Medium", "No"),
    
    # Shipping templates
    ("My order {order_id} has not arrived yet. It has been in transit for two weeks. Where is it?", "Shipping", "Negative", "High", "No"),
    ("Can you update the shipping address for order {order_id} to 120 Main Street, Austin TX?", "Shipping", "Neutral", "Medium", "No"),
    ("The tracking link for my parcel is broken and shows no updates from the logistics courier.", "Shipping", "Negative", "Medium", "No"),
    ("My package arrived yesterday but the retail box was damaged and the secure router pro is cracked.", "Shipping", "Negative", "High", "Yes"),
    ("When will my order ship? The expected delivery date was supposed to be yesterday.", "Shipping", "Negative", "Medium", "No"),
    
    # Product templates
    ("Are the specs of secure router pro dual-band or tri-band? I need to verify specifications.", "Product", "Neutral", "Low", "No"),
    ("Is the premium support SLA license compatible with older gigabit switches?", "Product", "Neutral", "Low", "No"),
    ("What are the exact dimensions and weight of the mesh extender hardware?", "Product", "Neutral", "Low", "No"),
    ("Does the Secure Router Pro come with a standard replacement warranty?", "Product", "Neutral", "Low", "No"),
    ("Can I use the Enterprise VPN client on multiple devices simultaneously?", "Product", "Neutral", "Medium", "No")
]

ERROR_CODES = ["ERR_DB_TIMEOUT_504", "ERR_AUTH_FAILED_401", "ERR_CONN_RESET_104", "ERR_SYS_CRASH_500"]

def generate_full_dataset(num_customers: int = 10000) -> dict:
    random.seed(42) # Deterministic generation
    
    customers = []
    orders = []
    queries = []

    start_date = datetime(2025, 1, 1)

    for i in range(1, num_customers + 1):
        # 1. Generate Customer Profile
        cust_id = f"CUST-{i:06d}"
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{random.randint(10, 99)}@example.com"
        phone = f"+1-{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"
        city, state = random.choice(CITIES)
        created_days_ago = random.randint(10, 365)
        created_date = start_date + timedelta(days=created_days_ago)
        segment = random.choice(SEGMENTS)
        support_count = random.randint(0, 5)

        customers.append({
            "Customer ID": cust_id,
            "Customer Name": name,
            "Email": email,
            "Phone Number": phone,
            "City": city,
            "State": state,
            "Account Created Date": created_date.strftime("%Y-%m-%d"),
            "Customer Segment": segment,
            "Support History Count": support_count
        })

        # 2. Generate Orders for Customer (1 to 5 orders)
        num_orders = random.randint(1, 5)
        cust_orders = []
        for j in range(1, num_orders + 1):
            order_id = f"ORD-{i:06d}-{j}"
            prod_name, prod_cat, prod_price = random.choice(PRODUCTS)
            
            # Order Date must be after account creation
            order_days_after = random.randint(1, 30)
            order_date = created_date + timedelta(days=order_days_after)
            
            # Logistics status
            pay_status = random.choices(["Paid", "Pending", "Failed"], weights=[0.85, 0.10, 0.05])[0]
            del_status = "Unshipped"
            if pay_status == "Paid":
                del_status = random.choices(["Delivered", "In Transit", "Unshipped"], weights=[0.70, 0.20, 0.10])[0]

            expected_date = order_date + timedelta(days=5)
            
            refund_status = "N/A"
            if pay_status == "Paid" and random.random() < 0.08:
                refund_status = random.choice(["Requested", "Refunded"])

            warranty_status = "Active" if prod_cat == "Hardware" and random.random() < 0.90 else "N/A"

            order_record = {
                "Order ID": order_id,
                "Customer ID": cust_id,
                "Product Name": prod_name,
                "Product Category": prod_cat,
                "Order Date": order_date.strftime("%Y-%m-%d"),
                "Order Amount": prod_price,
                "Payment Status": pay_status,
                "Delivery Status": del_status,
                "Expected Delivery Date": expected_date.strftime("%Y-%m-%d"),
                "Refund Status": refund_status,
                "Warranty Status": warranty_status
            }
            orders.append(order_record)
            cust_orders.append(order_record)

        # 3. Generate Support Query for Customer (every customer has exactly 1 linked ticket in this dataset)
        ticket_id = f"TKT-{i:06d}"
        linked_order = random.choice(cust_orders)
        
        # Select query template
        tpl_text, q_cat, q_sent, q_prio, q_esc = random.choice(QUERY_TEMPLATES)
        
        # Populate template parameters
        err = random.choice(ERROR_CODES)
        amt = f"${linked_order['Order Amount']:.2f}"
        prod = linked_order['Product Name']
        ord_ref = linked_order['Order ID']

        query_text = tpl_text.format(amount=amt, product=prod, order_id=ord_ref, error_code=err)

        # Resolution Status mapping
        res_status = random.choices(["resolved", "pending", "escalated"], weights=[0.60, 0.30, 0.10])[0]
        if q_esc == "Yes":
            res_status = "escalated"

        # Ticket date must be after order date
        order_dt = datetime.strptime(linked_order["Order Date"], "%Y-%m-%d")
        ticket_days_after = random.randint(1, 15)
        ticket_date = order_dt + timedelta(days=ticket_days_after)

        queries.append({
            "Ticket ID": ticket_id,
            "Customer ID": cust_id,
            "Order ID": ord_ref,
            "Customer Query": query_text,
            "Query Category": q_cat,
            "Sentiment": q_sent,
            "Priority": q_prio,
            "Created Date": ticket_date.strftime("%Y-%m-%d"),
            "Resolution Status": res_status,
            "Escalation Status": "Yes" if res_status == "escalated" else "No"
        })

    return {
        "customers": customers,
        "orders": orders,
        "queries": queries
    }

def save_full_dataset_to_csv_and_json(dataset: dict, target_dir: str):
    # Ensure folder is created
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(WORKSPACE_DATA_DIR, exist_ok=True)

    # 1. Save Customers CSV
    for folder in [target_dir, WORKSPACE_DATA_DIR]:
        cust_path = os.path.join(folder, "customers.csv")
        with open(cust_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dataset["customers"][0].keys())
            writer.writeheader()
            writer.writerows(dataset["customers"])

        # 2. Save Orders CSV
        order_path = os.path.join(folder, "orders.csv")
        with open(order_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dataset["orders"][0].keys())
            writer.writeheader()
            writer.writerows(dataset["orders"])

        # 3. Save Queries CSV
        query_path = os.path.join(folder, "support_queries.csv")
        with open(query_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dataset["queries"][0].keys())
            writer.writeheader()
            writer.writerows(dataset["queries"])

        # 4. Save Support Queries JSON (for easy training parsing)
        json_path = os.path.join(folder, "customer_support_queries.json")
        with open(json_path, "w", encoding="utf-8") as f:
            # Map values to match exactly what ml_module.ts expects
            json_list = []
            for q in dataset["queries"]:
                cust_match = [c for c in dataset["customers"] if c["Customer ID"] == q["Customer ID"]][0]
                json_list.append({
                    "id": q["Ticket ID"],
                    "query": q["Customer Query"],
                    "category": q["Query Category"],
                    "sentiment": q["Sentiment"],
                    "priority": q["Priority"],
                    "customerName": cust_match["Customer Name"],
                    "orderId": q["Order ID"]
                })
            json.dump(json_list, f, indent=2)

    print(f"Programmatic dataset files successfully saved to: {target_dir} and {WORKSPACE_DATA_DIR}")

if __name__ == "__main__":
    ds = generate_full_dataset(10000)
    save_full_dataset_to_csv_and_json(ds, DATA_DIR)
