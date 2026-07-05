/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Customer {
  id: string; // Customer ID
  name: string;
  email: string;
  phone: string;
  city: string;
}

export interface Order {
  orderId: string;
  customerId: string;
  productName: string;
  orderDate: string;
  deliveryStatus: "Not Shipped" | "In Transit" | "Delivered" | "Cancelled";
  paymentStatus: "Paid" | "Unpaid" | "Refunded" | "Processing Refund";
  orderAmount: number;
  shippingCity: string;
  expectedDeliveryDate: string;
  refundStatus: "None" | "Processing" | "Completed" | "Failed";
  warrantyStatus: "Active" | "Expired" | "No Warranty";
}

export interface SupportRecord {
  customer_id: string;
  customer_name: string;
  email: string;
  phone: string;
  city: string;
  order_id: string;
  product_name: string;
  amount: number;
  problem_category: "billing" | "technical" | "shipping" | "product_info" | "account";
  status: "new" | "pending" | "in_progress" | "resolved" | "escalated";
  issue_description: string;
  resolution: string;
  priority: "low" | "medium" | "high" | "urgent";
  error_code?: string;
  device?: string;
}

// Clear out all hardcoded arrays. All customer details are loaded dynamically from the generated CSV files
export const CUSTOMERS: Customer[] = [];
export const ORDERS: Order[] = [];
export const SUPPORT_RECORDS: SupportRecord[] = [];

export interface SampleQuery {
  label: string;
  query: string;
  category: string;
  customerId: string;
  customerName: string;
  email: string;
  orderId?: string;
}

export const SAMPLE_QUERIES: SampleQuery[] = [
  {
    label: "Delayed Laptop Tracking",
    query: "My shipment tracking has not updated in 3 days. It was supposed to ship to Austin. Where is my Quantum Laptop 15?",
    category: "Shipping",
    customerId: "CUST-10041",
    customerName: "Sam Evans",
    email: "sam.evans@example.com"
  },
  {
    label: "Duplicate Billing Hold",
    query: "I noticed a duplicate charge of $1499 on my credit card. Can you please initiate a gateway refund for this transaction?",
    category: "Billing",
    customerId: "CUST-10115",
    customerName: "Jordan Carter",
    email: "jordan.carter@example.com"
  },
  {
    label: "Firmware Error Crash",
    query: "The Secure Router Pro crashes on startup on my Windows Desktop. The system log throws error ERR_SSL_HANDSHAKE_FAILED.",
    category: "Technical",
    customerId: "CUST-10254",
    customerName: "Pat Hill",
    email: "pat.hill@example.com"
  }
];
