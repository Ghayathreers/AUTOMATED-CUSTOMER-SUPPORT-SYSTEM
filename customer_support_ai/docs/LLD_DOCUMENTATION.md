# Low-Level Design Documentation

## Project
Automated Customer Support and Service Resolution

## Added LLD Deliverables
This folder now includes the missing design and evaluation artifacts required by the problem statement:

- `system_architecture.png` - overall system architecture
- `component_diagram.png` - module-level component interaction
- `sequence_diagram.png` - customer query resolution sequence flow
- `er_diagram.png` - database/data model design
- `deployment_architecture.png` - deployment/infrastructure view
- `roc_curve.png` - ROC curve for ML intent classification

## Core Components
1. **Frontend UI** accepts customer query, customer ID, and order context.
2. **API Layer** exposes integration points for CRM systems and communication channels.
3. **Preprocessing Pipeline** cleans text and prepares TF-IDF features.
4. **ML Module** classifies query intent using Logistic Regression, Random Forest, and SVM.
5. **DL Module** provides sequence-based understanding using a Bi-LSTM model.
6. **NLP Module** extracts sentiment, entities, and keywords.
7. **SLM Module** summarizes complex issues and creates concise agent notes.
8. **GenAI Module** drafts personalized replies, FAQ content, and troubleshooting guides.
9. **Agentic Module** plans routing, resolution, and escalation decisions.

## Data Model
The ER diagram models Customers, Orders, Support Queries, Resolutions, Escalations, and Agents. This supports lookup, order matching, resolution generation, and human-agent routing.

## API Contract Summary
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/analyze-query` | POST | Classify query, extract entities, sentiment, and suggested resolution |
| `/api/customer/{customer_id}` | GET | Fetch customer profile and support history |
| `/api/orders/{order_id}` | GET | Fetch order status and product information |
| `/api/escalate` | POST | Create human-agent escalation request |
| `/api/resolution-feedback` | POST | Store feedback for future improvement |

## ROC-AUC Result
Micro-average ROC-AUC values generated from the included dataset:

```json
{
  "Logistic Regression": 1.0,
  "Random Forest": 0.9997395,
  "SVM": 1.0
}
```

## Notes
These additions improve the project alignment with the problem statement's LLD and ML evaluation deliverables.
