# NetSentinel AI
### AI-Powered Network Traffic Intelligence, Intrusion Detection & Congestion Prediction

---

## Project Overview

**NetSentinel AI** is a modular Computer Networks project engineered to provide real-time visibility and predictive analytics on network activity.

### Full Objective

NetSentinel AI is designed to:

1. **Analyze and visualize network traffic** — Extracting actionable flow and packet statistics to understand bandwidth, protocol distribution, and traffic patterns.
2. **Predict network congestion using machine learning** — Proactively anticipating network bottlenecks before service degradation occurs.
3. **Attempt intrusion detection as an advanced cybersecurity module** — Identifying anomalous and potentially malicious traffic patterns.

---

## Development Phases

Development is structured into strictly staged phases to ensure stability and reliable delivery:

- **Phase 1 — Network Traffic Intelligence** *(Guaranteed Core)*:
  Telemetry ingestion, flow processing, traffic statistical metrics, and baseline visualization.
- **Phase 2 — AI-Based Network Congestion Prediction** *(Guaranteed Core)*:
  Feature extraction from traffic flows, time-series/predictive modeling for network bottleneck and latency forecasting.
- **Phase 3 — Intrusion Detection** *(Advanced / Optional)*:
  Anomaly and security threat classification.
- **Phase 4 — Final Integration & Polish**:
  End-to-end dashboard integration, reporting, and performance evaluation.

---

## Important Development Principle

- **Phase 1 and Phase 2 are the guaranteed core of the project.**
- **Phase 3 is an advanced module and must never break or compromise the completed Phase 1 + Phase 2 system.**
- The project remains fully functional, testable, and demonstrable even if Phase 3 is not integrated.

---

## Data Storage Strategy

- **Supabase** will be used as the persistent storage layer for the network traffic dataset.
- **Schema Design Postponement:** The final database schema will be designed *after* the Phase 1 traffic engine is finalized to ensure schema fields accurately match real engine telemetry without guesswork or redundant migrations.

---

## Repository Structure

```text
NetSentinel-AI/
│
├── backend/
│   ├── app.py              # Flask server and API endpoints
│   └── __init__.py
│
├── frontend/
│   ├── index.html          # Lightweight status dashboard
│   ├── style.css           # Styling
│   └── script.js           # API health integration
│
├── data/                   # Dataset storage directory
├── models/                 # Model artifacts storage directory
├── utils/                  # Utility and client helpers
│   ├── __init__.py
│   └── supabase_client.py  # Safe Supabase configuration loader
│
├── tests/                  # Automated test suite
│   ├── __init__.py
│   └── test_health.py      # Health endpoint unit tests
│
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore configuration
├── requirements.txt        # Initial project dependencies
└── README.md               # Project documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (Current environment: Python 3.14)
- Git

### 2. Environment Setup

Clone the repository and create a Python virtual environment:

```bash
# Navigate to project directory
cd NetSentinel-AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Initial Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Copy `.env.example` to `.env` (optional for initial setup; credentials will be populated once Supabase project is provisioned):

```bash
cp .env.example .env
```

### 5. Running the Backend

```bash
python backend/app.py
```

The server starts at `http://127.0.0.1:5000`.

- Health Endpoint: `http://127.0.0.1:5000/api/health`
- Frontend Placeholder: `http://127.0.0.1:5000/`

### 6. Running Tests

```bash
python -m unittest discover tests
```
