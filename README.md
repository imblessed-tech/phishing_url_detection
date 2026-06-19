# Phishing URL Detector

> An ML-powered REST API that detects phishing URLs using structural URL analysis and live page content inspection — deployed as a production-style FastAPI service.

Part of the **"AI for Digital Trust"** portfolio series.

---

## What This Project Proves

Most data scientists can train a model in a notebook. This project shows you can:

- **Engineer features from raw data** (URL parsing + live HTML scraping)
- **Build a real REST API** with input validation, structured responses, and graceful fallbacks
- **Containerise with Docker** for reproducible, portable deployment
- **Deploy to production** on a free cloud tier

---

## How It Works

Phishing URLs leave structural fingerprints. This model learns those fingerprints
across two layers of features:

### Layer 1 — URL Structure (56 features)
Extracted instantly from the URL string. No network calls needed.

| Feature Group | Examples |
|---|---|
| Length signals | URL length, hostname length |
| Special characters | Count of `@`, `-`, `//`, `%`, `?` |
| Brand impersonation | Brand name in subdomain/path vs domain |
| Suspicious patterns | IP instead of domain, suspicious TLD, URL shorteners |
| Word statistics | Avg word length, longest/shortest word in path |

### Layer 2 — Page Content (24 features)
Extracted by fetching the live page HTML.

| Feature Group | Examples |
|---|---|
| Hyperlink ratios | % of internal vs external links |
| Form analysis | External form action, submit-to-email |
| Visual tricks | Hidden iframes, disabled right-click, popups |
| Page identity | Empty title, domain not in title/copyright |

**Graceful fallback:** If the page is unreachable (common for taken-down phishing sites),
the API falls back to Layer 1 features only and flags this in the response.

---

## Dataset

- **Source:** [Mendeley Data — Web Page Phishing Detection](https://data.mendeley.com/datasets/c2gw7fy2j4/3)
- **Size:** 11,430 URLs (5,715 phishing + 5,715 legitimate) — perfectly balanced
- **Features pre-extracted:** 87 total (we use 80: Layer 1 + Layer 2)
- **Layer 3 excluded:** WHOIS/DNS/Alexa features — unreliable at inference time

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/phishing-url-detector.git
cd phishing-url-detector
pip install -r requirements.txt
```

### 2. Download Dataset

Download the CSV from [Mendeley](https://data.mendeley.com/datasets/c2gw7fy2j4/3)
and place it at:
```
data/dataset.csv
```

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and prepare the dataset
- Train a Random Forest classifier on 80 features
- Evaluate on a held-out test set
- Save the model to `models/phishing_model.joblib`

Expected output:
```
  ROC-AUC Score: ~0.97
  Accuracy:      ~96%
```

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs

---

## Docker

### Build & Run

```bash
docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
```

### Or with Docker Compose

```bash
docker-compose up --build
```

---

## API Reference

### `POST /predict`

Analyse a URL and return a phishing prediction.

**Request:**
```json
{
  "url": "https://paypal-secure-login.tk/verify/account"
}
```

**Response:**
```json
{
  "url": "https://paypal-secure-login.tk/verify/account",
  "prediction": "phishing",
  "confidence": 0.97,
  "risk_level": "HIGH",
  "features_used": "url+content",
  "processing_time_ms": 3241.5,
  "warning": null,
  "top_signals": [
    { "name": "brand_in_subdomain", "value": 1.0, "layer": "url_structure" },
    { "name": "suspecious_tld",     "value": 1.0, "layer": "url_structure" },
    { "name": "login_form",         "value": 1.0, "layer": "page_content"  },
    { "name": "iframe",             "value": 1.0, "layer": "page_content"  },
    { "name": "phish_hints",        "value": 2.0, "layer": "url_structure" }
  ]
}
```

**Risk levels:**
- `HIGH` → confidence ≥ 80%
- `MEDIUM` → confidence 50–79%
- `LOW` → confidence < 50%

**Features used:**
- `url+content` → page was reachable, full feature set used
- `url_only` → page unreachable, URL features only (warning included)

---

### `GET /health`

```json
{ "status": "healthy", "model_loaded": true }
```

### `GET /features`

Returns all 80 features grouped by layer with descriptions.

---

## Deploy Free on Render.com

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and configures everything
5. Your API will be live at: `https://phishing-url-detector.onrender.com`

---

## Project Structure

```
phishing-url-detector/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  ← FastAPI app & endpoints
│   └── feature_engineering.py  ← Layer 1 + Layer 2 feature extraction
│
├── data/
│   ├── dataset.csv              ← Training data (download separately)
│   └── allbrands.txt            ← Brand list for impersonation detection
│
├── models/
│   └── phishing_model.joblib    ← Saved model (generated by train.py)
│
├── train.py                     ← Training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── render.yaml                  ← Free deployment config
└── README.md
```

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~96% |
| Precision (phishing) | ~96% |
| Recall (phishing) | ~96% |
| ROC-AUC | ~0.97 |

**Why Random Forest?**
Phishing signals are not individually conclusive — it's the combination that matters.
A long URL alone means nothing; but long URL + suspicious TLD + brand in subdomain +
external login form together is almost certainly phishing. Random Forest naturally
learns these feature interactions through its ensemble of decision trees.

---

## Portfolio Context

This is **Project 1** in the "AI for Digital Trust" series:

| # | Project | What it detects |
|---|---|---|
| P1 | **Phishing URL Detector** ← *this* | Malicious/fake websites |
| P2 | Deepfake Audio Detector | AI-generated fake voices |

---

## 📜 Data Attribution

Hannousse, Abdelhakim; Yahiouche, Salima (2021),
"Web page phishing detection", Mendeley Data, V3,
doi: 10.17632/c2gw7fy2j4.3
