"""
=============================================================
  PHISHING URL DETECTOR — FastAPI Application
  
  Endpoints:
    GET  /          → health check
    POST /predict   → predict if a URL is phishing
    GET  /features  → list all features and their descriptions
=============================================================
"""

import time
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator

from app.feature_engineering import extract_all_features, ALL_FEATURE_COLS

# ─────────────────────────────────────────────
#  LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "phishing_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print(f"[✓] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        f"Run 'python train.py' first to train and save the model."
    )


# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Phishing URL Detector",
    description=(
        "An ML-powered API that detects phishing URLs using "
        "structural URL features and page content analysis. "
        "Part of the 'AI for Digital Trust' portfolio project."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        if len(v) > 2048:
            raise ValueError("URL too long (max 2048 characters)")
        return v


class FeatureBreakdown(BaseModel):
    name: str
    value: float
    layer: str


class PredictResponse(BaseModel):
    url: str
    prediction: str          # "phishing" | "legitimate"
    confidence: float        # probability of being phishing (0.0 – 1.0)
    risk_level: str          # "HIGH" | "MEDIUM" | "LOW"
    features_used: str       # "url+content" | "url_only"
    processing_time_ms: float
    warning: str | None      # shown if page was unreachable
    top_signals: list[FeatureBreakdown]  # top 5 suspicious features


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def get_risk_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    elif confidence >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def get_top_signals(feature_dict: dict, n: int = 5) -> list[FeatureBreakdown]:
    """
    Returns the top N features that contributed most to the prediction.
    Uses feature importances from the trained model.
    """
    importances = dict(zip(ALL_FEATURE_COLS, model.feature_importances_))

    LAYER_1_COLS_SET = set([
        "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens",
        "nb_at", "nb_qm", "nb_and", "nb_or", "nb_eq", "nb_underscore",
        "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon",
        "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www",
        "nb_com", "nb_dslash", "http_in_path", "https_token",
        "ratio_digits_url", "ratio_digits_host", "punycode", "port",
        "tld_in_path", "tld_in_subdomain", "abnormal_subdomain",
        "nb_subdomains", "prefix_suffix", "random_domain",
        "shortening_service", "path_extension", "nb_redirection",
        "nb_external_redirection", "length_words_raw", "char_repeat",
        "shortest_words_raw", "shortest_word_host", "shortest_word_path",
        "longest_words_raw", "longest_word_host", "longest_word_path",
        "avg_words_raw", "avg_word_host", "avg_word_path", "phish_hints",
        "domain_in_brand", "brand_in_subdomain", "brand_in_path",
        "suspecious_tld", "statistical_report",
    ])

    # Score each feature: importance × |value| (non-zero features only)
    scored = []
    for name, value in feature_dict.items():
        if name in importances and value not in (-1, 0):
            score = importances[name] * abs(float(value))
            layer = "url_structure" if name in LAYER_1_COLS_SET else "page_content"
            scored.append((name, value, layer, score))

    scored.sort(key=lambda x: x[3], reverse=True)

    return [
        FeatureBreakdown(name=s[0], value=float(s[1]), layer=s[2])
        for s in scored[:n]
    ]


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Phishing URL Detector",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Analyse a URL and predict whether it is phishing or legitimate.

    - Extracts 56 URL structure features instantly (Layer 1)
    - Fetches the page and extracts 24 content features (Layer 2)
    - Falls back to URL-only features if the page is unreachable
    - Returns prediction, confidence score, risk level, and key signals
    """
    start = time.time()

    try:
        feature_vector, features_used, feature_dict = extract_all_features(
            request.url
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}"
        )

    # Predict
    feature_array = np.array(feature_vector).reshape(1, -1)
    prediction_int = model.predict(feature_array)[0]
    confidence     = float(model.predict_proba(feature_array)[0][1])  # P(phishing)

    prediction = "phishing" if prediction_int == 1 else "legitimate"
    risk_level = get_risk_level(confidence)

    elapsed_ms = (time.time() - start) * 1000

    warning = None
    if features_used == "url_only":
        warning = (
            "Page was unreachable. Prediction is based on URL structure only. "
            "Confidence may be lower than usual."
        )

    top_signals = get_top_signals(feature_dict)

    return PredictResponse(
        url=request.url,
        prediction=prediction,
        confidence=round(confidence, 4),
        risk_level=risk_level,
        features_used=features_used,
        processing_time_ms=round(elapsed_ms, 2),
        warning=warning,
        top_signals=top_signals,
    )


@app.get("/features", tags=["Info"])
def list_features():
    """
    Returns the list of all 80 features used by the model,
    grouped by layer with descriptions.
    """
    return {
        "total_features": len(ALL_FEATURE_COLS),
        "layers": {
            "layer_1_url_structure": {
                "count": 56,
                "description": "Extracted from URL string only. No network calls.",
                "features": ALL_FEATURE_COLS[:56],
            },
            "layer_2_page_content": {
                "count": 24,
                "description": "Extracted by fetching and parsing the live page HTML.",
                "features": ALL_FEATURE_COLS[56:],
            },
        },
    }
