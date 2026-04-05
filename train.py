"""
=============================================================
  PHISHING URL DETECTOR — TRAINING SCRIPT
  Step 1: Run this once to train and save the model.
  Usage: python train.py
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "data/dataset.csv"
MODEL_PATH  = "models/phishing_model.joblib"
FEATURES_PATH = "models/feature_columns.joblib"

# ─────────────────────────────────────────────
#  LAYER 1 — URL STRUCTURE FEATURES (56)
#  These are extracted from the URL string only.
#  No network calls needed at inference time.
# ─────────────────────────────────────────────
LAYER_1_FEATURES = [
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
]

# ─────────────────────────────────────────────
#  LAYER 2 — PAGE CONTENT FEATURES (24)
#  These require fetching the live page HTML.
#  At inference time: computed if page is reachable,
#  otherwise filled with -1 (unreachable fallback).
# ─────────────────────────────────────────────
LAYER_2_FEATURES = [
    "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks",
    "ratio_nullHyperlinks", "nb_extCSS", "ratio_intRedirection",
    "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors",
    "login_form", "external_favicon", "links_in_tags", "submit_email",
    "ratio_intMedia", "ratio_extMedia", "sfh", "iframe", "popup_window",
    "safe_anchor", "onmouseover", "right_clic", "empty_title",
    "domain_in_title", "domain_with_copyright",
]

ALL_FEATURES = LAYER_1_FEATURES + LAYER_2_FEATURES


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  PHISHING URL DETECTOR — TRAINING")
print("=" * 60)

print(f"\n[1/5] Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"      Rows: {len(df):,}  |  Columns: {len(df.columns)}")
print(f"      Labels: {df['status'].value_counts().to_dict()}")


# ─────────────────────────────────────────────
#  PREPARE FEATURES & LABELS
# ─────────────────────────────────────────────
print(f"\n[2/5] Preparing features...")
print(f"      Layer 1 features: {len(LAYER_1_FEATURES)}")
print(f"      Layer 2 features: {len(LAYER_2_FEATURES)}")
print(f"      Total features:   {len(ALL_FEATURES)}")

X = df[ALL_FEATURES]
y = (df["status"] == "phishing").astype(int)  # 1=phishing, 0=legitimate

# Fill any missing values with -1 (signals unavailable feature)
X = X.fillna(-1)

print(f"      Feature matrix: {X.shape}")
print(f"      Missing values after fill: {X.isnull().sum().sum()}")


# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
print(f"\n[3/5] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ─────────────────────────────────────────────
#  TRAIN MODEL
# ─────────────────────────────────────────────
print(f"\n[4/5] Training Random Forest classifier...")
print(f"      This may take 1-2 minutes...")

model = RandomForestClassifier(
    n_estimators=200,        # 200 trees — good balance of accuracy vs speed
    max_depth=None,          # Grow full trees
    min_samples_split=5,     # Prevents overfitting on noise
    min_samples_leaf=2,      # Smoother decision boundaries
    class_weight="balanced", # Handles any residual class imbalance
    n_jobs=-1,               # Use all CPU cores
    random_state=42,
    verbose=0,
)
model.fit(X_train, y_train)
print(f"      Training complete.")


# ─────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────
print(f"\n[5/5] Evaluating model on test set...")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print(f"\n{'─'*60}")
print(f"  RESULTS")
print(f"{'─'*60}")
print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
print(f"  ROC-AUC Score: {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:")
print(f"                   Predicted")
print(f"                   Legit    Phishing")
print(f"  Actual Legit   [{cm[0][0]:^7} {cm[0][1]:^9}]")
print(f"  Actual Phish   [{cm[1][0]:^7} {cm[1][1]:^9}]")

# Top 10 most important features
feat_importance = pd.Series(
    model.feature_importances_, index=ALL_FEATURES
).sort_values(ascending=False)

print(f"\n  Top 10 Most Important Features:")
for i, (feat, imp) in enumerate(feat_importance.head(10).items(), 1):
    layer = "L1" if feat in LAYER_1_FEATURES else "L2"
    bar = "█" * int(imp * 200)
    print(f"  {i:2}. [{layer}] {feat:<35} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────
#  SAVE MODEL & FEATURE LIST
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(ALL_FEATURES, FEATURES_PATH)

print(f"\n{'─'*60}")
print(f"  Model saved  → {MODEL_PATH}")
print(f"  Features saved → {FEATURES_PATH}")
print(f"{'─'*60}")
print(f"\n  Next step: Run the API with:")
print(f"  uvicorn app.main:app --reload")
print(f"{'─'*60}\n")
