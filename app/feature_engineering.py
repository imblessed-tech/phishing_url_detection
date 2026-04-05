"""
=============================================================
  FEATURE ENGINEERING — LAYER 1 + LAYER 2
  
  This module extracts features from a URL at inference time.
  It mirrors exactly what was computed during training so the
  feature vector matches what the model expects.

  Layer 1: URL string features  → always computed (no network)
  Layer 2: Page content features → computed if page is reachable
                                   fallback to -1 if unreachable
=============================================================
"""

import re
import math
import time
import requests
import tldextract
import urllib.parse
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pathlib import Path

# ─────────────────────────────────────────────
#  LOAD BRAND LIST (used for brand impersonation checks)
# ─────────────────────────────────────────────
BRANDS_PATH = Path(__file__).parent.parent / "data" / "allbrands.txt"

def _load_brands():
    try:
        with open(BRANDS_PATH, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

ALLBRANDS = _load_brands()

# Common phishing hint keywords found in malicious URLs
PHISH_HINTS = [
    'wp', 'login', 'includes', 'admin', 'content', 'site', 'images',
    'js', 'alibaba', 'css', 'myaccount', 'dropbox', 'themes', 'plugins',
    'signin', 'view', 'verify', 'secure', 'account', 'update', 'confirm',
    'banking', 'support', 'password', 'credential',
]

SHORTENING_SERVICES = re.compile(
    r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|'
    r'tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|'
    r'twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|'
    r'ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
    r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|'
    r'bit\.do|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|'
    r'tinyurl\.com|ity\.im|q\.gs|po\.st|bc\.vc|u\.to|j\.mp|buzurl\.com|'
    r'cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|'
    r'filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
    r'tr\.im|link\.zip\.net'
)

SUSPICIOUS_TLDS = {
    'fit', 'tk', 'gp', 'ga', 'work', 'ml', 'date', 'wang', 'men', 'icu',
    'online', 'click', 'country', 'stream', 'download', 'xin', 'racing',
    'jetzt', 'ren', 'mom', 'party', 'review', 'trade', 'accountants',
    'science', 'ninja', 'xyz', 'faith', 'zip', 'cricket', 'win',
    'accountant', 'realtor', 'top', 'christmas', 'gdn', 'link', 'asia',
    'club', 'exposed', 'website', 'media',
}


# ═════════════════════════════════════════════
#  LAYER 1 — URL STRUCTURE FEATURES
#  Input: URL string only
#  Output: dict of 56 numeric features
# ═════════════════════════════════════════════

def extract_layer1_features(url: str) -> dict:
    """
    Extract all URL-structure features from a URL string.
    No network calls. Always fast and reliable.
    """
    features = {}

    # Parse URL components
    parsed     = urlparse(url)
    extracted  = tldextract.extract(url)
    hostname   = parsed.netloc or ""
    path       = parsed.path or ""
    query      = parsed.query or ""
    scheme     = parsed.scheme or ""
    domain     = extracted.domain or ""
    subdomain  = extracted.subdomain or ""
    tld        = extracted.suffix or ""
    full_url   = url

    # Tokenise URL into words (split on delimiters)
    def tokenise(s):
        return [w for w in re.split(r'[-./\?=@&%:_]', s.lower()) if w]

    words_raw   = tokenise(domain + " " + subdomain + " " + path)
    words_host  = tokenise(domain + " " + subdomain)
    words_path  = tokenise(path)

    # ── Basic length features ──
    features["length_url"]      = len(full_url)
    features["length_hostname"] = len(hostname)

    # ── IP address in hostname ──
    ip_pattern = re.compile(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])'
        r'|((0x[0-9a-fA-F]{1,2})\.){3}(0x[0-9a-fA-F]{1,2})'
        r'|(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}'
    )
    features["ip"] = 1 if ip_pattern.search(full_url) else 0

    # ── Special character counts ──
    features["nb_dots"]        = full_url.count('.')
    features["nb_hyphens"]     = full_url.count('-')
    features["nb_at"]          = full_url.count('@')
    features["nb_qm"]          = full_url.count('?')
    features["nb_and"]         = full_url.count('&')
    features["nb_or"]          = full_url.count('|')
    features["nb_eq"]          = full_url.count('=')
    features["nb_underscore"]  = full_url.count('_')
    features["nb_tilde"]       = full_url.count('~')
    features["nb_percent"]     = full_url.count('%')
    features["nb_slash"]       = full_url.count('/')
    features["nb_star"]        = full_url.count('*')
    features["nb_colon"]       = full_url.count(':')
    features["nb_comma"]       = full_url.count(',')
    features["nb_semicolumn"]  = full_url.count(';')
    features["nb_dollar"]      = full_url.count('$')
    features["nb_space"]       = full_url.count(' ') + full_url.count('%20')

    # ── Keyword counts ──
    features["nb_www"] = sum(1 for w in words_raw if 'www' in w)
    features["nb_com"] = sum(1 for w in words_raw if 'com' in w)

    # ── Double slash after position 6 (redirection trick) ──
    dslash_positions = [m.start() for m in re.finditer('//', full_url)]
    features["nb_dslash"] = 1 if dslash_positions and dslash_positions[-1] > 6 else 0

    # ── HTTP in path (double protocol trick) ──
    features["http_in_path"] = 1 if 'http' in path.lower() else 0

    # ── HTTPS token in domain (fake security signal) ──
    features["https_token"] = 1 if 'https' in domain.lower() else 0

    # ── Digit ratios ──
    digits_url  = sum(c.isdigit() for c in full_url)
    digits_host = sum(c.isdigit() for c in hostname)
    features["ratio_digits_url"]  = digits_url / max(len(full_url), 1)
    features["ratio_digits_host"] = digits_host / max(len(hostname), 1)

    # ── Punycode (internationalised domain — often used to spoof) ──
    features["punycode"] = 1 if 'xn--' in full_url.lower() else 0

    # ── Non-standard port ──
    features["port"] = 1 if re.search(
        r'^[a-z][a-z0-9+\-.]*://[^/]*:([0-9]+)', full_url
    ) else 0

    # ── TLD in path or subdomain (double TLD trick) ──
    features["tld_in_path"]      = 1 if tld and tld in path else 0
    features["tld_in_subdomain"] = 1 if tld and tld in subdomain else 0

    # ── Abnormal subdomain (www2, www3 etc) ──
    features["abnormal_subdomain"] = 1 if re.search(
        r'^www[0-9]', subdomain
    ) else 0

    # ── Number of subdomains ──
    dot_count = len(re.findall(r'\.', full_url))
    if dot_count == 1:
        features["nb_subdomains"] = 1
    elif dot_count == 2:
        features["nb_subdomains"] = 2
    else:
        features["nb_subdomains"] = 3

    # ── Prefix/suffix separator (-) in domain ──
    features["prefix_suffix"] = 1 if re.findall(
        r'https?://[^\-]+-[^\-]+/', full_url
    ) else 0

    # ── Random-looking domain (high consonant ratio) ──
    def is_random_domain(d):
        if not d:
            return 0
        vowels = sum(c in 'aeiou' for c in d.lower())
        consonants = sum(c.isalpha() and c not in 'aeiou' for c in d.lower())
        ratio = consonants / max(len(d), 1)
        return 1 if ratio > 0.6 and len(d) > 6 else 0

    features["random_domain"] = is_random_domain(domain)

    # ── URL shortening service ──
    features["shortening_service"] = 1 if SHORTENING_SERVICES.search(full_url) else 0

    # ── File extension in path (e.g. .php, .exe) ──
    suspicious_ext = re.compile(r'\.(php|html|htm|exe|zip|rar|pdf|asp|aspx)$', re.I)
    features["path_extension"] = 1 if suspicious_ext.search(path) else 0

    # ── Redirection count ──
    features["nb_redirection"]          = full_url.count('//')
    features["nb_external_redirection"] = max(0, full_url.count('//') - 1)

    # ── Word-based statistical features ──
    features["length_words_raw"]   = len(words_raw)
    features["char_repeat"]        = max(
        (full_url.count(c) for c in set(full_url) if c.isalpha()), default=0
    )
    features["shortest_words_raw"] = min((len(w) for w in words_raw), default=0)
    features["shortest_word_host"] = min((len(w) for w in words_host), default=0)
    features["shortest_word_path"] = min((len(w) for w in words_path), default=0)
    features["longest_words_raw"]  = max((len(w) for w in words_raw), default=0)
    features["longest_word_host"]  = max((len(w) for w in words_host), default=0)
    features["longest_word_path"]  = max((len(w) for w in words_path), default=0)
    features["avg_words_raw"]      = (
        sum(len(w) for w in words_raw) / len(words_raw) if words_raw else 0
    )
    features["avg_word_host"] = (
        sum(len(w) for w in words_host) / len(words_host) if words_host else 0
    )
    features["avg_word_path"] = (
        sum(len(w) for w in words_path) / len(words_path) if words_path else 0
    )

    # ── Phishing hint keywords in URL ──
    features["phish_hints"] = sum(
        1 for hint in PHISH_HINTS if hint in full_url.lower()
    )

    # ── Brand impersonation checks ──
    features["domain_in_brand"] = 1 if domain.lower() in ALLBRANDS else 0
    features["brand_in_subdomain"] = int(any(
        b in subdomain.lower() for b in ALLBRANDS if b
    ))
    features["brand_in_path"] = int(any(
        ('.' + b + '.' in path and b not in domain) for b in ALLBRANDS if b
    ))

    # ── Suspicious TLD ──
    features["suspecious_tld"] = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

    # ── Statistical report (known bad domains/IPs) ──
    bad_domains = re.compile(
        r'at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|'
        r'sweddy\.com|myjino\.ru|96\.lt|ow\.ly'
    )
    features["statistical_report"] = 1 if bad_domains.search(full_url) else 0

    return features


# ═════════════════════════════════════════════
#  LAYER 2 — PAGE CONTENT FEATURES
#  Input: URL → fetches page → parses HTML
#  Output: dict of 24 numeric features + fetch status
# ═════════════════════════════════════════════

def fetch_page(url: str, timeout: int = 8):
    """
    Attempt to fetch the page HTML.
    Returns (html_content, final_url) or (None, None) if unreachable.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code == 200 and response.content:
            return response.content, response.url
        return None, None
    except Exception:
        # Try with www prefix if direct fetch fails
        try:
            parsed = urlparse(url)
            if not parsed.netloc.startswith('www'):
                alt_url = f"{parsed.scheme}://www.{parsed.netloc}{parsed.path}"
                response = requests.get(alt_url, headers=headers, timeout=timeout)
                if response.status_code == 200:
                    return response.content, response.url
        except Exception:
            pass
        return None, None


def extract_layer2_features(url: str, html_content: bytes) -> dict:
    """
    Extract page content features from fetched HTML.
    Called only when the page is reachable.
    """
    features = {}

    parsed   = urlparse(url)
    hostname = parsed.netloc or ""
    extracted = tldextract.extract(url)
    domain   = extracted.domain + '.' + extracted.suffix

    soup = BeautifulSoup(html_content, 'html.parser', from_encoding='iso-8859-1')
    content_str = str(html_content)

    # ── Collect all link types ──
    internals, externals, null_links = [], [], []
    safe_anchors, unsafe_anchors     = [], []
    media_int, media_ext             = [], []
    form_int, form_ext, form_null    = [], [], []
    css_int, css_ext                 = [], []
    favicon_int, favicon_ext         = [], []
    link_int, link_ext               = [], []
    iframes_visible, iframes_hidden  = [], []

    NULL_HREFS = {"", "#", "#nothing", "#doesnotexist", "#null",
                  "#void", "javascript::void(0)", "javascript::void(0);",
                  "javascript::;", "javascript"}

    def is_internal(href):
        return (hostname in href or domain in href
                or not href.startswith('http'))

    # Anchor tags
    for tag in soup.find_all('a', href=True):
        href = tag['href']
        if href in NULL_HREFS:
            null_links.append(href)
        elif is_internal(href):
            internals.append(href)
            if '#' in href or 'javascript' in href.lower() or 'mailto' in href.lower():
                unsafe_anchors.append(href)
            else:
                safe_anchors.append(href)
        else:
            externals.append(href)
            safe_anchors.append(href)

    # Media tags (img, audio, embed, iframe)
    for tag in soup.find_all(['img', 'audio', 'embed'], src=True):
        src = tag['src']
        (media_int if is_internal(src) else media_ext).append(src)

    for tag in soup.find_all('iframe', src=True):
        src   = tag['src']
        style = tag.get('style', '')
        width = tag.get('width', '1')
        height= tag.get('height', '1')
        if ('display:none' in style.replace(' ', '').lower()
                or width == '0' or height == '0'):
            iframes_hidden.append(src)
        else:
            iframes_visible.append(src)

    # Link tags (CSS/favicon)
    for tag in soup.find_all('link', href=True):
        href = tag['href']
        rel  = ' '.join(tag.get('rel', []))
        if 'icon' in rel.lower():
            (favicon_int if is_internal(href) else favicon_ext).append(href)
        elif 'stylesheet' in rel.lower():
            (css_int if is_internal(href) else css_ext).append(href)
        (link_int if is_internal(href) else link_ext).append(href)

    # Form tags
    for tag in soup.find_all('form', action=True):
        action = tag['action']
        if action in NULL_HREFS:
            form_null.append(action)
        elif is_internal(action):
            form_int.append(action)
        else:
            form_ext.append(action)

    # ── Compute ratios ──
    total_links = len(internals) + len(externals) + len(null_links)
    total_media = len(media_int) + len(media_ext)
    total_anchor = len(safe_anchors) + len(unsafe_anchors)
    all_int = len(internals) + len(link_int) + len(media_int) + len(form_int) + len(css_int) + len(favicon_int)
    all_ext = len(externals) + len(link_ext) + len(media_ext) + len(form_ext) + len(css_ext) + len(favicon_ext)
    total_hyper = all_int + all_ext

    def safe_ratio(num, denom):
        return num / denom if denom > 0 else 0

    features["nb_hyperlinks"]        = total_hyper
    features["ratio_intHyperlinks"]  = safe_ratio(all_int, total_hyper)
    features["ratio_extHyperlinks"]  = safe_ratio(all_ext, total_hyper)
    features["ratio_nullHyperlinks"] = safe_ratio(len(null_links), max(total_links, 1))
    features["nb_extCSS"]            = len(css_ext)

    # Redirection ratios (approximated from link counts)
    features["ratio_intRedirection"] = safe_ratio(len(internals), max(all_int, 1))
    features["ratio_extRedirection"] = safe_ratio(len(externals), max(all_ext, 1))
    features["ratio_intErrors"]      = 0  # Cannot check live at training parity
    features["ratio_extErrors"]      = 0

    # ── Form-based features ──
    features["login_form"] = 1 if (form_ext or form_null) else 0
    features["sfh"]        = 1 if form_null else 0

    submit_to_email = any(
        'mailto:' in f or 'mail()' in f
        for f in form_int + form_ext
    )
    features["submit_email"] = 1 if submit_to_email else 0

    # ── Favicon ──
    features["external_favicon"] = 1 if favicon_ext else 0

    # ── Link tag ratio ──
    total_link_tags = len(link_int) + len(link_ext)
    features["links_in_tags"] = safe_ratio(len(link_int), total_link_tags)

    # ── Media ratios ──
    features["ratio_intMedia"] = safe_ratio(len(media_int), total_media) * 100
    features["ratio_extMedia"] = safe_ratio(len(media_ext), total_media) * 100

    # ── IFrame ──
    features["iframe"] = 1 if iframes_hidden else 0

    # ── Anchor safety ──
    features["safe_anchor"] = safe_ratio(len(unsafe_anchors), max(total_anchor, 1)) * 100

    # ── JavaScript tricks ──
    features["popup_window"]  = 1 if 'prompt(' in content_str.lower() else 0
    features["onmouseover"]   = 1 if 'onmouseover="window.status=' in content_str.lower().replace(' ', '') else 0
    features["right_clic"]    = 1 if re.search(r'event\.button\s*==\s*2', content_str) else 0

    # ── Title checks ──
    title_tag = soup.find('title')
    title_text = title_tag.get_text() if title_tag else ''
    features["empty_title"]        = 1 if not title_text.strip() else 0
    features["domain_in_title"]    = 0 if domain.lower() in title_text.lower() else 1

    # ── Copyright domain mismatch ──
    try:
        m = re.search(r'(©|™|®)', content_str)
        if m:
            ctx = content_str[max(0, m.start()-50): m.start()+50]
            features["domain_with_copyright"] = 0 if domain.lower() in ctx.lower() else 1
        else:
            features["domain_with_copyright"] = 0
    except Exception:
        features["domain_with_copyright"] = 0

    return features


def layer2_fallback_features() -> dict:
    """
    Returns default feature values when the page is unreachable.
    Uses -1 to signal 'unknown' — the model was trained with -1
    for missing values so this is consistent.
    """
    return {
        "nb_hyperlinks": -1, "ratio_intHyperlinks": -1,
        "ratio_extHyperlinks": -1, "ratio_nullHyperlinks": -1,
        "nb_extCSS": -1, "ratio_intRedirection": -1,
        "ratio_extRedirection": -1, "ratio_intErrors": -1,
        "ratio_extErrors": -1, "login_form": -1,
        "external_favicon": -1, "links_in_tags": -1,
        "submit_email": -1, "ratio_intMedia": -1,
        "ratio_extMedia": -1, "sfh": -1, "iframe": -1,
        "popup_window": -1, "safe_anchor": -1,
        "onmouseover": -1, "right_clic": -1,
        "empty_title": -1, "domain_in_title": -1,
        "domain_with_copyright": -1,
    }


# ═════════════════════════════════════════════
#  COMBINED FEATURE EXTRACTION
#  This is what the API calls for every request
# ═════════════════════════════════════════════

LAYER_1_COLS = [
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

LAYER_2_COLS = [
    "nb_hyperlinks", "ratio_intHyperlinks", "ratio_extHyperlinks",
    "ratio_nullHyperlinks", "nb_extCSS", "ratio_intRedirection",
    "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors",
    "login_form", "external_favicon", "links_in_tags", "submit_email",
    "ratio_intMedia", "ratio_extMedia", "sfh", "iframe", "popup_window",
    "safe_anchor", "onmouseover", "right_clic", "empty_title",
    "domain_in_title", "domain_with_copyright",
]

ALL_FEATURE_COLS = LAYER_1_COLS + LAYER_2_COLS


def extract_all_features(url: str) -> tuple[list, str, dict]:
    """
    Main entry point for the API.

    Returns:
        feature_vector  : list of 80 numeric values (model input)
        features_used   : "url+content" | "url_only"
        feature_details : raw dict of all feature names + values
    """
    # Always extract Layer 1
    l1 = extract_layer1_features(url)

    # Attempt Layer 2
    html_content, _ = fetch_page(url)

    if html_content:
        try:
            l2 = extract_layer2_features(url, html_content)
            features_used = "url+content"
        except Exception:
            l2 = layer2_fallback_features()
            features_used = "url_only"
    else:
        l2 = layer2_fallback_features()
        features_used = "url_only"

    # Merge and order features to match training column order
    all_feats = {**l1, **l2}
    feature_vector = [all_feats.get(col, -1) for col in ALL_FEATURE_COLS]

    return feature_vector, features_used, all_feats
