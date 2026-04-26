# sentiment_analysis/preprocess.py — text cleaning & feature engineering

import re
import html
import string
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (safe to call multiple times)
for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

# Keep negation words — critical for sentiment accuracy
STOPS = set(stopwords.words("english")) - {
    "not", "no", "never", "nor", "neither", "hardly", "barely", "scarcely",
}

lemmatizer = WordNetLemmatizer()

# Informal contractions → standard form
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am",
}


def expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def clean(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. HTML unescape
      2. URL / mention / hashtag removal
      3. Contraction expansion
      4. Punctuation normalisation (keep ! and ?)
      5. Lowercasing
      6. Stop-word removal (negations preserved)
      7. Lemmatisation
    """
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)   # URLs
    text = re.sub(r"@\w+", "", text)                     # @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # #hashtag → word
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)           # loooong → loo
    text = expand_contractions(text)
    text = re.sub(r"[^a-zA-Z0-9 !?']", " ", text)
    text = text.lower().strip()
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOPS and len(w) > 1
    ]
    return " ".join(tokens)


def detect_sarcasm_features(text: str) -> Dict[str, float]:
    """
    Lightweight lexical sarcasm signals used as auxiliary features.
    Returns a dict of binary / ratio features.
    """
    positive_words = r"\b(great|amazing|wonderful|fantastic|brilliant|perfect)\b"
    negative_ctx   = r"\b(not|never|worst|terrible|awful|horrible)\b"
    return {
        "pos_word_neg_ctx": float(
            bool(re.search(positive_words, text, re.I))
            and bool(re.search(negative_ctx, text, re.I))
        ),
        "has_ellipsis":     float("..." in text),
        "all_caps_ratio":   sum(1 for c in text if c.isupper()) / (len(text) + 1),
        "exclamation_count": float(text.count("!")),
        "question_count":    float(text.count("?")),
    }


def batch_clean(texts: List[str]) -> List[str]:
    """Vectorised cleaning over a list of strings."""
    return [clean(t) for t in texts]
