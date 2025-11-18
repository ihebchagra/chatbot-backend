"""
model_intents.py — version stable finale
Multilingue, sans warning TensorFlow, cohérente avec Flask + audio_handler.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

# ------------------------
# Réduction des logs TensorFlow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

DEFAULT_MODEL_DIR = Path("./Code/intents_model")
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_ENCODER_CACHE: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------
def _get_encoder(model_name: str = _DEFAULT_ENCODER_NAME) -> SentenceTransformer:
    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        _ENCODER_CACHE = SentenceTransformer(model_name)
    return _ENCODER_CACHE


def clean_text(s: str) -> str:
    """Nettoyage et normalisation des textes pour le modèle."""
    if not isinstance(s, str):
        s = str(s or "")
    s = s.lower()
    s = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s]", " ", s)

    replacements = {
        "sessions": "programme",
        "planning": "programme",
        "événements": "programme",
        "événement": "programme",
        "agenda": "programme",
        "plan": "programme",
        "activité": "programme",
        "activités": "programme",
        "seances": "programme",
        "event": "événement",
        "journee": "jour",
        "journée": "jour",
        "journées": "jours",
        "date": "jour",
        "details": "détail",
        "détails": "détail",
        "participants": "visiteurs",
        "intervenants": "orateurs",
        "conférenciers": "orateurs",
        "stands": "exposants",
        "expositions": "exposants",
        "localisation": "pays",
        "emplacement": "stands",
        "origine": "pays",
        "fermeture": "fin",
        "clôture": "fin",
        "ouverture": "début",
        "lancement": "début",
        "inauguration": "début",
    }

    for k, v in replacements.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------
def train_and_save(json_path: str | Path,
                   model_dir: str | Path = DEFAULT_MODEL_DIR,
                   model_name: str = _DEFAULT_ENCODER_NAME):
    """Entraînement et sauvegarde du modèle d’intentions."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Le JSON doit contenir une liste d’intentions/patterns.")

    texts, intents = [], []
    for item in data:
        intent = item.get("intent", "").strip()
        for p in item.get("patterns", []):
            if p and p.strip():
                texts.append(clean_text(p))
                intents.append(intent)

    print(f"📚 {len(texts)} exemples — {len(set(intents))} intentions.")

    encoder = SentenceTransformer(model_name)
    X = encoder.encode(texts, convert_to_numpy=True)

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(intents)

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X, y)

    acc = accuracy_score(y, clf.predict(X))
    print(f"✅ Entraînement terminé — accuracy = {acc:.3f}")

    joblib.dump(clf, model_dir / "classifier.joblib")
    joblib.dump(label_enc, model_dir / "label_encoder.joblib")
    (model_dir / "model_name.txt").write_text(model_name, encoding="utf-8")
    print(f"💾 Modèle sauvegardé dans {model_dir.resolve()}")

    return {"accuracy": acc, "samples": len(texts)}


# ---------------------------------------------------------------------
def predict_intent(text: str, model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict:
    """Prédit l’intention principale d’un texte utilisateur."""
    text = (text or "").strip()
    if not text:
        return {"intent": "unknown", "confidence": 0.0, "mode": "unknown", "answer": "Texte vide."}

    model_dir = Path(model_dir)
    if not (model_dir / "classifier.joblib").exists():
        print(model_dir);
        raise FileNotFoundError("⚠️ Classifier non trouvé. Lancez d’abord l’entraînement.")

    model_name = (model_dir / "model_name.txt").read_text(encoding="utf-8").strip()
    encoder = _get_encoder(model_name)
    clf = joblib.load(model_dir / "classifier.joblib")
    label_enc = joblib.load(model_dir / "label_encoder.joblib")

    X = encoder.encode([clean_text(text)], convert_to_numpy=True)
    probs = clf.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    intent = label_enc.inverse_transform([idx])[0]
    conf = float(probs[idx])

    # Détection du mode
    try:
        from mode_detector import detect_mode
        mode = detect_mode(text)
    except Exception:
        mode = "breve" if len(text.split()) < 7 else "detaille"

    # Renforcement si indices clairs
    low = text.lower()
    if any(k in low for k in ["7 mai", "dernier jour", "clôture"]):
        intent, conf = "get_programme_07_mai", max(conf, 0.8)
    elif any(k in low for k in ["28 avril", "premier jour", "ouverture"]):
        intent, conf = "get_programme_28_avril", max(conf, 0.8)

    if conf < 0.25:
        return {"intent": "unknown", "confidence": conf, "mode": mode, "answer": "Pouvez-vous reformuler votre question ?"}

    return {"intent": intent, "confidence": conf, "mode": mode, "answer": None}
