import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify, session # type: ignore
from flask_session import Session # type: ignore
from flask_cors import CORS # type: ignore
from flask_bcrypt import Bcrypt # type: ignore
from pymongo import MongoClient # type: ignore
from pymongo.errors import ConnectionFailure, ConfigurationError
from bson import ObjectId # type: ignore
import json
import difflib
import random
import base64
import tempfile 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from datetime import datetime
from dotenv import load_dotenv # type: ignore
from faq_retrieval import traiter_question
from chatbot_story import generer_storytelling , normalize_story, intro_narrative_programme, anecdote_aleatoire , format_events
from audio_handler import text_to_speech
from model_intents import predict_intent
from nlp_engine import traiter_question_utilisateur
from questions import get_bot_response  
from logic.programmes import (
   programmes_foire_2023, programmes_enfant_2023, get_programme_28_avril, get_programme_07_mai,
   get_programme_duration_global, get_programme_by_date_global, get_programme_enfant_general_global,
   get_all_programme_combined_dates_global, get_foire_end_date_global, get_foire_start_date_global,
   get_programme_date_range, get_event_locations_global, get_event_hours_global, get_event_price_global,    get_editors_count_global,get_event_locations_detailed, get_event_hours_detailed, get_programme_enfant_general_detailed, 
   get_editors_count_detailed,  get_event_duration_detailed, get_event_hours_detailed, get_event_locations_detailed, get_programme_by_date_detailed,
   get_event_price_detailed, get_programme_date_range_detailed ,get_all_programmes_detailed,  get_editors_countries_of_origin,
) 
import traceback


# ────────────────────────────────────────────── # ⚙️ Config # ───────────────────────────────────────── #

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
app = Flask(__name__) 
app.secret_key = SECRET_KEY or "fallback_secret" 
app.config["SESSION_TYPE"] = "filesystem"
Session(app)



# Enable CORS with credentials (important for front-end sessions)
CORS(app, supports_credentials=True)

bcrypt = Bcrypt(app)
try:
    if MONGO_URI:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # Timeout de 5s
        # 🔎 Test rapide de la connexion
        client.admin.command("ping")
        db = client["chatbotEvent"]
        programmes = db["programmes_foire_2023"]
        programmes_enfant = db["programmes_enfant_2023"]
        editeurs = db["editeurs"]
        users_collection = db["comptes"]
        childtracking = db["child_tracking"]
        infos_generales = db["infos_generales"]
        print("✅ Connexion MongoDB réussie.")
    else:
        raise ValueError("❌ MONGO_URI non défini dans les variables d'environnement.")
except (ConnectionFailure, ConfigurationError) as e:
    print(f"⚠️ Erreur de connexion MongoDB : {e}")
    client = None
    db = None
    programmes = None
    programmes_enfant = None
    users_collection = None
    editeurs = None
    childtracking = None 
    infos_generales = None
# Helper to serialize MongoDB ObjectId
def serialize_user(user):
    return {
        "id": str(user.get("_id")),
        "username": user.get("username"),
    }

def _markdown_from_answer(ans: dict, mode: str = "brief") -> str:
    """Transforme une réponse {summary, details} en chaîne Markdown lisible."""
    if not isinstance(ans, dict):
        return str(ans)

    summary = (ans.get("summary") or "").strip()
    details = ans.get("details")

    out = []
    out.append(f"- **Résumé**\n  {summary if summary else '—'}")

    if isinstance(details, list):
        out.append("- **Détails**")
        for item in details:
            if isinstance(item, dict):
                titre = item.get("titre") or item.get("nom") or item.get("date") or "Événement"
                bloc = f"- **{titre}**"
                for k, v in item.items():
                    if k not in {"titre", "nom"} and v:
                        bloc += f"\n  - {k.capitalize()} : {v}"
                out.append(bloc)
            else:
                out.append(f"- {item}")
    elif isinstance(details, dict):
        out.append("- **Détails**")
        for k, v in details.items():
            if isinstance(v, (dict, list)):
                out.append(f"- **{k.capitalize()}** : {str(v)}")
            else:
                out.append(f"- **{k.capitalize()}** : {v}")
    else:
        det = (details or "").strip()
        if det:
            lines = [l.strip() for l in det.split("\n") if l.strip()]
            if len(lines) > 1:
                out.append("- **Détails**")
                out.extend([f"- {l}" for l in lines])
            else:
                out.append(f"- **Détails**\n  {det}")
        else:
            out.append("- **Détails**\n  —")

    return "\n".join(out)


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    mot_de_passe = data.get("mot_de_passe")

    if not username or not mot_de_passe:
        return jsonify({"error": "Champs manquants"}), 400

    if users_collection.find_one({"username": username}):
        return jsonify({"error": "Utilisateur existe déjà"}), 400

    hashed_pw = bcrypt.generate_password_hash(mot_de_passe).decode("utf-8")
    users_collection.insert_one({
        "username": username,
        "mot_de_passe": hashed_pw
    })

    return jsonify({"message": "Inscription réussie"}), 201

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    mot_de_passe = (data.get("mot_de_passe") or "").strip()

    print("🔹 Tentative de connexion pour username:", username)
    user = users_collection.find_one({"username": username})
    if not user:
        print("❌ Utilisateur non trouvé")
        return jsonify({"error": "Identifiants invalides"}), 401

    print("🔹 Hash stocké:", user.get("mot_de_passe"))
    if not bcrypt.check_password_hash(user["mot_de_passe"], mot_de_passe):
        print("❌ Mot de passe incorrect")
        return jsonify({"error": "Identifiants invalides"}), 401

    session["user_id"] = str(user["_id"])
    session["authenticated"] = True
    session["username"] = username

    print("✅ Connexion réussie pour", username)
    return jsonify({
        "message": "Connexion réussie",
        "user": serialize_user(user),
        "type_compte": user.get("type_compte", "normal")
    })


@app.route("/api/is-auth", methods=["GET"])
def is_auth():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"authenticated": False}), 200
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({"authenticated": False}), 200
        return jsonify({"authenticated": True, "user": serialize_user(user)}), 200
    except Exception as e:
        app.logger.error(f"is-auth erreur: {e}")
        return jsonify({"authenticated": False}), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)
    return jsonify({"message": "Déconnexion réussie"})

@app.route("/api/edit_account", methods=["PUT"])
def edit_account():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Non authentifié"}), 401

    data = request.get_json()
    new_username = data.get("username")
    new_password = data.get("mot_de_passe")

    update_data = {}
    if new_username:
        update_data["username"] = new_username
    if new_password:
        update_data["mot_de_passe"] = bcrypt.generate_password_hash(new_password).decode("utf-8")

    if update_data:
        users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
        return jsonify({"message": "Compte mis à jour"})

    return jsonify({"error": "Aucune donnée fournie"}), 400

# ---------------------------------------------------------
# Delete account (uniformisé bcrypt )
# ---------------------------------------------------------
@app.route("/api/delete_account", methods=["DELETE"])
def delete_account():
    if not session.get("authenticated"):
        return jsonify({"message": "Non autorisé."}), 403

    username = session.get("username")
    if not username:
        return jsonify({"message": "Session invalide."}), 400

    result = users_collection.delete_one({"username": username})
    session.clear()

    if result.deleted_count == 0:
        return jsonify({"message": "Utilisateur non trouvé."}), 404

    return jsonify({"message": "Compte supprimé avec succès."}), 200

# ---------------------------------------------------------
# Helper (normalise le mode demandé)
# ---------------------------------------------------------
def _normalize_mode(v: str) -> str:
    v = (v or "").strip().lower()
    detail = {"detaille", "detaillé", "détaillé", "detailed", "long", "detail"}
    if v in detail:
        return "detaille"
    return "breve"



# ---------------------------------------------------------
# Endpoint /api/ask
# - supporte input 'text' ou 'audio' (audio : base64)
# - supporte mode 'breve' ou 'detaille'
# NOTE : transcription audio n'est pas implémentée → stub (sauvegarde temporaire)
# ---------------------------------------------------------



# ──────────────
# Route /api/ask
# ──────────────

@app.route("/api/ask", methods=["POST"])
def ask():
    """
    ✅ Route /api/ask — version finale et stable.
    - Compatible avec : model_intents, chatbot_story, audio_handler
    - Supporte : texte + audio (base64)
    - Supprime complètement les réponses d’incertitude.
    - Storytelling étendu (programmes + éditeurs/pays)
    """
    try:
        data = request.get_json(force=True)

        # --- 1️⃣ Extraction du message (texte ou audio) ---
        user_message = (data.get("message") or data.get("question") or "").strip()
        b64_audio = data.get("audio")

        # Si un audio est envoyé → le transcrire avant tout
        if not user_message and b64_audio:
            from audio_handler import speech_to_text_from_base64
            user_message = speech_to_text_from_base64(b64_audio, language="fr") or ""

        if not user_message:
            return jsonify({"answer": "❌ Veuillez entrer ou prononcer une question."}), 200

        print(f"\n[ASK] 🔎 Message reçu : {user_message}")

        # --- 2️⃣ Étape : Prédiction de l’intention ---
        try:
            intent_data = predict_intent(user_message)
            intent = intent_data.get("intent", "unknown")
            confidence = float(intent_data.get("confidence", 0.0))
            print(f"[ASK] 🎯 Intent détecté : {intent} (confiance = {confidence:.2f})")
        except Exception as e:
            print(f"[ERREUR INTENT MODEL] {e}")
            traceback.print_exc()
            intent = "unknown"
            confidence = 0.0

        # --- 3️⃣ Mapping des fonctions logiques ---
        mapping = {
            "programmes_foire_2023": programmes_foire_2023,
            "programmes_enfant_2023": programmes_enfant_2023,
            "get_programme_28_avril": get_programme_28_avril,
            "programme_28_avril": get_programme_28_avril,
            "get_programme_07_mai": get_programme_07_mai,
            "programme_07_mai": get_programme_07_mai,
            "get_programme_duration_global": get_programme_duration_global,
            "get_programme_by_date_global": get_programme_by_date_global,
            "get_programme_enfant_general_global": get_programme_enfant_general_global,
            "get_all_programme_combined_dates_global": get_all_programme_combined_dates_global,
            "get_foire_end_date_global": get_foire_end_date_global,
            "get_foire_start_date_global": get_foire_start_date_global,
            "get_programme_date_range": get_programme_date_range,
            "get_event_locations_global": get_event_locations_global,
            "get_event_hours_global": get_event_hours_global,
            "get_event_price_global": get_event_price_global,
            "get_editors_count_global": get_editors_count_global,
        
            "get_event_locations_detailed": get_event_locations_detailed,
            "get_event_hours_detailed": get_event_hours_detailed,
            "get_programme_enfant_general_detailed": get_programme_enfant_general_detailed,
            "get_editors_count_detailed": get_editors_count_detailed,
            "get_event_duration_detailed": get_event_duration_detailed,
            "get_programme_by_date_detailed": get_programme_by_date_detailed,
            "get_event_price_detailed": get_event_price_detailed,
            "get_programme_date_range_detailed": get_programme_date_range_detailed,
            "get_all_programmes_detailed": get_all_programmes_detailed,
            "get_editors_countries_of_origin": get_editors_countries_of_origin,
        }


        # --- 4️⃣ Si l’intent correspond à une action logique ---
        if intent in mapping:
            try:
                func = mapping[intent]

                # 🔍 Déterminer si la fonction accepte un argument
                if func.__code__.co_argcount == 0:
                    print(f"[ASK] 🚀 Appel de {func.__name__}() sans argument")
                    result = func()
                else:
                    print(f"[ASK] 🚀 Appel de {func.__name__}('{user_message}')")
                    result = func(user_message)

                # --- 🧩 Storytelling contextuel ---
                if "programme" in intent:
                    print("[ASK] 📖 Génération du storytelling (programme)...")
                    from chatbot_story import generer_storytelling
                    story = generer_storytelling(result, question=user_message)
                    final_answer = story.get("details", "Aucune donnée détaillée.")

                elif "editeur" in intent or "editor" in intent:
                    print("[ASK] 📖 Génération du storytelling (éditeurs/pays)...")
                    from chatbot_story import format_editors_countries, intro_narrative_editors_countries, anecdote_aleatoire
                    formatted = format_editors_countries(result)
                    countries_list = formatted.get("details", [])
                    
                    # Nombre total d’éditeurs (valeurs de ta base)
                    n_editors_countries = 129 + 74 + 17 + 11 + 7 + 4 + 3 + 3 + 3 + 3 + 2 + 2 + 2 + 2 + 2 + 3 + 2 + 1 + 1
                    
                    intro = intro_narrative_editors_countries(n_editors_countries, user_message)
                    countries_text = ", ".join(countries_list)
                    conclusion = anecdote_aleatoire("editeurs")
                    final_answer = f"{intro}\n\n🌍 Pays représentés : {countries_text}.\n\n{conclusion}"

                else:
                    final_answer = result

                # --- 🎧 Conversion en audio (uniquement si mode audio) ---
                from audio_handler import text_to_speech

                input_mode = (data.get("mode") or data.get("input_mode") or "").strip().lower()
                if not input_mode:
                    input_mode = "audio" if b64_audio else "text"

                b64_tts = None
                if input_mode == "audio":
                    b64_tts = text_to_speech(final_answer, language="fr")
                    print("[ASK] 🔊 TTS généré pour mode audio.")

                return jsonify({
                    "answer": final_answer,
                    "audio": b64_tts
                }), 200

            except Exception as e:
                print(f"[ERREUR LOGIQUE] {e}")
                return jsonify({
                    "answer": f"⚠️ Erreur lors du traitement logique : {e}"
                }), 200

        # --- 5️⃣ Si aucune correspondance d’intent trouvée ---
        if intent == "unknown" or confidence < 0.3:
            return jsonify({
                "answer": "🤔 Désolé, je n’ai trouvé aucune donnée correspondant précisément à votre demande."
            }), 200

        # --- 6️⃣ Lecture fallback depuis intents_data.json ---
        try:
            intents_path = os.path.join(os.path.dirname(__file__), "intents_data.json")
            with open(intents_path, "r", encoding="utf-8") as f:
                intents = json.load(f)
            if intent in intents:
                responses = intents[intent].get("responses", [])
                if responses:
                    answer = random.choice(responses)
                    from audio_handler import text_to_speech
                    b64_tts = text_to_speech(answer, language="fr")
                    return jsonify({"answer": answer, "audio": b64_tts}), 200
        except Exception as e:
            print(f"[ERREUR LECTURE INTENTS] {e}")

        # --- 7️⃣ Dernier recours ---
        neutral_msg = "Je n’ai pas trouvé d’information exacte à ce sujet."
        from audio_handler import text_to_speech
        b64_tts = text_to_speech(neutral_msg, language="fr")
        return jsonify({"answer": neutral_msg, "audio": b64_tts}), 200

    except Exception as e:
        print(f"[ERREUR GLOBALE /api/ask] {e}")
        return jsonify({"answer": f"Erreur interne : {str(e)}"}), 500





# ➕ Ajouter un enfant
@app.route("/api/child/add", methods=["POST"])
def add_child():
    data = request.get_json()
    # utilise la collection childtracking existante
    childtracking.insert_one(data)
    return jsonify({"message": "Enfant ajouté avec succès."}), 201

@app.route("/api/child/all", methods=["GET"])
def get_all_children():
    children = list(childtracking.find())
    for c in children:
        c["_id"] = str(c["_id"])
    return jsonify(children), 200
    return jsonify(children), 200

# ✏️ Mettre à jour un enfant
@app.route("/api/child/update/<string:child_id>", methods=["PUT"])
def update_child(child_id):
    data = request.get_json()
    result = childtracking.update_one({"_id": ObjectId(child_id)}, {"$set": data})
    return jsonify({"updated": result.modified_count}), 200

# ❌ Supprimer un enfant
@app.route("/api/child/delete/<string:child_id>", methods=["DELETE"])
def delete_child(child_id):
    result = childtracking.delete_one({"_id": ObjectId(child_id)})
    return jsonify({"deleted": result.deleted_count}), 200

# 📍 Récupérer la position d’un enfant
@app.route("/api/child/<child_id>/location", methods=["GET"])
def get_child_location(child_id):
    child = childtracking.find_one({"_id": ObjectId(child_id)})
    if child and "latitude" in child and "longitude" in child:
        return jsonify({"latitude": child["latitude"], "longitude": child["longitude"]}), 200
    return jsonify({"error": "Localisation introuvable"}), 404

# 📍 Mettre à jour la position d’un enfant
@app.route("/api/child/<child_id>/position", methods=["PUT"])
def update_child_position(child_id):
    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        return jsonify({"error": "Latitude et longitude sont obligatoires."}), 400

    result = childtracking.update_one(
        {"_id": ObjectId(child_id)},
        {"$set": {"latitude": latitude, "longitude": longitude}}
    )

    if result.modified_count == 1:
        return jsonify({"message": "Position mise à jour avec succès."}), 200
    return jsonify({"message": "Aucune modification effectuée."}), 404



# ---------------------------------------------------------
# Route principale du chatbot qui utilise la base de connaissances
# get_bot_response (questions.py)
# ---------------------------------------------------------
@app.route("/chatbot", methods=["POST"])
def chatbot_general():
    if not session.get("authenticated"):
        return jsonify({"response": "❌ Connectez-vous."}), 403

    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"response": "❌ Aucune question reçue."}), 400

    mode_input = (data.get("response_type") or data.get("mode") or "brief").strip().lower()
    map_mode = {
        "breve": "brief", "brève": "brief", "brief": "brief", "short": "brief",
        "detaille": "detailed", "détaillé": "detailed", "detailed": "detailed", "long": "detailed"
    }
    mode = map_mode.get(mode_input, "brief")

    # 1️⃣ FAQ direct
    faq_answer = traiter_question(question, response_type=mode)
    if faq_answer and not faq_answer.startswith("⚠️") and not faq_answer.startswith("❌"):
        return jsonify({"response": faq_answer}), 200

    # 2️⃣ Fallback bot principal (intention + storytelling)
    bot_result = get_bot_response(question, response_type=mode)
    answer_dict = bot_result.get("answer") if isinstance(bot_result, dict) else {
        "summary": str(bot_result), "details": str(bot_result)
    }

    md = _markdown_from_answer(answer_dict, mode=mode)

    return jsonify({
        "response": md,
        "answer": answer_dict,
        "intent": bot_result.get("intent")
    }), 200


@app.route("/api/programme/duration", methods=["GET"])
def programme_duration():
    return jsonify({"result": get_programme_duration_global()})

@app.route("/api/programme/date/<date_str>", methods=["GET"])
def programme_by_date(date_str):
    return jsonify(get_programme_by_date_detailed(date_str))

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


@app.route("/ping")
def ping():
    return "pong", 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5432))
    app.run(host="0.0.0.0", port=port)


