import os
import re
import uuid
import base64
import tempfile
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from helper import get_embeddings
from prompt import system_prompt, nutrition_prompt_template

load_dotenv()

# ── Tesseract binary path (needed on some systems / conda envs) ──────────────
import pytesseract
tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
SECRET_KEY       = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
DB_PATH = os.getenv("SQLITE_DB_PATH", "chat_history.db")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("PINECONE_API_KEY or GROQ_API_KEY missing in .env!")

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

# ── SQLite — zero install, built into Python ─────────────────────────────────
import sqlite3, json

def get_db():
    """Return a new SQLite connection (thread-safe: one per request)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    """Create tables on first run."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                last_disease  TEXT DEFAULT '',
                last_severity TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                ts         TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON sessions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sess ON messages(session_id)")
        # Migrate existing DBs that don't have the new columns yet
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN last_disease  TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN last_severity TEXT DEFAULT ''")
        except Exception:
            pass

_init_db()
print("✅ SQLite ready →", DB_PATH)

# ── Vector Stores — two Pinecone namespaces ──────────────────────────────────
def _init_vector_stores():
    embeddings = get_embeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Index '{INDEX_NAME}' not found. Run store_index.py first.")
    general = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings, namespace="general")
    nutrition = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings, namespace="nutrition")
    return general, nutrition

print("🔧 Loading vector stores…")
vs_general, vs_nutrition = _init_vector_stores()
retriever_general   = vs_general.as_retriever(search_kwargs={"k": 4})
retriever_nutrition = vs_nutrition.as_retriever(search_kwargs={"k": 5})
print("✅ Vector stores ready")

# ── LLM ──────────────────────────────────────────────────────────────────────
chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=2,
)

# ── Chain A: Disease identification ──────────────────────────────────────────
disease_chain = (
    {
        "context": itemgetter("input") | retriever_general,
        "input":   itemgetter("input"),
    }
    | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    | chat_model
    | StrOutputParser()
)

# ── Chain B: Nutrition recommendation ────────────────────────────────────────
nutrition_chain = (
    {
        "context": itemgetter("disease") | retriever_nutrition,
        "disease": itemgetter("disease"),
    }
    | ChatPromptTemplate.from_messages([
        ("system", nutrition_prompt_template),
        ("human", "Disease: {disease}\n\nProvide the daily nutritional requirements.")
    ])
    | chat_model
    | StrOutputParser()
)

# ════════════════════════════════════════════════════════════════════════════
# USER / SESSION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def get_current_user() -> str:
    """
    Returns user_id from Flask session.
    In production: replace with your JWT decode → user_id.
    In dev: auto-creates an anonymous session-scoped id.
    """
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
    return session["user_id"]

def _ensure_session(user_id: str, session_id) -> str:
    """Return existing session_id if valid, otherwise create a new one."""
    with get_db() as conn:
        if session_id:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id=? AND user_id=?",
                (session_id, user_id)
            ).fetchone()
            if row:
                return session_id
        # Create new session
        new_sid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, created_at) VALUES (?,?,?)",
            (new_sid, user_id, datetime.now(timezone.utc).isoformat())
        )
    return new_sid

def _append_msgs(user_id: str, sid: str, msgs: list):
    """Insert one or more messages into the messages table."""
    ts = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.executemany(
            "INSERT INTO messages (session_id, user_id, role, content, ts) VALUES (?,?,?,?,?)",
            [(sid, user_id, m["role"], m["content"], ts) for m in msgs]
        )

def _get_msgs(user_id: str, sid: str) -> list:
    """Return all messages for a session as a list of dicts."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content, ts FROM messages WHERE session_id=? AND user_id=? ORDER BY id",
            (sid, user_id)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]

def _update_session_meta(user_id: str, sid: str, disease: str, severity: str):
    """Save the latest disease and severity onto the session row."""
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET last_disease=?, last_severity=? "
            "WHERE session_id=? AND user_id=?",
            (disease, severity, sid, user_id)
        )

def _history_ctx(msgs: list, max_turns: int = 6) -> str:
    tail = msgs[-(max_turns * 2):]
    return "\n".join(
        ("Patient" if m["role"] == "user" else "Assistant") + ": " + m["content"]
        for m in tail
    )

# ════════════════════════════════════════════════════════════════════════════
# SEVERITY DETECTION
# ════════════════════════════════════════════════════════════════════════════

# Conditions that always trigger CRITICAL alert regardless of LLM
CRITICAL_CONDITIONS = {
    # Cardiac
    "heart attack", "myocardial infarction", "cardiac arrest", "heart failure",
    "acute coronary syndrome", "ventricular fibrillation", "cardiac tamponade",
    # Neurological
    "stroke", "brain hemorrhage", "brain bleed", "intracranial hemorrhage",
    "subarachnoid hemorrhage", "transient ischemic attack", "tia",
    "status epilepticus", "meningitis", "encephalitis",
    # Respiratory
    "pulmonary embolism", "tension pneumothorax", "respiratory failure",
    "severe asthma attack", "epiglottitis",
    # Shock / Infection
    "anaphylaxis", "anaphylactic shock", "sepsis", "septic shock",
    "toxic shock syndrome",
    # Abdominal
    "appendicitis", "bowel perforation", "ruptured spleen", "aortic aneurysm",
    "aortic dissection", "ectopic pregnancy", "ruptured ectopic",
    # Endocrine
    "diabetic ketoacidosis", "dka", "hypoglycemic coma", "thyroid storm",
    "addisonian crisis", "adrenal crisis",
    # Trauma / Other
    "spinal injury", "severe burn", "drowning", "poisoning", "overdose",
    "internal bleeding", "severe hemorrhage", "hypertensive crisis",
    "eclampsia", "pre-eclampsia", "placental abruption",
}

# Symptoms in the user query that alone warrant urgent/critical attention
URGENT_SYMPTOMS = {
    # Chest
    "chest pain", "chest tightness", "chest pressure", "crushing chest",
    "radiating to arm", "radiating to jaw",
    # Breathing
    "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "breathless", "coughing blood", "blood in sputum",
    # Neuro
    "sudden severe headache", "worst headache of my life", "thunderclap headache",
    "sudden confusion", "slurred speech", "face drooping", "arm weakness",
    "sudden numbness", "sudden vision loss", "double vision",
    "loss of consciousness", "unconscious", "fainted", "fainting",
    "seizure", "convulsion", "not responding",
    # Bleeding
    "vomiting blood", "blood in stool", "rectal bleeding",
    "severe bleeding", "won't stop bleeding", "coughing up blood",
    # Allergic
    "throat swelling", "tongue swelling", "lips swelling",
    "allergic reaction", "hives all over",
    # Vitals
    "fever above 103", "fever above 104", "high fever chills",
    "bluish lips", "blue lips", "cold sweat", "clammy skin",
    "heart racing", "rapid heartbeat", "palpitations",
    # Abdomen
    "severe abdominal pain", "rigid abdomen", "cannot urinate",
    # Misc
    "overdose", "poisoned", "severe burn", "not breathing",
    "baby not moving", "labour pain",
}

def _assess_severity(query: str, disease: str, disease_info: str) -> dict:
    """
    3-step severity assessment:
    Step 1 — instant keyword match (fast, conservative — errs toward higher severity)
    Step 2 — LLM triage with strict prompt
    Step 3 — take the WORSE of the two
    """
    query_lower        = query.lower()
    disease_lower      = disease.lower()
    disease_info_lower = disease_info.lower()

    # ── Step 1: Instant keyword check ───────────────────────────────────────
    matched_critical = next(
        (c for c in CRITICAL_CONDITIONS
         if c in disease_lower or c in disease_info_lower or c in query_lower), None)

    matched_urgent = next(
        (s for s in URGENT_SYMPTOMS if s in query_lower), None)

    instant_level = None
    if matched_critical:
        instant_level = "CRITICAL"
        print(f"🚨 Instant CRITICAL match: '{matched_critical}'")
    elif matched_urgent:
        instant_level = "URGENT"
        print(f"⚠️  Instant URGENT match: '{matched_urgent}'")

    # ── Step 2: LLM severity assessment ─────────────────────────────────────
    severity_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior emergency medicine physician doing triage.\n"
         "Classify the severity of this condition as exactly ONE of these levels:\n\n"
         "CRITICAL — life-threatening, requires emergency room or ambulance IMMEDIATELY.\n"
         "           Examples: heart attack, stroke, anaphylaxis, sepsis, overdose,\n"
         "           severe bleeding, difficulty breathing, loss of consciousness.\n"
         "           When in doubt between CRITICAL and URGENT, choose CRITICAL.\n\n"
         "URGENT   — serious, needs a doctor or urgent care TODAY. Cannot wait.\n"
         "           Examples: high fever, severe pain, suspected fracture,\n"
         "           worsening infection, uncontrolled vomiting/diarrhea.\n\n"
         "MODERATE — needs a doctor visit within 2-3 days.\n"
         "           Examples: mild infection, rash, persistent cough, UTI.\n\n"
         "MILD     — can safely be managed at home with rest and OTC medication.\n"
         "           Examples: common cold, minor cut, mild headache, indigestion.\n\n"
         "IMPORTANT: Always err on the side of caution. If there is ANY doubt, "
         "classify one level HIGHER than you think. A wrong MILD when it is CRITICAL "
         "can cost a life.\n\n"
         "Respond in EXACTLY this format (no extra text):\n"
         "SEVERITY: <CRITICAL|URGENT|MODERATE|MILD>\n"
         "REASON: <one sentence clinical reason>\n"
         "ACTION: <one sentence — what should the patient do RIGHT NOW>"),
        ("human",
         "Disease identified: {disease}\n\n"
         "Clinical description:\n{disease_info}\n\n"
         "Patient's own words: {query}")
    ])

    severity_chain = severity_prompt | chat_model | StrOutputParser()

    try:
        llm_resp = severity_chain.invoke({
            "disease":      disease,
            "disease_info": disease_info[:800],
            "query":        query[:300],
        })
        sev_match    = re.search(r"SEVERITY:\s*(CRITICAL|URGENT|MODERATE|MILD)",
                                 llm_resp, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)",  llm_resp, re.IGNORECASE)
        action_match = re.search(r"ACTION:\s*(.+)",  llm_resp, re.IGNORECASE)

        llm_level  = sev_match.group(1).upper()   if sev_match    else "URGENT"
        llm_reason = reason_match.group(1).strip() if reason_match else ""
        llm_action = action_match.group(1).strip() if action_match else ""
        print(f"🤖 LLM severity: {llm_level} — {llm_reason}")

    except Exception as e:
        print(f"⚠️  Severity LLM error: {e} — defaulting to URGENT")
        llm_level  = "URGENT"   # safe default on error
        llm_reason = ""
        llm_action = ""

    # ── Step 3: Take the more severe of instant vs LLM ──────────────────────
    order = {"CRITICAL": 4, "URGENT": 3, "MODERATE": 2, "MILD": 1}
    final_level = max(
        instant_level or llm_level,
        llm_level,
        key=lambda x: order.get(x, 0)
    )
    print(f"✅ Final severity: {final_level}")

    # ── Build alert payload ──────────────────────────────────────────────────
    alert_config = {
        "CRITICAL": {
            "show":    True,
            "color":   "#ff1744",
            "icon":    "🚨",
            "title":   "EMERGENCY — Go to Hospital Immediately!",
            "message": (llm_action or
                        "This condition may be life-threatening. "
                        "Call emergency services (108 / 112) or go to the nearest "
                        "emergency room RIGHT NOW. Do not wait."),
            "call":    "108",
        },
        "URGENT": {
            "show":    True,
            "color":   "#ff6d00",
            "icon":    "⚠️",
            "title":   "See a Doctor Today — Do Not Delay",
            "message": (llm_action or
                        "Your symptoms need prompt medical attention. "
                        "Please visit a doctor or urgent care clinic today."),
            "call":    None,
        },
        "MODERATE": {
            "show":    False,
            "color":   "#ffd600",
            "icon":    "ℹ️",
            "title":   "Doctor Visit Recommended",
            "message": (llm_action or
                        "Consider scheduling a doctor appointment within the next 2-3 days."),
            "call":    None,
        },
        "MILD": {
            "show":    False,
            "color":   "#00c853",
            "icon":    "✅",
            "title":   "Manageable at Home",
            "message": (llm_action or
                        "Monitor your symptoms. See a doctor if they worsen."),
            "call":    None,
        },
    }

    cfg = alert_config.get(final_level, alert_config["URGENT"])
    hospital_url = "https://www.google.com/maps/search/hospital+near+me"

    return {
        "severity":      final_level,
        "show_alert":    cfg["show"],
        "alert_color":   cfg["color"],
        "alert_icon":    cfg["icon"],
        "alert_title":   cfg["title"],
        "alert_message": cfg["message"],
        "emergency_call":cfg["call"],
        "hospital_url":  hospital_url,
        "reason":        llm_reason,
    }

# ════════════════════════════════════════════════════════════════════════════
# PIPELINE HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _extract_disease(text: str) -> str:
    m = re.search(r"IDENTIFIED_DISEASE:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else "general"

def run_pipeline(query: str) -> dict:
    disease_raw        = disease_chain.invoke({"input": query})
    identified_disease = _extract_disease(disease_raw)
    clean_disease      = re.sub(
        r"\nIDENTIFIED_DISEASE:.*", "", disease_raw, flags=re.IGNORECASE).strip()
    nutrition_resp = nutrition_chain.invoke({"disease": identified_disease})

    # ── Severity assessment ──────────────────────────────────────────────────
    severity = _assess_severity(query, identified_disease, clean_disease)

    return {
        "identified_disease": identified_disease,
        "disease_info":       clean_disease,
        "nutrition_plan":     nutrition_resp,
        "reply":              f"{clean_disease}\n\n{nutrition_resp}",
        "severity":           severity,
    }


# ════════════════════════════════════════════════════════════════════════════
# VOICE HELPER  (Groq Whisper)
# ════════════════════════════════════════════════════════════════════════════
def transcribe_audio(audio_bytes: bytes, mime: str = "audio/webm") -> str:
    import httpx
    ext = "webm" if "webm" in mime else ("mp3" if "mp3" in mime else "wav")
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes); tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (f"audio.{ext}", f, mime)},
                data={"model": "whisper-large-v3"},
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json().get("text", "")
    finally:
        os.unlink(tmp_path)

# ════════════════════════════════════════════════════════════════════════════
# IMAGE HELPER — 3-Layer Pipeline
#
#  Layer 1 : Tesseract OCR        → raw text extraction from image
#  Layer 2 : Groq Vision LLM      → understand handwriting + medical context
#  Layer 3 : RAG cross-check      → validate drug names / dosages via medic_book
# ════════════════════════════════════════════════════════════════════════════

def _preprocess_image_for_ocr(image_bytes: bytes):
    """
    Enhance image quality before feeding to Tesseract.
    Converts to grayscale, increases contrast, removes noise.
    Returns a PIL Image object.
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── Upscale small images (Tesseract works best at 300+ DPI) ──
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ── Grayscale ──
    img = img.convert("L")

    # ── Sharpen edges (helps with handwriting) ──
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)   # double-sharpen for cursive

    # ── Boost contrast ──
    img = ImageEnhance.Contrast(img).enhance(2.5)

    # ── Slight denoise ──
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def _layer1_tesseract_ocr(image_bytes: bytes) -> str:
    """
    Layer 1: Tesseract OCR
    Extracts raw text — works well for printed prescriptions,
    partial results for handwriting which Layer 2 will fix.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        img = _preprocess_image_for_ocr(image_bytes)

        # PSM 6 = assume a uniform block of text (good for prescriptions)
        # OEM 3 = use LSTM neural net engine (most accurate)
        config = "--oem 3 --psm 6 -l eng"
        text   = pytesseract.image_to_string(img, config=config).strip()

        if text:
            print(f"✅ Layer 1 (Tesseract): extracted {len(text)} chars")
        else:
            print("⚠️  Layer 1 (Tesseract): no text found, Layer 2 will handle")

        return text

    except ImportError:
        print("⚠️  pytesseract not installed — skipping Layer 1")
        return ""
    except Exception as e:
        print(f"⚠️  Layer 1 Tesseract error: {e}")
        return ""


def _layer2_groq_vision(image_bytes: bytes, mime: str,
                         ocr_text: str) -> str:
    """
    Layer 2: Groq Vision LLM
    - Sees the actual image
    - Gets the OCR text as context to cross-check / fill gaps
    - Specialised prompt for doctor prescriptions + handwriting
    """
    import httpx

    b64 = base64.b64encode(image_bytes).decode()

    ocr_context = (
        f"\n\nFor reference, an OCR tool extracted this raw text from the image "
        f"(may have errors especially for handwriting):\n```\n{ocr_text}\n```\n"
        f"Use this as a hint but trust your own visual reading over OCR errors."
        if ocr_text else ""
    )

    vision_prompt = (
        "You are an expert medical prescription reader with years of experience "
        "deciphering doctor handwriting.\n\n"

        "Carefully examine this image and extract ALL of the following if present:\n"
        "1. Patient name and age\n"
        "2. Doctor name and registration number\n"
        "3. Date of prescription\n"
        "4. Diagnosed condition or symptoms\n"
        "5. Medicines prescribed — for EACH medicine extract:\n"
        "   - Full medicine name (expand abbreviations, e.g. 'Amox' → 'Amoxicillin')\n"
        "   - Dosage (mg/ml)\n"
        "   - Frequency (e.g. twice daily, morning/night)\n"
        "   - Duration (e.g. 5 days, 1 week)\n"
        "   - Route (oral/topical/injection)\n"
        "6. Special instructions (e.g. take with food, avoid dairy)\n"
        "7. Follow-up date if mentioned\n\n"

        "For handwritten text:\n"
        "- Common abbreviations: b.d./b.i.d.=twice daily, t.d.s./t.i.d.=three times daily, "
        "o.d.=once daily, q.i.d.=four times daily, p.r.n.=as needed, "
        "a.c.=before meals, p.c.=after meals, h.s.=at bedtime, "
        "stat=immediately, sos=if needed\n"
        "- If a word is unclear, write your best interpretation followed by [?]\n"
        "- Never guess a dosage number — if unclear write [illegible]\n\n"

        "Format your response as structured text with clear sections.\n"
        f"{ocr_context}"
    )

    # Normalize mime type — browser sometimes sends empty or wrong type
    if not mime or mime == "application/octet-stream":
        mime = "image/jpeg"
    # Groq vision only accepts jpeg/png/webp/gif
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        mime = "image/jpeg"

    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # latest Groq vision model
                "messages": [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": vision_prompt}
                ]}],
                "max_tokens": 800,
            },
            timeout=45,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"]
        print(f"✅ Layer 2 (Groq Vision): {len(result)} chars extracted")
        return result
    except Exception as e:
        print(f"❌ Layer 2 Groq Vision error: {e}")
        # Fallback: return OCR text + note so Layer 3 can still run
        fallback = ocr_text if ocr_text else "Image could not be analysed by vision model."
        return f"[Vision model unavailable. OCR result:]\n{fallback}"


def _layer3_rag_crosscheck(vision_text: str) -> str:
    """
    Layer 3: RAG cross-check via medic_book
    - Extracts medicine names from Layer 2 output
    - Validates them against the medical knowledge base
    - Flags unknown drugs or dangerous combinations
    - Returns an enriched, validated summary
    """
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical pharmacist validating a prescription reading. "
         "Using the retrieved medical context below, validate and enrich the "
         "prescription details provided by the user.\n\n"
         "For each medicine mentioned:\n"
         "1. Confirm it is a real medicine (flag if not found in context)\n"
         "2. Verify the dosage is within safe range\n"
         "3. Note common side effects briefly\n"
         "4. Note any important food/drug interactions\n"
         "5. If a medicine name looks like a handwriting error, suggest "
         "   the most likely correct name\n\n"
         "Also decode any remaining Latin abbreviations.\n"
         "End with a clean SUMMARY section listing all medicines with "
         "confirmed names, dosages and schedule.\n\n"
         "Context:\n{context}"),
        ("human", "Prescription reading:\n{input}")
    ])

    rag_chain = (
        {
            "context": itemgetter("input") | retriever_general,
            "input":   itemgetter("input"),
        }
        | rag_prompt
        | chat_model
        | StrOutputParser()
    )

    result = rag_chain.invoke({"input": vision_text})
    print(f"✅ Layer 3 (RAG cross-check): validation complete")
    return result


def describe_image(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """
    Full 3-layer prescription / medical image analysis pipeline.

    Layer 1 → Tesseract OCR        (raw text, fast, free)
    Layer 2 → Groq Vision LLM      (handwriting understanding + structure)
    Layer 3 → RAG cross-check      (drug validation via medic_book)

    For non-prescription images (skin conditions, wounds):
    Layer 1 produces no text → Layer 2 does visual medical description →
    Layer 3 validates against medical knowledge base.
    """
    print("\n🔬 Starting 3-layer image analysis…")

    # ── Layer 1: Tesseract OCR ───────────────────────────────────────────────
    ocr_text = _layer1_tesseract_ocr(image_bytes)

    # ── Layer 2: Groq Vision ─────────────────────────────────────────────────
    vision_text = _layer2_groq_vision(image_bytes, mime, ocr_text)

    # ── Layer 3: RAG cross-check ─────────────────────────────────────────────
    final_result = _layer3_rag_crosscheck(vision_text)

    return final_result

# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")

# ── Text chat (your original /get endpoint, now upgraded) ───────────────────
@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        user_id    = get_current_user()
        user_msg   = request.form.get("msg", "").strip()
        session_id = request.form.get("session_id") or None

        if not user_msg:
            return jsonify({"error": "Please enter a message!"}), 400

        sid     = _ensure_session(user_id, session_id)
        history = _get_msgs(user_id, sid)
        ctx     = _history_ctx(history)
        query   = f"{ctx}\nPatient: {user_msg}" if ctx else user_msg

        result  = run_pipeline(query)

        _append_msgs(user_id, sid, [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": result["reply"]},
        ])
        _update_session_meta(user_id, sid,
                             result["identified_disease"],
                             result["severity"]["severity"])

        return jsonify({
            "session_id":    sid,
            "disease":       result["identified_disease"],
            "disease_info":  result["disease_info"],
            "nutrition_plan":result["nutrition_plan"],
            "reply":         result["reply"],
            "severity":      result["severity"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Sorry, I'm having trouble. Please try again."}), 500

# ── Voice ────────────────────────────────────────────────────────────────────
@app.route("/get/voice", methods=["POST"])
def get_voice_response():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "audio file required"}), 400
        user_id    = get_current_user()
        audio_file = request.files["audio"]
        session_id = request.form.get("session_id") or None
        mime       = audio_file.mimetype or "audio/webm"

        transcript = transcribe_audio(audio_file.read(), mime)
        if not transcript.strip():
            return jsonify({"error": "Could not transcribe audio"}), 422

        sid     = _ensure_session(user_id, session_id)
        history = _get_msgs(user_id, sid)
        ctx     = _history_ctx(history)
        query   = f"{ctx}\nPatient: {transcript}" if ctx else transcript
        result  = run_pipeline(query)

        _append_msgs(user_id, sid, [
            {"role": "user",      "content": f"[Voice] {transcript}"},
            {"role": "assistant", "content": result["reply"]},
        ])
        _update_session_meta(user_id, sid,
                             result["identified_disease"],
                             result["severity"]["severity"])
        return jsonify({"session_id": sid, "transcript": transcript, **result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Image ────────────────────────────────────────────────────────────────────
@app.route("/get/image", methods=["POST"])
def get_image_response():
    try:
        if "image" not in request.files:
            return jsonify({"error": "image file required"}), 400
        user_id    = get_current_user()
        image_file = request.files["image"]
        extra_msg  = request.form.get("message", "").strip()
        session_id = request.form.get("session_id") or None
        mime       = image_file.mimetype or "image/jpeg"

        description = describe_image(image_file.read(), mime)
        combined    = f"{extra_msg}\n\nImage analysis: {description}" if extra_msg else description

        sid     = _ensure_session(user_id, session_id)
        history = _get_msgs(user_id, sid)
        ctx     = _history_ctx(history)
        query   = f"{ctx}\nPatient: {combined}" if ctx else combined
        result  = run_pipeline(query)

        _append_msgs(user_id, sid, [
            {"role": "user",      "content": f"[Image] {combined}"},
            {"role": "assistant", "content": result["reply"]},
        ])
        _update_session_meta(user_id, sid,
                             result["identified_disease"],
                             result["severity"]["severity"])
        return jsonify({"session_id": sid, "image_analysis": description, **result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500
@app.route("/history", methods=["GET"])
def get_history():
    user_id    = get_current_user()
    session_id = request.args.get("session_id")

    with get_db() as conn:
        if session_id:
            # Verify session belongs to this user
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id=? AND user_id=?",
                (session_id, user_id)
            ).fetchone()
            if not row:
                return jsonify({"error": "Session not found"}), 404
            msgs = conn.execute(
                "SELECT role, content, ts FROM messages WHERE session_id=? ORDER BY id",
                (session_id,)
            ).fetchall()
            return jsonify({
                "session_id": session_id,
                "messages":   [{"role": r["role"], "content": r["content"], "ts": r["ts"]}
                               for r in msgs]
            })

        # Return summary of all sessions for this user
        sessions = conn.execute(
            "SELECT session_id, created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()

        summary = []
        for s in sessions:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id=?",
                (s["session_id"],)
            ).fetchone()["c"]
            first = conn.execute(
                "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id LIMIT 1",
                (s["session_id"],)
            ).fetchone()
            summary.append({
                "session_id": s["session_id"],
                "created_at": s["created_at"],
                "msg_count":  count,
                "preview":    (first["content"][:80] if first else ""),
            })
    return jsonify({"sessions": summary})

@app.route("/history", methods=["DELETE"])
def clear_history():
    user_id    = get_current_user()
    session_id = request.args.get("session_id")
    with get_db() as conn:
        if session_id:
            conn.execute(
                "DELETE FROM messages WHERE session_id=? AND user_id=?",
                (session_id, user_id))
            conn.execute(
                "DELETE FROM sessions WHERE session_id=? AND user_id=?",
                (session_id, user_id))
            return jsonify({"deleted": session_id})
        # Delete all sessions + messages for this user
        conn.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM sessions WHERE user_id=?",  (user_id,))
    return jsonify({"deleted": "all"})

@app.route("/checkin", methods=["GET"])
def checkin():
    """
    Called on page load for returning users.
    Returns whether a check-in prompt should be shown,
    and what the last condition + severity was.
    Rules:
      - First time user         → no check-in
      - Last session < 3 days   → no check-in
      - Last session ≥ 3 days   → show check-in
    """
    user_id = get_current_user()
    with get_db() as conn:
        row = conn.execute(
            """SELECT session_id, created_at, last_disease, last_severity
               FROM sessions
               WHERE user_id=? AND last_disease != '' AND last_disease != 'general'
               ORDER BY created_at DESC LIMIT 1""",
            (user_id,)
        ).fetchone()

    if not row:
        return jsonify({"show": False, "reason": "first_time"})

    last_date_str = row["created_at"]
    try:
        last_date = datetime.fromisoformat(last_date_str.replace("Z", "+00:00"))
        days_ago  = (datetime.now(timezone.utc) - last_date).days
    except Exception:
        return jsonify({"show": False, "reason": "date_parse_error"})

    if days_ago < 3:
        return jsonify({"show": False, "reason": "too_soon", "days_ago": days_ago})

    # Format a friendly date string
    friendly_date = last_date.strftime("%B %d")   # e.g. "March 08"

    return jsonify({
        "show":          True,
        "days_ago":      days_ago,
        "last_date":     friendly_date,
        "last_disease":  row["last_disease"],
        "last_severity": row["last_severity"],
        "last_session":  row["session_id"],
        "question":      (
            f"Welcome back! 👋 Last time you visited on {friendly_date}, "
            f"you were dealing with **{row['last_disease']}** "
            f"(severity: {row['last_severity']}).\n\n"
            f"How are you feeling now compared to then?"
        )
    })


@app.route("/checkin/reply", methods=["POST"])
def checkin_reply():
    """
    User replies to the check-in question.
    LLM compares old condition vs new reply and gives a progress assessment.
    """
    try:
        user_id      = get_current_user()
        user_reply   = request.form.get("reply", "").strip()
        last_disease = request.form.get("last_disease", "")
        last_severity= request.form.get("last_severity", "")
        last_date    = request.form.get("last_date", "")
        session_id   = request.form.get("session_id") or None

        if not user_reply:
            return jsonify({"error": "Reply is required"}), 400

        # ── LLM progress assessment ──────────────────────────────────────────
        progress_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a caring medical follow-up assistant. "
             "A patient is returning after some days. "
             "Based on their previous condition and how they describe feeling now, "
             "assess their recovery progress.\n\n"
             "Classify progress as exactly ONE of:\n"
             "RECOVERING   — clearly improving, symptoms reduced\n"
             "STABLE       — no significant change, neither better nor worse\n"
             "WORSENING    — symptoms are getting worse\n"
             "NEW_SYMPTOMS — new unrelated symptoms have appeared\n\n"
             "Respond in EXACTLY this format:\n"
             "PROGRESS: <RECOVERING|STABLE|WORSENING|NEW_SYMPTOMS>\n"
             "ASSESSMENT: <2-3 warm, encouraging sentences summarising their progress>\n"
             "ADVICE: <one actionable sentence — what they should do next>"),
            ("human",
             "Previous condition ({last_date}): {last_disease} — severity was {last_severity}\n\n"
             "Patient says now: {reply}")
        ])

        progress_chain = progress_prompt | chat_model | StrOutputParser()
        llm_resp = progress_chain.invoke({
            "last_date":     last_date,
            "last_disease":  last_disease,
            "last_severity": last_severity,
            "reply":         user_reply,
        })

        prog_match   = re.search(r"PROGRESS:\s*(\w+)",    llm_resp, re.IGNORECASE)
        assess_match = re.search(r"ASSESSMENT:\s*(.+?)(?=ADVICE:|$)",
                                 llm_resp, re.IGNORECASE | re.DOTALL)
        advice_match = re.search(r"ADVICE:\s*(.+)",       llm_resp, re.IGNORECASE)

        progress   = prog_match.group(1).upper()   if prog_match   else "STABLE"
        assessment = assess_match.group(1).strip() if assess_match else llm_resp
        advice     = advice_match.group(1).strip() if advice_match else ""

        # Progress → emoji + color
        prog_config = {
            "RECOVERING":   {"icon": "📈", "color": "#00c853",
                             "label": "Recovering"},
            "STABLE":       {"icon": "➡️",  "color": "#ffd600",
                             "label": "Stable"},
            "WORSENING":    {"icon": "📉", "color": "#ff1744",
                             "label": "Worsening"},
            "NEW_SYMPTOMS": {"icon": "🔔", "color": "#ff6d00",
                             "label": "New Symptoms"},
        }
        cfg = prog_config.get(progress, prog_config["STABLE"])

        # Save this check-in as a new session message
        sid = _ensure_session(user_id, session_id)
        _append_msgs(user_id, sid, [
            {"role": "user",      "content": f"[Check-in] {user_reply}"},
            {"role": "assistant", "content": f"[Progress: {progress}] {assessment} {advice}"},
        ])

        return jsonify({
            "session_id": sid,
            "progress":   progress,
            "icon":       cfg["icon"],
            "color":      cfg["color"],
            "label":      cfg["label"],
            "assessment": assessment,
            "advice":     advice,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
