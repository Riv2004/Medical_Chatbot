# 🏥 Medical Chatbot — RAG-Powered Pre-Screening Assistant

An AI-powered medical chatbot that identifies diseases from symptoms and provides
personalised daily nutrition plans. Built with LangChain, Pinecone, Groq, and Flask.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🔬 Symptom extraction | Step 0: LLM parses query into structured JSON (symptoms, duration, temp, age, onset, worsening) |
| 🧠 Two-pass disease chain | Chain A (RAG) → fallback pure-LLM if RAG returns `general` |
| 📊 Personalised nutrition | 7-nutrient table with ↑↓ vs normal — values specific to the identified disease |
| 🚨 Smart severity triage | Structured rules (age/temp/duration) + keyword match + LLM → worst-case wins |
| 📷 Prescription reader | 3-layer OCR: Tesseract → Groq Vision (llama-4-scout) → RAG cross-check |
| 🎙️ Voice input | Groq Whisper transcription |
| 💾 Chat history | SQLite per-user sessions (no extra DB server needed) |
| 🔄 Recovery check-in | Auto check-in for returning users after 3+ days |
| 📈 Progress tracking | RECOVERING / STABLE / WORSENING / NEW_SYMPTOMS |

---

## 🔁 Pipeline Architecture

```
User Query (Text / Voice / Image)
         |
         v  Step 0 -- _extract_symptoms()
              LLM -> JSON { symptoms, duration, temp, age, worsening, ... }
         |
         v  Step 1 -- disease_chain  (RAG, namespace: general)
              symptom_context + retrieved chunks -> Llama-3.3-70B
              -> IDENTIFIED_DISEASE: <name>
         |
         +-- if = "general" --> Step 2 -- fallback_disease_chain
         |                                Pure LLM, no Pinecone
         v
         Step 3 -- nutrition_chain  (RAG, namespace: nutrition)
              Richer query: "disease + nutritional requirements dietary guidelines"
              + disease_info from Step 1/2 -> 7-row table (up/down vs normal)
         |
         v  Step 4 -- _assess_severity()
              Structured rules: temp >= 104F -> URGENT, >= 105F -> CRITICAL
                                age < 2 or > 70 -> minimum MODERATE
                                7+ days + worsening -> URGENT
              Keyword match:   CRITICAL_CONDITIONS, URGENT_SYMPTOMS sets
              LLM triage:      balanced prompt with calibration rules
              Final =          worst of all three
```

### Nutrition Table -- up/down Symbols

Each row compares against a healthy adult baseline:

| Symbol | Meaning | Example |
|---|---|---|
| `up` | Higher than normal | Protein up for TB -- tissue repair |
| `down` | Lower than normal | Carbs down for Diabetes -- glucose control |
| `Restrict` | Severely limited | Potassium for CKD |
| (no symbol) | At normal level | Fiber for most infections |

Normal baseline: Carbs 45-65%, Protein 10-35%, Fat 20-35%, Water 2.7 L/day, Fiber 25-30 g/day.

---

## 🗂️ Project Structure

```
Medical_Chatbot/
├── app.py                  ← Flask app (main entry point)
├── store_index.py          ← Pinecone index builder (run once)
├── setup.py
├── requirements.txt
├── .env                    ← API keys (never commit)
├── chat_history.db         ← SQLite DB (auto-created on first run)
├── data/
│   ├── medic_book.pdf
│   ├── Herbal_Nutrients_and_Their_Health_Benefits.pdf
│   └── nutrition.pdf
├── src/
│   ├── __init__.py
│   ├── helper.py           ← PDF loading, embeddings, splitting
│   └── prompt.py           ← System prompts for both RAG chains
├── templates/
│   └── index.html          ← Chat UI
├── static/
│   └── style.css           ← Dark theme styles
└── research/
    └── medical_chatbot_nutrition_rag.ipynb  ← Backup notebook
```

---

## ⚙️ How to Run

### STEP 01 — Clone the repository

```bash
git clone https://github.com/Riv2004/Medical_Chatbot.git
cd Medical_Chatbot
```

### STEP 02 — Create and activate conda environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### STEP 03 — Install Tesseract binary (for prescription OCR)

```bash
# Mac
brew install tesseract

# Ubuntu / WSL
sudo apt install tesseract-ocr -y
```

### STEP 04 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### STEP 05 — Set up environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_INDEX_NAME=medical-chatbot
SECRET_KEY=any-random-string
TESSERACT_CMD=/usr/local/bin/tesseract
```

### STEP 06 — Add your PDFs

Put these 3 PDFs inside the `data/` folder:
- `medic_book.pdf`
- `Herbal_Nutrients_and_Their_Health_Benefits.pdf`
- `nutrition.pdf`

### STEP 07 — Build the Pinecone index (run only once)

```bash
python store_index.py
```

This splits the PDFs into chunks and upserts them into two Pinecone namespaces:
- `general` → disease + herbal chunks
- `nutrition` → nutrition chunks

> ⏭️ **Skip this step** if you already have a populated Pinecone index.

### STEP 08 — Run the app

```bash
python app.py
```

Open **http://localhost:5050** in your browser.

---

## 🔑 API Keys Required

| Key | Where to get |
|---|---|
| `PINECONE_API_KEY` | [pinecone.io](https://www.pinecone.io) |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |

---

## 🚀 Every Time You Run

```bash
conda activate medibot
python app.py
```

---

## 🛡️ Disclaimer

> This chatbot is for **pre-screening purposes only**.  
> It is not a substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified healthcare professional.
