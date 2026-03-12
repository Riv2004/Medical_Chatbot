# 🏥 Medical Chatbot — RAG-Powered Pre-Screening Assistant

An AI-powered medical chatbot that identifies diseases from symptoms and provides
personalised daily nutrition plans. Built with LangChain, Pinecone, Groq, and Flask.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🧠 Dual RAG chains | Chain A: disease identification · Chain B: nutrition table |
| 📊 Nutrition plan | 7-nutrient daily % table (Carbs, Protein, Fat, Vitamins, Minerals, Water, Fiber) |
| 🚨 Severity alerts | 3-step triage: keyword match → LLM → worst-case wins (CRITICAL/URGENT/MODERATE/MILD) |
| 📷 Prescription reader | 3-layer OCR: Tesseract → Groq Vision → RAG cross-check |
| 🎙️ Voice input | Groq Whisper transcription |
| 💾 Chat history | SQLite per-user sessions (no extra DB server needed) |
| 🔄 Recovery check-in | Auto check-in for returning users after 3+ days |
| 📈 Progress tracking | RECOVERING / STABLE / WORSENING / NEW_SYMPTOMS |

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
