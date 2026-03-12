#!/bin/bash

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p src
mkdir -p research
mkdir -p data

# ── Create source files ───────────────────────────────────────────────────────
touch src/__init__.py
touch src/helper.py
touch src/prompt.py

# ── Create root files ─────────────────────────────────────────────────────────
touch .env
touch setup.py
touch app.py
touch store_index.py
touch requirements.txt

# ── Create research / backup notebook ────────────────────────────────────────
touch research/medical_chatbot_nutrition_rag.ipynb

# ── Create Flask template folder ─────────────────────────────────────────────
mkdir -p templates
mkdir -p static
touch templates/index.html
touch static/style.css

echo "✅ Directory and file structure created successfully."
echo ""
echo "📁 Project structure:"
echo "   ├── app.py              (Flask app — main entry point)"
echo "   ├── store_index.py      (Pinecone index builder — run once)"
echo "   ├── setup.py"
echo "   ├── requirements.txt"
echo "   ├── .env                (API keys — never commit this)"
echo "   ├── data/               (put your PDFs here)"
echo "   ├── src/"
echo "   │   ├── __init__.py"
echo "   │   ├── helper.py       (PDF loading, embeddings, splitting)"
echo "   │   └── prompt.py       (system prompts for both chains)"
echo "   ├── templates/"
echo "   │   └── index.html      (chat UI)"
echo "   ├── static/"
echo "   │   └── style.css       (dark theme styles)"
echo "   └── research/"
echo "       └── medical_chatbot_nutrition_rag.ipynb  (backup notebook)"
