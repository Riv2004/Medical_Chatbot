import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from helper import get_embeddings
from prompt import system_prompt

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("PINECONE_API_KEY or GEMINI_API_KEY missing!")

app = Flask(__name__)
CORS(app)

def get_vector_store():
    embeddings = get_embeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        raise ValueError(f"Pinecone index '{INDEX_NAME}' not found. Run store_index.py first.")
    
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# chat_model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=GEMINI_API_KEY,
#     temperature=0,
#     max_output_tokens=200,
# )
chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",  # or "llama-3.1-8b-instant"
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

rag_chain = (
    {
        "context": itemgetter("input") | retriever,
        "input": itemgetter("input")
    }
    | prompt
    | chat_model
    | StrOutputParser()
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])  # ← Matches your HTML exactly!
def get_bot_response():
    try:
        user_message = request.form.get("msg", "").strip()
        
        if not user_message:
            return "Please enter a message!", 400
        
        answer = rag_chain.invoke({"input": user_message})
        return answer  # ← Plain text response (your HTML expects this)
    
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I'm having trouble processing that. Please try again.", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
