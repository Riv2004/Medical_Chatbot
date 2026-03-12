import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from helper import load_pdf_files, filter_to_minimal_docs, split_docs, get_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in environment variables.")

# ── Create / verify index ────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = pc.list_indexes().names()
print("Existing indexes:", existing_indexes)

if INDEX_NAME not in existing_indexes:
    print(f"Creating new index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status.ready:
        print("Waiting for index to be ready...")
        time.sleep(2)
else:
    print(f"Using existing index: {INDEX_NAME}")

# ── Load, filter, split ───────────────────────────────────────────────────────
print("\nLoading and processing PDFs...")
docs         = load_pdf_files()
minimal_docs = filter_to_minimal_docs(docs)
chunks       = split_docs(minimal_docs)
print(f"Total chunks: {len(chunks)}")

embeddings = get_embeddings()

# ── Split by namespace ────────────────────────────────────────────────────────
general_chunks   = [c for c in chunks if c.metadata.get("book_type") in ("disease", "herbal")]
nutrition_chunks = [c for c in chunks if c.metadata.get("book_type") == "nutrition"]

# Fallback: if book_type not tagged, put everything in general
if not general_chunks and not nutrition_chunks:
    print("⚠️  No book_type tags found — upserting all chunks to namespace 'general'")
    general_chunks = chunks

print(f"\nDisease + Herbal chunks → namespace 'general'  : {len(general_chunks)}")
print(f"Nutrition chunks        → namespace 'nutrition' : {len(nutrition_chunks)}")

# ── Upsert ────────────────────────────────────────────────────────────────────
if general_chunks:
    print("\nUpserting general chunks...")
    PineconeVectorStore.from_documents(
        documents=general_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace="general"
    )
    print(f"✅ {len(general_chunks)} chunks → namespace 'general'")

if nutrition_chunks:
    print("\nUpserting nutrition chunks...")
    PineconeVectorStore.from_documents(
        documents=nutrition_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace="nutrition"
    )
    print(f"✅ {len(nutrition_chunks)} chunks → namespace 'nutrition'")

print("\n✅ Index populated successfully.")
