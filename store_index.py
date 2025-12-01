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
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in environment variables.")

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

index = pc.Index(INDEX_NAME)
print("Index ready:", INDEX_NAME)

print("Loading and processing PDFs...")
docs = load_pdf_files()
minimal_docs = filter_to_minimal_docs(docs)
texts_chunk = split_docs(minimal_docs)
print(f"Number of text chunks: {len(texts_chunk)}")

embeddings = get_embeddings()

print("Creating / updating Pinecone vector store...")
vector_store = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Index populated successfully.")
