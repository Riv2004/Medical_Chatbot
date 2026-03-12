# helper.py
from typing import List
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = os.getenv(
    "DATA_DIR",
    "/Users/ritabratabhattacharya/Desktop/Python/PROJECTS/Medical_Chatbot/data"
)

# Maps filename → Pinecone namespace tag
# disease + herbal go to namespace "general"
# nutrition goes to namespace "nutrition"
PDF_BOOK_MAP = {
    "medic_book.pdf":                                "disease",
    "Herbal_Nutrients_and_Their_Health_Benefits.pdf": "herbal",
    "nutrition.pdf":                                  "nutrition",
}


def load_pdf_files(data_dir: str = DATA_DIR) -> List[Document]:
    """Load all PDFs and inject book_type metadata on every page."""
    all_docs = []
    for filename, book_type in PDF_BOOK_MAP.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            # Fallback: load whatever PDFs exist without tagging
            print(f"⚠️  Not found: {filepath}, skipping")
            continue
        loader = PyPDFLoader(filepath)
        pages  = loader.load()
        for page in pages:
            page.metadata["book_type"] = book_type
            page.metadata["source"]    = filename
        all_docs.extend(pages)
        print(f"✅ Loaded {len(pages):>4} pages  [{book_type}]  {filename}")

    # Fallback: if none matched by name, load all PDFs in directory (no tagging)
    if not all_docs:
        print("⚠️  No named PDFs found — loading all PDFs without book_type tag")
        loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        all_docs = loader.load()

    return all_docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only source + book_type metadata (strips noisy PDF metadata)."""
    minimal: List[Document] = []
    for doc in docs:
        minimal.append(Document(
            page_content=doc.page_content,
            metadata={
                "source":    doc.metadata.get("source"),
                "book_type": doc.metadata.get("book_type", "general"),
            }
        ))
    return minimal


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
