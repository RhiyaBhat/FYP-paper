import os
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH   = "data/docs"
CHROMA_PATH = "chroma_db"


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)       # fix hyphenated line breaks
    text = re.sub(r"\n", " ", text)        # newlines → spaces
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text.strip()


# =====================================================
# LOAD DOCUMENTS
# =====================================================

def load_documents():
    docs = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_PATH}")

    for file in pdf_files:
        loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
        loaded_docs = loader.load()
        for d in loaded_docs:
            d.metadata["source"] = file
        docs.extend(loaded_docs)
        print(f"  Loaded: {file} ({len(loaded_docs)} pages)")

    return docs


# =====================================================
# SPLIT + CLEAN
# =====================================================

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.page_content = clean_text(c.page_content)
    return chunks


# =====================================================
# BUILD VECTOR STORE
# =====================================================

def build_vectorstore():
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f"Vector store already exists at '{CHROMA_PATH}'. Skipping rebuild.")
        print("Delete the folder and re-run to force a rebuild.")
        return

    print("Loading documents...")
    docs = load_documents()

    print("Splitting and cleaning chunks...")
    chunks = split_documents(docs)

    print(f"Building vector store from {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ Indexed {len(chunks)} chunks into ChromaDB at '{CHROMA_PATH}'")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    build_vectorstore()