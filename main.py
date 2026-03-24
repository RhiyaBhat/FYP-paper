from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Existing
from src.rag.baseline import baseline_pipeline
from src.rag.hybrid import hybrid_pipeline

# New pipelines
from src.rag.bm25 import bm25_pipeline
from src.rag.hybrid_qe import hybrid_qe_pipeline
from src.rag.hybrid_rerank import hybrid_rerank_pipeline
from src.rag.full_pipeline import full_pipeline


def main():
    print("Loading vector store...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    queries = [
        "What is MuDoc?",
        "Explain the preprocessing pipeline",
        "What are the limitations of the system?"
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # =============================
        # BM25
        # =============================
        bm25_answer, _ = bm25_pipeline(vectordb, query)
        print("\n--- 🟡 BM25 ---\n")
        print(bm25_answer)

        # =============================
        # VECTOR (Baseline)
        # =============================
        base_answer, _ = baseline_pipeline(vectordb, query)
        print("\n--- 🔵 VECTOR ---\n")
        print(base_answer)

        # =============================
        # HYBRID
        # =============================
        hybrid_answer, hybrid_docs = hybrid_pipeline(vectordb, query)
        print("\n--- 🟢 HYBRID ---\n")
        print(hybrid_answer)

        # =============================
        # HYBRID + QUERY EXPANSION
        # =============================
        qe_answer, _ = hybrid_qe_pipeline(vectordb, query)
        print("\n--- 🟣 HYBRID + QUERY EXPANSION ---\n")
        print(qe_answer)

        # =============================
        # HYBRID + RERANK
        # =============================
        rerank_answer, _ = hybrid_rerank_pipeline(vectordb, query)
        print("\n--- 🟠 HYBRID + RERANK ---\n")
        print(rerank_answer)

        # =============================
        # FULL SYSTEM (YOUR CONTRIBUTION)
        # =============================
        full_answer, full_docs = full_pipeline(vectordb, query)
        print("\n--- 🔴 FULL SYSTEM ---\n")
        print(full_answer)

        # =============================
        # CONTEXT PREVIEW (BEST METHOD)
        # =============================
        print("\n--- 🔍 CONTEXT PREVIEW (FULL SYSTEM) ---\n")
        for d in full_docs[:2]:
            print(d.page_content[:200])
            print("-----")


if __name__ == "__main__":
    main()