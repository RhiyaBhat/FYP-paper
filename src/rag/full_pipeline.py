"""
full_pipeline.py

The complete system combining all three techniques:
  1. Query Expansion       — improves BM25 keyword recall
  2. Hybrid Retrieval      — vector (MMR) + BM25
  3. Cross-Encoder Rerank  — re-scores retrieved chunks for precision

This is the top-performing pipeline and your primary research contribution.
"""

from src.rag.hybrid import HybridRetriever
from src.llm import get_llm
from langchain_core.prompts import PromptTemplate


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a document QA assistant.

Answer using ONLY the provided context.
Extract the most relevant information and explain clearly.
Even if the answer is partially present, form a complete explanation.
Only say "Not specified in the provided documents" if absolutely no relevant information exists.

Context:
{context}

Question:
{question}

Answer:
"""
)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def expand_query(query: str) -> str:
    return query + " explanation details process method overview"


def full_pipeline(vectorstore, query: str, debug: bool = False):
    """
    Full pipeline: Query Expansion → Hybrid Retrieval → Reranking → LLM.

    Steps:
      1. Expand the query with general keywords
      2. Run hybrid retrieval (vector MMR + BM25) on expanded query
      3. Rerank results with a cross-encoder
      4. Build context from top-5 reranked chunks
      5. Generate answer with Gemini
    """

    # Step 1: Query expansion
    expanded_query = expand_query(query)

    # Step 2: Hybrid retrieval (fetch more so reranker has room to work)
    retriever = HybridRetriever(vectorstore=vectorstore, k=10)
    docs = retriever.invoke(expanded_query)

    # Step 3: Rerank
    reranker = get_reranker()
    pairs = [(query, d.page_content[:400]) for d in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    ]

    # Step 4: Build context from top 5
    top_docs = ranked_docs[:5]

    context_parts = []
    for d in top_docs:
        text = d.page_content.strip().replace("\n", " ")
        context_parts.append(text[:600])
    context = "\n\n".join(context_parts)

    if debug:
        print("\n=== FULL PIPELINE CONTEXT ===\n", context[:1000])

    # Step 5: Generate answer
    llm = get_llm()
    answer = llm.invoke(QA_PROMPT.format(context=context, question=query))

    return answer, top_docs