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

# Lazy-loaded — only imported when this pipeline is first called,
# not at startup, so it doesn't slow down the whole app.
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def hybrid_rerank_pipeline(vectorstore, query: str):
    """
    Hybrid retrieval followed by cross-encoder reranking.
    Retrieves top-10, reranks, uses top-5 for context.
    """
    retriever = HybridRetriever(vectorstore=vectorstore, k=10)
    docs = retriever.invoke(query)

    # Rerank using cross-encoder
    reranker = get_reranker()
    pairs = [(query, d.page_content[:400]) for d in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    ]

    # Use top 5 after reranking
    top_docs = ranked_docs[:5]

    context_parts = []
    for d in top_docs:
        text = d.page_content.strip().replace("\n", " ")
        context_parts.append(text[:600])
    context = "\n\n".join(context_parts)

    llm = get_llm()
    answer = llm.invoke(QA_PROMPT.format(context=context, question=query))

    return answer, top_docs