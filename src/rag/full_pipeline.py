from src.rag.hybrid import HybridRetriever
from src.llm import get_llm
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use ONLY the context.
If missing → say: Not specified in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""
)


def hybrid_rerank_pipeline(vectorstore, query):

    retriever = HybridRetriever(vectorstore=vectorstore, k=10)

    docs = retriever.invoke(query)

    # rerank
    pairs = [(query, d.page_content[:300]) for d in docs]
    scores = reranker.predict(pairs)

    ranked_docs = [
        d for _, d in sorted(zip(scores, docs), reverse=True)
    ][:5]

    context = "\n\n".join([d.page_content[:400] for d in ranked_docs[:3]])

    llm = get_llm()
    answer = llm.invoke(QA_PROMPT.format(context=context, question=query))

    return answer, ranked_docs