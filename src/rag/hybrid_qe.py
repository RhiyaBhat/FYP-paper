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


def expand_query(query: str) -> str:
    """
    Simple rule-based query expansion.
    Appends general retrieval keywords to improve BM25 recall.
    """
    return query + " explanation details process method overview"


def hybrid_qe_pipeline(vectorstore, query: str):
    """
    Hybrid retrieval with query expansion.
    The expanded query is used for retrieval only;
    the original query is passed to the LLM.
    """
    expanded_query = expand_query(query)

    retriever = HybridRetriever(vectorstore=vectorstore, k=5)
    docs = retriever.invoke(expanded_query)

    context_parts = []
    for d in docs[:5]:
        text = d.page_content.strip().replace("\n", " ")
        context_parts.append(text[:600])
    context = "\n\n".join(context_parts)

    llm = get_llm()
    answer = llm.invoke(QA_PROMPT.format(context=context, question=query))

    return answer, docs