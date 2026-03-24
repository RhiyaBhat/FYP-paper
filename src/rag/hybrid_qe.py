from src.rag.hybrid import HybridRetriever
from src.llm import get_llm
from langchain_core.prompts import PromptTemplate


def expand_query(query):
    return query + " explanation details process method"


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


def hybrid_qe_pipeline(vectorstore, query):

    expanded_query = expand_query(query)

    retriever = HybridRetriever(vectorstore=vectorstore, k=5)

    docs = retriever.invoke(expanded_query)

    context = "\n\n".join([d.page_content[:400] for d in docs[:3]])

    llm = get_llm()
    answer = llm.invoke(QA_PROMPT.format(context=context, question=query))

    return answer, docs