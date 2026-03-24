from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.llm import get_llm
from langchain_core.prompts import PromptTemplate


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


def bm25_pipeline(vectorstore, query):

    results = vectorstore.get()

    docs = [
        Document(page_content=c, metadata=m)
        for c, m in zip(
            results.get("documents", []),
            results.get("metadatas", [])
        )
    ]

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 5

    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content[:400] for d in docs[:3]])

    llm = get_llm()
    answer = llm.invoke(QA_PROMPT.format(context=context, question=query))

    return answer, docs