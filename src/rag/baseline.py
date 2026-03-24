from langchain_core.prompts import PromptTemplate
from src.llm import get_llm


# =====================================================
# PROMPT
# =====================================================

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


# =====================================================
# BASELINE PIPELINE
# =====================================================

def baseline_pipeline(vectorstore, query, debug=False):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(query)

    context_parts = []
    for d in docs:
        text = d.page_content.strip().replace("\n", " ")
        context_parts.append(text[:600])

    context = "\n\n".join(context_parts)

    if debug:
        print("\n=== BASELINE CONTEXT ===\n", context[:1000])

    prompt = QA_PROMPT.format(context=context, question=query)

    llm = get_llm()
    answer = llm.invoke(prompt)

    return answer, docs