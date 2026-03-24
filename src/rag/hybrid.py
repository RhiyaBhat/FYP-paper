from typing import List, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from pydantic import model_validator

from src.llm import get_llm


# =====================================================
# HYBRID RETRIEVER (OPTIMIZED)
# =====================================================

class HybridRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 5
    vector_retriever: Any = None
    bm25_retriever: Any = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _build_retrievers(self):

        # 🔥 VECTOR (MMR optimized)
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.k,
                "fetch_k": 100,      # 🔥 reduced from 300
                "lambda_mult": 0.7   # 🔥 balanced relevance/diversity
            }
        )

        # 🔥 BM25 setup
        docs = self._load_all_documents()

        if docs:
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = self.k
            self.bm25_retriever = bm25

        return self

    def _load_all_documents(self) -> List[Document]:
        results = self.vectorstore.get()

        docs = []
        for content, metadata in zip(
            results.get("documents", []),
            results.get("metadatas", [])
        ):
            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    # 🔥 IMPROVED MERGING (IMPORTANT FIX)
    def _merge_results(
        self,
        vector_docs: List[Document],
        keyword_docs: List[Document]
    ) -> List[Document]:

        seen = set()
        merged = []

        # 🔥 prioritize vector results
        for doc in vector_docs:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)

        # 🔥 then add keyword results
        for doc in keyword_docs:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)

        return merged[:self.k]

    def _get_relevant_documents(self, query: str) -> List[Document]:

        vector_docs = self.vector_retriever.invoke(query)

        if self.bm25_retriever is None:
            return vector_docs[:self.k]

        keyword_docs = self.bm25_retriever.invoke(query)

        return self._merge_results(vector_docs, keyword_docs)


# =====================================================
# PROMPT (CLEAN + FAST)
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
# HYBRID PIPELINE (OPTIMIZED)
# =====================================================

def hybrid_pipeline(vectorstore, query, debug=False):

    retriever = HybridRetriever(vectorstore=vectorstore, k=5)

    docs = retriever.invoke(query)

    # 🔥 OPTIMIZED CONTEXT (BIG SPEED BOOST)
    context_parts = []

    for d in docs[:3]:  # 🔥 limit to top 3
        text = d.page_content.strip().replace("\n", " ")
        context_parts.append(text[:300])  # 🔥 reduced size

    context = "\n\n".join(context_parts)

    if debug:
        print("\n=== HYBRID CONTEXT ===\n", context)

    prompt = QA_PROMPT.format(
        context=context,
        question=query
    )

    llm = get_llm()
    answer = llm.invoke(prompt)

    return answer, docs