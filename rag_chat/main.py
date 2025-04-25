from models.pdf_processor import PDFProcessor
from models.embeddings import EmbeddingModel
from rag_chat.models.llm_model import LLMModel
from models.reranking import ReRanker
from rag_chat.models.rag_pipeline import RAGSystem
from configs.logging_config import setup_logging
from prompts.prompt_templates import SystemPrompt
from langchain.vectorstores import Chroma


def is_oneliner_request(user_input: str) -> bool:
    one_liner_keywords = [
        "one line",
        "oneliner",
        "brief answer",
        "in short",
    ]
    return any(keyword in user_input.lower() for keyword in one_liner_keywords)


if __name__ == "__main__":
    setup_logging()

    # 1. Load and split documents
    processor = PDFProcessor(file_path="docs/attention.pdf")
    all_splits = processor.load_and_split()

    # 2. Embedding model
    embedder = EmbeddingModel()

    # 3. Vectorstore from documents
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=embedder.model, persist_directory="chroma_db"
    )

    # 4. Setup retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # 5. LLM
    llm_model = LLMModel()

    question = "What is transformer in deep learning?"

    # 7. Select prompt based on user input
    prompt_template = (
        SystemPrompt().get_oneliner_prompt()
        if is_oneliner_request(question)
        else SystemPrompt().get_paragraph_prompt()
    )

    # --- Similarity-based retriever answer ---
    retrieved_docs = retriever.invoke(question)
    rag_similarity = RAGSystem(
        reranker=None, llm=llm_model, prompt_template=prompt_template
    )
    print("\n--- Similarity Search Response ---")
    print(rag_similarity.ask_question(docs=retrieved_docs, question=question))

    # --- Reranked answer ---
    reranker = ReRanker(embedding_model=embedder)
    rag_reranked = RAGSystem(
        reranker=reranker, llm=llm_model, prompt_template=prompt_template
    )
    print("\n--- Reranker-enhanced Response ---")
    print(rag_reranked.ask_question(docs=all_splits, question=question))
