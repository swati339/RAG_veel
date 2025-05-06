import os
import logging
from dotenv import load_dotenv

from models.pdf_processor import PDFProcessor
from models.embeddings import EmbeddingModel
from models.llm_model import LLMModel
from models.reranking import ReRanker
from models.rag_pipeline import RAGSystem
from configs.logging_config import setup_logging
from prompts.prompt_templates import SystemPrompt

from langchain_community.vectorstores import Chroma


def is_oneliner_request(user_input: str) -> bool:
    one_liner_keywords = ["one line", "oneliner", "brief answer", "in short"]
    return any(keyword in user_input.lower() for keyword in one_liner_keywords)


def run_rag_pipeline():
    setup_logging()
    logger = logging.getLogger(__name__)
    load_dotenv()

    logger.info("Starting RAG pipeline...")

    PERSIST_DIR = "chroma_db"
    PDF_PATH = "/home/swati/Documents/veel_projects/RAG_veel/Docs/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    QUESTION = "What is transformer in deep learning?"

    # Step 1: Load and split PDF
    processor = PDFProcessor(file_path=PDF_PATH)
    all_splits = processor.load_and_split()

    # Step 2: Load embedding model
    embedder = EmbeddingModel()

    # Step 3: Load or create vectorstore
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        logger.info("Loading existing Chroma vectorstore...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedder.model
        )
    else:
        logger.info("Creating new Chroma vectorstore from documents...")
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embedder.model,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Step 4: Load LLM
    llm_model = LLMModel(model_name="gpt-4o-mini")

    # Step 5: Select prompt
    prompt_template = (
        SystemPrompt().get_oneliner_prompt()
        if is_oneliner_request(QUESTION)
        else SystemPrompt().get_paragraph_prompt()
    )

    # Step 6: RAG without reranker
    retrieved_docs = retriever.invoke(QUESTION)
    rag_similarity = RAGSystem(reranker=None, llm=llm_model, prompt_template=prompt_template)

    logger.info("--- Similarity Search Response ---")
    response_sim = rag_similarity.ask_question(docs=retrieved_docs, question=QUESTION)
    print("\n[Similarity-Based Answer]:\n", response_sim)

    # Step 7: RAG with reranker
    reranker = ReRanker(embedding_model=embedder)
    rag_reranked = RAGSystem(reranker=reranker, llm=llm_model, prompt_template=prompt_template)

    logger.info("--- Reranker-enhanced Response ---")
    response_reranked = rag_reranked.ask_question(docs=all_splits, question=QUESTION)
    print("\n[Re-ranked Answer]:\n", response_reranked)
