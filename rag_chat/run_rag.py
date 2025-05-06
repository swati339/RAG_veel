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
from redis_utils.redis_client import RedisSetup

redis_handler = RedisSetup()


def is_oneliner_request(user_input: str) -> bool:
    one_liner_keywords = ["one line", "oneliner", "brief answer", "in short","single line"]
    return any(keyword in user_input.lower() for keyword in one_liner_keywords)


def run_rag_pipeline():
    setup_logging()
    logger = logging.getLogger(__name__)
    load_dotenv()

    logger.info("Starting RAG pipeline...")

    PERSIST_DIR = "chroma_db"
    PDF_PATH = "/home/swati/Documents/veel_projects/RAG_veel/Docs/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    QUESTION = "What is regularization? I want answer in a paragraph."

    # Step 0: Redis cache check
    cached_answer = redis_handler.redis_get(QUESTION)
    if cached_answer:
        print("\n[Answer from Redis Cache]:\n", cached_answer)
        logger.info("Answer served from Redis cache.")
        return

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

    # Step 5: Select system prompt
    prompt_template = (
        SystemPrompt().get_oneliner_prompt()
        if is_oneliner_request(QUESTION)
        else SystemPrompt().get_paragraph_prompt()
    )

    # Step 6: Retrieve similar docs
    retrieved_docs = retriever.invoke(QUESTION)

    # Step 7: Use reranker
    reranker = ReRanker(embedding_model=embedder)
    reranked_docs = reranker.rerank_documents(query=QUESTION, docs=retrieved_docs)

    # Step 8: Final RAG response
    rag = RAGSystem(reranker=None, llm=llm_model, prompt_template=prompt_template)
    response = rag.ask_question(docs=reranked_docs, question=QUESTION)

    print("\n[Final RAG Answer]:\n", response)

    # Step 9: Store result in Redis
    redis_handler.redis_set(QUESTION, response)
    logger.info("Answer stored in Redis for future queries.")
