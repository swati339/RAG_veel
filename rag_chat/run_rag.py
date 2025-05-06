import logging
import os
from dotenv import load_dotenv
from rag_chat.models.pdf_processor import PDFProcessor
from rag_chat.models.embeddings import EmbeddingModel
from rag_chat.models.llm_model import LLMModel
from rag_chat.models.reranking import ReRanker
from rag_chat.models.rag_pipeline import RAGSystem
from rag_chat.configs.logging_config import setup_logging
from rag_chat.prompts.prompt_templates import SystemPrompt
from langchain_community.vectorstores import Chroma
from rag_chat.redis_utils.redis_client import RedisSetup 

# Setup
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize Redis handler
redis_handler = RedisSetup()

class PipeLine:
    def __init__(self, reranker, llm, prompt_template):
        self.reranker = reranker
        self.llm = llm
        self.prompt_template = prompt_template

    @staticmethod
    def is_oneliner_request(user_input: str) -> bool:
        one_liner_keywords = ["one line", "oneliner", "brief answer", "in short", "single line"]
        return any(keyword in user_input.lower() for keyword in one_liner_keywords)

    def run_rag_pipeline(self, question: str):
        logger.info("Starting RAG pipeline for question: %s", question)

        PERSIST_DIR = "chroma_db"
        PDF_PATH = "/home/swati/Documents/veel_projects/RAG_veel/Docs/NIPS-2017-attention-is-all-you-need-Paper.pdf"

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

        # Step 4: Retrieve similar docs
        retrieved_docs = retriever.invoke(question)

        # Step 5: Use reranker
        reranker = ReRanker(embedding_model=embedder)
        reranked_docs = reranker.rerank_documents(query=question, docs=retrieved_docs)

        # Step 6: Redis cache check (hash: "answers")
        cached_answer = redis_handler.redis_hget("answers", question)
        if cached_answer:
            logger.info("Returning cached answer from Redis.")
            print("\n[Answer from Redis Cache]:\n", cached_answer)
            return cached_answer

        # Step 7: Load LLM
        llm_model = LLMModel(model_name="gpt-4o-mini")

        # Step 8: Choose prompt type
        prompt_template = (
            SystemPrompt().get_oneliner_prompt()
            if self.is_oneliner_request(question)
            else SystemPrompt().get_paragraph_prompt()
        )

        # Step 9: Run RAG
        rag = RAGSystem(reranker=None, llm=llm_model, prompt_template=prompt_template)
        response = rag.ask_question(docs=reranked_docs, question=question)

        # Step 10: Store in Redis
        redis_handler.redis_hset("answers", question, response)
        logger.info("Answer stored in Redis hash for question.")

        print("\n[Final RAG Answer]:\n", response)
        return response
