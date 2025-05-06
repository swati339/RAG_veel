# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import logging
# import traceback
# import hashlib
# import json

# from rag_chat.models.pdf_processor import PDFProcessor
# from rag_chat.models.embeddings import EmbeddingModel
# from rag_chat.models.llm_model import LLMModel
# from rag_chat.models.reranking import ReRanker
# from rag_chat.models.rag_pipeline import RAGSystem
# from rag_chat.prompts.prompt_templates import SystemPrompt
# from rag_chat.configs.logging_config import setup_logging
# from rag_chat.redis_utils.redis_client import RedisSetup

# from langchain_community.vectorstores import Chroma

# # Initialize FastAPI app
# app = FastAPI()

# # Setup logging
# setup_logging()
# logger = logging.getLogger(__name__)

# # Request model
# class QuestionRequest(BaseModel):
#     question: str

# # Initialize Redis
# redis_store = RedisSetup()


# def get_redis_key(question: str, prompt_type: str) -> str:
#     """Generate a hash key for storing in Redis."""
#     question_hash = hashlib.sha256(question.strip().lower().encode()).hexdigest()
#     return f"rag_cache:{prompt_type}:{question_hash}"


# def run_rag_pipeline(question: str, prompt_type: str = "oneliner"):
#     PDF_PATH = "Docs/NIPS-2017-attention-is-all-you-need-Paper.pdf"
#     PERSIST_DIR = "chroma_db"

#     # Step 1: Load and split PDF
#     processor = PDFProcessor(file_path=PDF_PATH)
#     all_splits = processor.load_and_split()

#     # Step 2: Load embedder
#     embedder = EmbeddingModel()

#     # Step 3: Load or create vectorstore
#     vectorstore = Chroma(
#         persist_directory=PERSIST_DIR,
#         embedding_function=embedder.model
#     )
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#     retrieved_docs = retriever.invoke(question)

#     # Step 4: Re-rank
#     reranker = ReRanker(embedding_model=embedder)
#     reranked_docs = reranker.rerank_documents(query=question, docs=retrieved_docs)

#     # Step 5: Select prompt and LLM
#     llm_model = LLMModel(model_name="gpt-4o-mini")
#     if prompt_type == "oneliner":
#         prompt_template = SystemPrompt().get_oneliner_prompt()
#     else:
#         prompt_template = SystemPrompt().get_paragraph_prompt()

#     # Step 6: Run RAG system
#     rag = RAGSystem(reranker=None, llm=llm_model, prompt_template=prompt_template)
#     return rag.ask_question(docs=reranked_docs, question=question)


# @app.post("/oneliner")
# async def get_oneliner_answer(request: QuestionRequest):
#     return await get_answer_from_cache_or_rag(request.question, "oneliner")


# @app.post("/paragraph")
# async def get_paragraph_answer(request: QuestionRequest):
#     return await get_answer_from_cache_or_rag(request.question, "paragraph")


# async def get_answer_from_cache_or_rag(question: str, prompt_type: str):
#     try:
#         redis_key = get_redis_key(question, prompt_type)
#         cached_result = redis_store.redis_hget(redis_key, "answer")

#         if cached_result:
#             logger.info("Returning cached answer from Redis.")
#             return {"cached": True, "answer": cached_result}

#         logger.info("No cached result found. Running RAG pipeline...")
#         response = run_rag_pipeline(question=question, prompt_type=prompt_type)

#         # Save to Redis for future use
#         redis_store.redis_hset(redis_key, "answer", response)
#         return {"cached": False, "answer": response}

#     except Exception as e:
#         logger.error(f"{prompt_type} error: {str(e)}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Internal Server Error")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import traceback

from app.models.pdf_processor import PDFProcessor
from app.models.embeddings import EmbeddingModel
from app.models.llm_model import LLMModel
from app.models.reranking import ReRanker
from app.models.rag_pipeline import RAGSystem
from app.prompts.prompt_templates import SystemPrompt
from app.redis_client import RedisSetup
from app.configs.logging_config import setup_logging

from langchain_community.vectorstores import Chroma

app = FastAPI()
setup_logging()
logger = logging.getLogger(__name__)

redis_store = RedisSetup()

class QuestionRequest(BaseModel):
    question: str

def run_rag_pipeline(question: str, prompt_type: str = "oneliner"):
    PDF_PATH = "Docs/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    PERSIST_DIR = "chroma_db"

    processor = PDFProcessor(file_path=PDF_PATH)
    all_splits = processor.load_and_split()

    embedder = EmbeddingModel()

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedder.model
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)

    reranker = ReRanker(embedding_model=embedder)
    reranked_docs = reranker.rerank_documents(query=question, docs=retrieved_docs)

    llm_model = LLMModel(model_name="gpt-4o-mini")
    if prompt_type == "oneliner":
        prompt_template = SystemPrompt().get_oneliner_prompt()
    else:
        prompt_template = SystemPrompt().get_paragraph_prompt()

    rag = RAGSystem(reranker=None, llm=llm_model, prompt_template=prompt_template)
    return rag.ask_question(docs=reranked_docs, question=question)

@app.post("/oneliner")
async def get_oneliner_answer(request: QuestionRequest):
    try:
        cached = redis_store.redis_hget("oneliner_answers", request.question)
        if cached:
            return {"answer": cached}

        response = run_rag_pipeline(request.question, prompt_type="oneliner")
        redis_store.redis_hset("oneliner_answers", request.question, response)
        return {"answer": response}
    except Exception as e:
        logger.error("Oneliner error: %s", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/paragraph")
async def get_paragraph_answer(request: QuestionRequest):
    try:
        cached = redis_store.redis_hget("paragraph_answers", request.question)
        if cached:
            return {"answer": cached}

        response = run_rag_pipeline(request.question, prompt_type="paragraph")
        redis_store.redis_hset("paragraph_answers", request.question, response)
        return {"answer": response}
    except Exception as e:
        logger.error("Paragraph error: %s", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
