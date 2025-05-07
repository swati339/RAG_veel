from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import traceback

from rag_chat.models.pdf_processor import PDFProcessor
from rag_chat.models.embeddings import EmbeddingModel
from rag_chat.models.llm_model import LLMModel
from rag_chat.models.reranking import ReRanker
from rag_chat.models.rag_pipeline import RAGSystem
from rag_chat.prompts.prompt_templates import SystemPrompt
from rag_chat.redis_utils.redis_client import RedisSetup
from rag_chat.configs.logging_config import setup_logging

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
