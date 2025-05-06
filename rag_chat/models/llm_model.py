import os
from dotenv import load_dotenv
import logging
from langchain_openai import ChatOpenAI
from rag_chat.configs.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, model_name="gpt-4o-mini"):
        load_dotenv()  # Load from .env file
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0.7,
            max_tokens=512
        )

        logger.info(f"LLM model initialized with OpenAI model: {model_name}")
