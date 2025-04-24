from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import logging
from rag_chat.configs.logging_config import setup_logging

class LLMModel:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        pipe = pipeline("text-generation", model=model_name, max_new_tokens=512)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        logging.info("LLM model initialized")
