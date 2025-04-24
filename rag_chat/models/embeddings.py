from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging
from rag_chat.configs.logging_config import setup_logging

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)

    def embed_docs(self, docs):
        return Chroma.from_documents(documents=docs, embedding=self.embedding)

    def verify_vectors(self, docs):
        logging.info("Verifying vector embedding")
        v1 = self.embedding.embed_query(docs[0].page_content)
        v2 = self.embedding.embed_query(docs[1].page_content)

        logging.info(f"Vector 1: {v1[:5]}")
