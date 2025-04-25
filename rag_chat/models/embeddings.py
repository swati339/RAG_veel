from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

class EmbeddingModel:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed_text(self, text: str):
        # Embed a single text query
        return self.model.embed_query(text)

    def embed_texts(self, docs):
        # Extract plain text from Document objects and embed them
        texts = [doc.page_content for doc in docs]  # Extract text content from Document
        return self.model.embed_documents(texts)  # Return document embeddings

