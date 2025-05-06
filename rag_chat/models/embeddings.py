from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

class EmbeddingModel:
    def __init__(self):
        load_dotenv()  # Load the environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        
        self.model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

    def embed_text(self, text: str):
        # Embed a single text query
        return self.model.embed_query(text)

    def embed_texts(self, docs):
        # Extract plain text from Document objects and embed them
        texts = [doc.page_content for doc in docs]
        return self.model.embed_documents(texts)
