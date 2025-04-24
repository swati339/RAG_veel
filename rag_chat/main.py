from models.pdf_processor import PDFProcessor
from models.embeddings import EmbeddingModel
from rag_chat.models.llm_model import LLMModel
from models.rag_pipeline import RAGSystem
import logging
from configs.logging_config import setup_logging

def main():
    # Load and split documents
    processor = PDFProcessor(file_path=r"docs/attention.pdf")
    all_splits = processor.load_and_split()

    # Generate embeddings and vector store
    embedder = EmbeddingModel()
    embedder.verify_vectors(all_splits)
    vectorstore = embedder.embed_docs(all_splits)

    # Load LLM
    llm_model = LLMModel()

    # Build RAG pipeline
    rag = RAGSystem(retriever=vectorstore.as_retriever(), llm=llm_model)
    rag_chain = rag.build_chain()

    # Ask a question
    rag.ask(rag_chain, "What is attention in python?")

if __name__ == "__main__":
    main()
