from models.pdf_processor import PDFProcessor
from models.embeddings import EmbeddingModel
from rag_chat.models.llm_model import LLMModel
from models.reranking import ReRanker 
from rag_chat.models.rag_pipeline import RAGSystem
from configs.logging_config import setup_logging
import logging

if __name__ == "__main__":
    setup_logging()  # Initialize logging

    # Load and split documents
    processor = PDFProcessor(file_path=r"docs/attention.pdf")
    all_splits = processor.load_and_split()

    # Generate embeddings
    embedder = EmbeddingModel()
    # This will generate embeddings for your documents
    document_embeddings = embedder.embed_texts(all_splits)

    # Load LLM model
    llm_model = LLMModel()

    # Initialize the ReRanker with your embedding model
    reranker = ReRanker(embedding_model=embedder)

    # Build RAG system with reranker (since you're not using a separate retriever)
    rag_system = RAGSystem(reranker=reranker, llm=llm_model)

    # Ask a question using the RAG system
    question = "What is attention in python?"
    response = rag_system.ask_question(docs=all_splits, question=question)

    # Print the result
    print(response)


