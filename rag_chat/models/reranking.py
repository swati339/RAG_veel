from sklearn.metrics.pairwise import cosine_similarity
import logging

class ReRanker:
    def __init__(self, embedding_model, top_k=3):
        self.embedding_model = embedding_model
        self.top_k = top_k

    def rerank_documents(self, query: str, docs: list):
        logging.info("Starting re-ranking process.")

        # Embed the query and documents
        query_embedding = self.embedding_model.embed_text(query)
        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = self.embedding_model.embed_texts(docs)

        # Compute cosine similarity between query and each document
        scores = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Sort documents by similarity score (highest first)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:self.top_k]]

        logging.info(f"Re-ranking complete. Top {self.top_k} documents selected.")
        return top_docs
