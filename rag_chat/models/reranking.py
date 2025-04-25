from sklearn.metrics.pairwise import cosine_similarity
import logging


class ReRanker:
    def __init__(self, embedding_model, top_k=3):
        self.embedding_model = embedding_model
        self.top_k = top_k

    def rerank_documents(self, query: str, docs: list):
        logging.info("Starting re-ranking process...")

        # Step 1: Embed the query
        query_embedding = self.embedding_model.embed_text(query)

        # Step 2: Extract document texts and embed them
        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = self.embedding_model.model.embed_documents(doc_texts)

        # Step 3: Calculate cosine similarity between query and documents
        scores = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Step 4: Sort by similarity score
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[: self.top_k]]

        logging.info(f"Top {self.top_k} documents selected after re-ranking.")
        return top_docs
