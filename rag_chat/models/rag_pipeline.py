from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging


class RAGSystem:
    def __init__(self, reranker, llm, prompt_template):
        self.reranker = reranker
        self.llm = llm
        self.prompt_template = (
            prompt_template  # <-- Accept prompt_template from main.py
        )

    def ask_question(self, docs, question):
        logging.info("Processing question: %s", question)

        # Step 1: If reranker is provided, use it to re-rank docs
        if self.reranker:
            logging.info("Re-ranking documents...")
            docs = self.reranker.rerank_documents(query=question, docs=docs)

        # Step 2: Create a QA chain using the prompt and the LLM
        logging.info("Creating QA chain with provided prompt...")
        question_answer_chain = create_stuff_documents_chain(
            self.llm.llm, self.prompt_template
        )

        # Step 3: Get response
        logging.info("Generating response...")
        response = question_answer_chain.invoke(
            {
                "input": question,
                "context": docs,  # Top re-ranked or retrieved docs
            }
        )

        logging.info("Generated response: %s", response)
        return response
