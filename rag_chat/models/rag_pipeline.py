from langchain.chains.combine_documents import create_stuff_documents_chain
from rag_chat.configs.logging_config import setup_logging
import logging
import json

setup_logging()
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, reranker, llm, prompt_template):
        self.reranker = reranker
        self.llm = llm
        self.prompt_template = prompt_template

    def ask_question(self, docs, question):
        logger.info("Processing question: %s", question)

        # Step 1: Re-rank documents if reranker is available
        if self.reranker:
            logger.info("Re-ranking documents...")
            docs = self.reranker.rerank_documents(query=question, docs=docs)

        # Step 2: Create the QA chain using the given LLM and prompt
        logger.info("Creating QA chain with provided prompt...")

        # Handle both tuple (paragraph prompt) and single (oneliner prompt)
        prompt = self.prompt_template[0] if isinstance(self.prompt_template, tuple) else self.prompt_template

        question_answer_chain = create_stuff_documents_chain(
            self.llm.llm, prompt
        )

        # Step 3: Generate the response
        logger.info("Generating response...")
        response = question_answer_chain.invoke({
            "input": question,
            "context": docs,
        })

        # Step 4: Handle JSON only for paragraph-style prompt
        if isinstance(self.prompt_template, tuple):  # paragraph prompt returns a tuple
            try:
                parsed_response = json.loads(response)
                logger.info("Successfully parsed structured JSON response.")
                return parsed_response
            except json.JSONDecodeError:
                logger.warning("Failed to parse structured output as JSON. Returning raw response.")
                return {"raw_response": response}

        # For one-liner prompt, return as plain text
        return {"answer": response}

