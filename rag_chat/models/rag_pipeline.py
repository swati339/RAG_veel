from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from rag_chat.configs.logging_config import setup_logging
from langchain_huggingface import HuggingFacePipeline
import logging

class RAGSystem:
    def __init__(self, reranker, llm):
        self.reranker = reranker
        self.llm = llm

    def create_prompt(self):
        system_prompt = (
            "You are an assistant for a question-answering task. Use the pieces of retrieved data to answer the asked questions. "
            "If you don't know the answer, just say you don't have anything related to it in your knowledge base. "
            "Answer the question in a maximum of 3 sentences. Do not go outside the context. Answer concisely.\n\n"
            "{context}"
        )
        return ChatPromptTemplate.from_messages([  # Prompt setup
            ("system", system_prompt),
            ("user", "{input}")
        ])

    def ask_question(self, docs, question):
        # Perform re-ranking based on the query
        logging.info("Re-ranking documents for question: %s", question)
        top_docs = self.reranker.rerank_documents(query=question, docs=docs)

        # Now use these re-ranked documents in the prompt
        prompt = self.create_prompt()
        question_answer_chain = create_stuff_documents_chain(self.llm.llm, prompt)
        
        logging.info("Generating response using LLM...")
        response = question_answer_chain.invoke({
            "input": question,
            "context": top_docs
        })

        logging.info("Response: %s", response)
        return response
