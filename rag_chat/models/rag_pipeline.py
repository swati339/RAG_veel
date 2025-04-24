from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class RAGSystem:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def create_prompt(self):
        system_prompt = (
            "You are an assistant for question-answering task. Use the pieces of retrieved data to answer the asked questions. "
            "If you don't know the answer, just say you don't have anything related to it in your knowledge base. Answer the question in maximum 3 sentences. "
            "Donot go outside the context."
            "Answer concise."

             
             "{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("User", "{input}")
        ])

    def build_chain(self):
        prompt = self.create_prompt()
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(self.retriever, question_answer_chain)

    def ask_qestion(self, chain, question):
        response = chain.invoke({"input": question})
        print(response["answer"])
