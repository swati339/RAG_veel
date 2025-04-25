from langchain_core.prompts import ChatPromptTemplate
class SystemPrompt:
    def get_oneliner_prompt(self):
        system_prompt = (
            "Answer in a single concise sentence using the retrieved information. "
            "If unknown, say you don't have an answer.\n\n{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])

    def get_paragraph_prompt(self):
        system_prompt = (
            "You are an assistant for a question-answering task. Use the pieces of retrieved data to answer the asked questions. "
            "If you don't know the answer, just say you don't have anything related to it in your knowledge base. "
            "Answer the question in a maximum of 3 sentences. Do not go outside the context. Answer concisely.\n\n"
            "{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
