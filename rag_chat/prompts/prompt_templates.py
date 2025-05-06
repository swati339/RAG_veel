from langchain_core.prompts import ChatPromptTemplate

class SystemPrompt:
    def get_oneliner_prompt(self):
        system_prompt = (
            "Answer in a single concise sentence using the retrieved information. "
            "Donot give irrelevant informations."
            "If unknown, say you don't have an answer.\n\n{context}"
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])

    def get_paragraph_prompt(self):
        system_prompt = (
            "You are an assistant for a question-answering task. Use the pieces of retrieved data to answer the asked questions. "
            "If you don't know the answer, just say you don't have anything related to it in your knowledge base.\n\n"
            "Return the response in the following structured JSON format:\n"
            "{{\n"
            "  \"question\": \"<original question>\",\n"
            "  \"answer\": \"<concise paragraph (max 3 sentences) strictly based on context>\",\n"
            "  \"source_summary\": \"<brief summary of relevant context or section>\"\n"
            "}}\n\n"
            "Only return valid JSON.\n\nContext:\n{context}"
        )
        return ChatPromptTemplate.from_messages([   
            ("system", system_prompt),
            ("human", "{input}")
        ])
