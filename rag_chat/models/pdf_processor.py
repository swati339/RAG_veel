from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self, chunk_size=512, chunk_overlap=50):
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)
