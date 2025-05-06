from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self, chunk_size=512, chunk_overlap=50):
        loader = PyPDFLoader(self.file_path)
        
        # Load and sort by page number
        docs = sorted(loader.load(), key=lambda x: x.metadata.get("page", 0))

        # Clean up page content
        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", " ").strip()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        print(f"[PDFProcessor] Loaded {len(chunks)} chunks.")
        print(f"[Sample Chunk Preview]:\n{chunks[0].page_content[:500]}...\n")

        return chunks
