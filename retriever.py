from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from pathlib import Path
import os

def document_retriever():
    # ✅ Build safe path to knowledge base
    base_dir = Path(__file__).parent
    file_path = base_dir / "data" / "Health_knowledge_base.txt"

    # 🧪 Debug print
    print("📂 Current Working Directory:", os.getcwd())
    print("📄 Resolved Path:", file_path)
    print("✅ File exists?", file_path.exists())

    # ✅ Load the file using the resolved absolute path
    loader = TextLoader(str(file_path), encoding="utf-8")
    raw_text_extract = loader.load()[0].page_content

    # ✅ Split the document by markdown headers
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "topic"), ("##", "subtopic")]
    )
    docs = text_splitter.split_text(raw_text_extract)

    # ✅ Embed and build vector store
    embeddings = OpenAIEmbeddings()
    vectorstore_database = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore_database.as_retriever(search_kwargs={"k": 1})

    return retriever

# ✅ Test query if run standalone
if __name__ == "__main__":
    retriever = document_retriever()
    query = "treatment required for ADHD"
    results = retriever.get_relevant_documents(query)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---\n{doc.page_content[:500]}...\n")
