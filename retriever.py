from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

def document_retriever():
    loader = TextLoader("data/Health_knowledge_base.txt", encoding="utf-8")
    raw_text_extract = loader.load()[0].page_content

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "topic"), ("##","subtopic")])
    docs = text_splitter.split_text(raw_text_extract)

    final_docs = docs

    embeddings = OpenAIEmbeddings()
    vectorstore_database = FAISS.from_documents(final_docs, embeddings)

    retriever = vectorstore_database.as_retriever(search_kwargs={"k": 1})
    return retriever

if __name__ == "__main__":
    retriever = document_retriever()
    query = "treatment required for ADHD"
    results = retriever.get_relevant_documents(query)
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---\n{doc.page_content[:500]}...\n")  # ✅ Removed the extra parenthesis at the end
