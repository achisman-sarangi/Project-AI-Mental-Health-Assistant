
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from retriever import document_retriever

llm_retriever = document_retriever()

model = ChatOpenAI(model="gpt-4o", temperature=0.4)

template = """You are a helpful mental health assistant. Please answer the user's query based on the following context.
If you don't know the answer, simply say: \"I have no information about this topic.\"

Context:
{context}

User Query:
{question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]  
)

# Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=llm_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def get_response(user_query: str) -> str:
    result = qa_chain.invoke({"query": user_query})  # ✅ Use "question" key
    source_docs = result.get("source_documents", [])

    if not source_docs or all(doc.page_content.strip() == "" for doc in source_docs):
        return "I have no information about this topic."

    return result["result"]

# CLI test
if __name__ == "__main__":
    query = "who is srk"
    print("question:", query)
    print("Response:", get_response(query))
