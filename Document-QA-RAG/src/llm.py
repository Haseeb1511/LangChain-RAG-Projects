from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


prompt = PromptTemplate(
    template="""You are a highly accurate assistant.
        Use ONLY the given context to answer the user's question.
        If the context does not contain the information needed, simply reply:
        "I don't know based on the given context."
        CONTEXT:
        {context}
        QUESTION:
        {question}
        Your Answer:""",
input_variables=["context", "question"])

parser = StrOutputParser()
model = ChatGroq(model ="gemma2-9b-it",max_tokens=300)


path = "../vector_store/chroma"
query=input("user: ")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=path,embedding_function=embedding)
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})


def context_text(docs):
    context_text ="\n\n".join(doc.page_content for doc in docs)
    return context_text


parallel_chain = RunnableParallel({
    "context":retriever | RunnableLambda(context_text),
    "question":RunnablePassthrough()

})

chain = parallel_chain | prompt | model | parser

response = chain.invoke(query)
print("AI:",response)


