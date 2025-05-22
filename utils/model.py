from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever,EnsembleRetriever
from IPython.display import display,Markdown
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
from vector_store import document_load,document_splitting
load_dotenv()


electrical = document_load("data/Electric")
ai = document_load("data/AI")

chunks_electrical = document_splitting(electrical)
chunks_ai = document_splitting(ai)

callback = [StreamingStdOutCallbackHandler()]
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
rerank_model = "rerank-english-v3.0"
model = ChatGroq(model="deepseek-r1-distill-llama-70b",streaming = True,callbacks=callback)


vector_store_electrical = FAISS.load_local(embeddings=embedding_model,allow_dangerous_deserialization=True,folder_path="FAISS/electical/") 
electrical_retriever = vector_store_electrical.as_retriever(search_type="mmr",search_kwargs={"k":4})

vector_store_ai = FAISS.load_local(embeddings=embedding_model,allow_dangerous_deserialization=True,folder_path = "FAISS/ai/") 
ai_retriever = vector_store_ai.as_retriever(search_type="similarity",search_kwargs={"k":5})

bm25_retriever_electrical = BM25Retriever.from_documents(chunks_electrical)
bm25_retriever_electrical.k=5
bm25_retriever_ai = BM25Retriever.from_documents(chunks_ai)
bm25_retriever_ai.k=5

hybrid_retriever_electrical = EnsembleRetriever(
    retrievers=[electrical_retriever,bm25_retriever_electrical],
    weight=[0.7,0.3]
)
hybrid_retriever_ai = EnsembleRetriever(
    retrievers=[ai_retriever,bm25_retriever_ai],
    weight=[0.65,0.35]
)

prompt = PromptTemplate(
    template="""
You are an AI interview assistant. Answer the question strictly based on the provided context anser the quesion in points.
- If the context does not contain enough information to answer the question, respond with: "Context not enough."
- Do not use any external knowledge.
CONTEXT:
{context}
QUESTION:
{question}
ANSWER:
""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

def cleaning(documents):
    return "\n\n".join(doc.page_content for doc in documents)


rerank = CohereRerank(model=rerank_model)

final_retriever = ContextualCompressionRetriever(
    base_compressor=rerank,
    base_retriever=hybrid_retriever_ai
    )


parallel_chain = RunnableParallel({
    "context": final_retriever | RunnableLambda(cleaning),
    "question":RunnablePassthrough()
})

chain = parallel_chain | prompt | model | parser

query = input("Enter query here:")

result = chain.invoke(query)
display(Markdown(result))

