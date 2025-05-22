import streamlit as st
from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever,EnsembleRetriever
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
import re
load_dotenv()

st.set_page_config(page_title="😍Interview Mentor😘",layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
rerank_model = "rerank-english-v3.0"
model = ChatGroq(model="deepseek-r1-distill-llama-70b")


@st.cache_resource(show_spinner="Document loading..")
def document_load(path):
    "This function load the Document"
    loader = DirectoryLoader(path=path,glob="*.pdf",loader_cls=PyMuPDFLoader) #It does not load PDFs in subdirectories unless you use a recursive loader or modify the glob pattern (e.g., "**/*.pdf" with recursive=True).
    return loader.load()
@st.cache_resource(show_spinner="chunking the document...")
def document_splitting(_file):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=180)
    return splitter.split_documents(_file)


electrical = document_load("data/Electric")
ai = document_load("data/AI")
chunks_electrical = document_splitting(electrical)
chunks_ai = document_splitting(ai)

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
            weight=[0.7,0.3])
hybrid_retriever_ai = EnsembleRetriever(
            retrievers=[ai_retriever,bm25_retriever_ai],
            weight=[0.65,0.35])
        

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
                    input_variables=["context", "question"])
parser = StrOutputParser()

def cleaning(documents):
            return "\n\n".join(doc.page_content for doc in documents)
rerank = CohereRerank(model=rerank_model)


def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


query = st.chat_input("enter")


def stream_output(query, chain):
    raw_stream = chain.stream(query)
    output_text = ""
    
    # Display AI message in streaming block
    with st.chat_message("AI"):
        response_container = st.empty()  # st.empty() in Streamlit creates a placeholder container in your app where you can later insert or update content dynamically
        for chunk in raw_stream:
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            output_text += text
            response_container.markdown(output_text + "▌")  # show streaming cursor
        final_output = remove_think_tags(output_text.strip())
        response_container.markdown(final_output)
    return final_output


def main(query,hybrid_retriever):
    final_retriever = ContextualCompressionRetriever(
    base_compressor=rerank,
    base_retriever=hybrid_retriever)

    parallel_chain = RunnableParallel({
            "context": final_retriever | RunnableLambda(cleaning),
            "question":RunnablePassthrough()})                                        
    chain = parallel_chain | prompt | model 
            
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user","content":query})

    response = stream_output(query, chain)
    st.session_state.messages.append({"role": "AI", "content": response})


choice = st.sidebar.selectbox("Chose the Field:",["Electrical Engineering(EE)","Artificall Intelligence(AI)"])

if choice == "Electrical Engineering(EE)":
     st.markdown("### ⚡ Here you can ask Interview Question about Electrical Engineering🔌")
     
elif choice=="Artificall Intelligence(AI)":
    st.markdown("###🤖 Here you can ask Interview Question about Artificall Intelligence💻 ")


if query:
    if choice == "Electrical Engineering(EE)":
        hybrid_retriever = hybrid_retriever_electrical
        
    elif choice=="Artificall Intelligence(AI)":
         hybrid_retriever = hybrid_retriever_ai

    main(query,hybrid_retriever)





# [hasattr(object, attribute_name)]  it check weather the object has the specific attribute

# class Dog:
#     def __init__(self):
#         self.name = "Buddy"
# dog = Dog()

# print(hasattr(dog, "name"))     # True
# print(hasattr(dog, "age"))      # False
