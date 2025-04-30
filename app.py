import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader,UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()


st.title("Documnet Q&A")

#   file_type = 
upload_file = st.file_uploader("Upload a PDF or Word file", type=["pdf", "docx"])
@st.cache_resource(show_spinner="loading_pdf")
def load_document(uploaded_file):
    file_type = os.path.splitext(uploaded_file.name)[-1].lower()
    temp_file_path=f"temp file{file_type}"

    with open(temp_file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    if file_type==".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_type ==".docx":
        loader = UnstructuredWordDocumentLoader(temp_file_path)
    else:
        raise ValueError("Unsupported Document type")
    return loader.load()


@st.cache_resource(show_spinner="Loading Embedding model.....")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Creating Vector store")
def split_document(_document):  # If the parameter name starts with an underscore (_), Streamlit will ignore it during caching. we dont want to cache chunks as they are list so we use _
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
    chunks = splitter.split_documents(_document)
    embedding=get_embedding_model()
    vector_store = Chroma.from_documents(chunks,embedding=embedding)
    retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})
    return retriever



if upload_file:
    pdf = load_document(upload_file)
    if pdf:
        retriever = split_document(pdf)
        prompt = PromptTemplate(
            template="""You are a highly accurate assistant.
                Use ONLY the given context to answer the user's question.
                If the context does not contain the information needed, simply reply:
                "I don't know based on the given context."
                CONTEXT:
                {context}
                QUESTION:
                {question}
                Your Answer:
                Your Answer (with citations like [1], [2]):
                """,
        input_variables=["context", "question"])
        parser = StrOutputParser()
        model = ChatGroq(model ="gemma2-9b-it",max_tokens=300)

        def context_text(docs):
            context = []
            for i ,doc in enumerate(docs,1):
                metadata = doc.metadata.get("page","unknown")
                page_content = doc.page_content
                context.append(f"[{i}]{page_content}\n(page:{metadata})")
            return "\n\n".join(context)


        parallel_chain = RunnableParallel({
            "context":retriever | RunnableLambda(context_text),
            "question":RunnablePassthrough()

        })

        chain = parallel_chain | prompt | model | parser

#----------------------------Chat casching-------------------------------------------------------------------------
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])
#-----------------------------------------------------------------------------------------------------------------  
        query = st.chat_input()
        if query:
            st.chat_message("user:").markdown(query)
            st.session_state.messages.append({"role":"user","content":query})

            result = chain.invoke(query)
            response = result
            with st.chat_message("AI"):
                with st.spinner("Generating Response"):
                    st.markdown(response)
                    st.session_state.messages.append({"role":"AI","content":response})
else:
    st.info("Please upload a PDF file to start.")



