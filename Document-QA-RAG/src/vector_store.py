from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

path = "../PDF_BOOKS/"
def load_pdf(document):
    loader = DirectoryLoader(
        path=document,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()
pdf = load_pdf(document=path)

def split_document(document):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
    return splitter.split_documents(document)
chunks = split_document(document=pdf)
print("len of chiunks:",len(chunks))

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding=get_embedding_model()
path = "../vector_store/chroma"
vector_store = Chroma.from_documents(chunks,embedding=embedding,persist_directory=path)
vector_store.persist_directory = path 
retriever = vector_store.as_retriever()
print("Vector store successfuly created")

