from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


def document_load(path):
    "This function load the Document"
    loader = DirectoryLoader(path=path,glob="*.pdf",loader_cls=PyMuPDFLoader) #It does not load PDFs in subdirectories unless you use a recursive loader or modify the glob pattern (e.g., "**/*.pdf" with recursive=True).
    return loader.load()

electrical = document_load("data/Electric")
ai = document_load("data/AI")

print("length of electric ",len(electrical))
print("length of AI ",len(ai))



def document_splitting(file):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=180)
    return splitter.split_documents(file)

chunks_electrical = document_splitting(electrical)
chunks_ai = document_splitting(ai)

print("length of ",len(chunks_electrical))
print("length of ",len(chunks_ai))


if __name__=="__main__":

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def vector_store_creation(chunks,path):
        vector_store = FAISS.from_documents(
            chunks,
            embedding_model
            )
        vector_store.save_local(path)
        return vector_store

    vector_store_electrical = vector_store_creation(chunks_electrical,"FAISS/electical/")
    vector_store_ai = vector_store_creation(chunks_ai,"FAISS/ai/")

    print("Vector store successfully created")