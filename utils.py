# RAG and LLM implementation for analysis of docs 


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pypdf import PdfReader

#RAG Document loading / extraction 

def extract_pdf(file):
    reader = PdfReader(file)
    text = "" #str type
    for page in reader.pages:
        text += page.extract_text()
    return text

# text splitting or chunking of data plus overlapping for better context 
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 70 , chunk_overlap = 20)
    return splitter.split_text(text)

# embeding and vector storage 
def vector_creation(text):
    chunks = split_text(text)
    docs = [Document(page_content = c) for c in chunks]
    embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2") 
    vectorstore = FAISS.from_documents(docs, embedding)
    return vectorstore
    
    
    
