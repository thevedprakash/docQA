
import faiss
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

loader = PyPDFLoader("data/keph201.pdf")
pages = loader.load_and_split()
print(pages[0])

db = FAISS.from_documents(pages, HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: "))
db.save_local("faiss_index")

# sentence-transformers
# InstructorEmbedding
# faiss
# pypdf
# langchain
# llama-cpp-python
# streamlit
# huggingface_hub==0.20.1
# pydantic==1.10.11
# typing-inspect==0.8.0 
# typing_extensions==4.5.0