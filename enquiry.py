from langchain_community.document_loaders import TextLoader  # loads text documents
from langchain.text_splitter import CharacterTextSplitter  # splits text into chunks
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings as embd   # embeds text into vectors
from langchain_openai import OpenAIEmbeddings
# import joblib
import os
from langchain_community.vectorstores import FAISS   # vector database
from langchain.chains import RetrievalQAWithSourcesChain   # question answering chain
from dotenv import load_dotenv

load_dotenv()


# takes in a file and returns a list of chunks
def chunking(file):
    loader = TextLoader(file)
    data = loader.load()

    splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1600,
    chunk_overlap=100
    )

    chunks = splitter.split_documents(data)

    return chunks
    

# takes in a list of chunks and returns a vector database with chunks embedded as vectors
def embedder(chunks):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("openai_api_key"))
    vectordb =  FAISS.from_documents(chunks, embeddings)
    return vectordb


def generate_vectordb(file):
    chunks = chunking(file)
    vectordb = embedder(chunks)
    vectordb.save_local("faiss_index")


# takes in a vector database and a question and returns the answer
def query(vectordb, llm, question):
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectordb.as_retriever() )
    result = chain({'question': question}, return_only_outputs=True)
    return result['answer']

if __name__ == "__main__":
    file = "nitj.txt"
    generate_vectordb(file)