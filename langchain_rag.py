import getpass
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
import requests
from dotenv import load_dotenv

#load environment variables
load_dotenv()

# Access the Langsmith API key
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")

# Print to confirm (optional, for debugging)
print(f"Langsmith API Key: {langsmith_api_key}")
print(f"Langchain Tracing V2: {langchain_tracing_v2}")

# Configure the local Ollama Mistral model
OLLAMA_URL = "http://localhost:11434/api/v1/generate"  # Default local Ollama API endpoint

# Define the local Ollama LLM class
class OllamaLLM:
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def invoke(self, prompt: str):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature
        }
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            raise RuntimeError(f"Failed to query Ollama API: {response.status_code} {response.text}")

# Instantiate the local Mistral model
llm = OllamaLLM(model="mistral", temperature=0)


#Defining the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#Define vector stores
vector_store = Chroma(embedding_function=embeddings)

#Simple indexing pipeline that answers questions about the website content
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

#Load and chunk the contents of the blog
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

print(docs[0].page_content[:500])

#Text splitter to chunk the content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

#Index the chunks and add to the vector store
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

#Define the prompt for question - answering
prompt = hub.pull("rlm/rag-prompt")


example_messages = prompt.invoke(
    {"context": "Who is Lillian Weng?", "question": "Who is Lillian Weng?"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)