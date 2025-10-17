from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "devops_docs"
DOCUMENTS_PATH = "./documents/devops"

print("🔄 Ładowanie dokumentów...")
loader = DirectoryLoader(
    DOCUMENTS_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
print(f"✅ Załadowano {len(documents)} stron")

print("🔄 Dzielenie na chunki...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Utworzono {len(chunks)} chunków")

print("🔄 Tworzenie embeddingów...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

print("🔄 Indeksowanie w Qdrant...")
client = QdrantClient(url=QDRANT_URL)

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    force_recreate=True
)

print(f"✅ Zaindeksowano {len(chunks)} chunków w '{COLLECTION_NAME}'!")
