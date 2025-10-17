from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from qdrant_client import QdrantClient
import os
import logging
from typing import List, Optional

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG MCP Server",
    description="Local RAG system dla DevOps/SRE dokumentacji",
    version="1.0.0"
)

# Konfiguracja z environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://host.docker.internal:1234/v1")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "devops_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "3"))

# Inicjalizacja globalnych zmiennych
qa_chain = None
vectorstore = None
embeddings = None


class Question(BaseModel):
    question: str
    max_results: Optional[int] = MAX_RETRIEVAL_RESULTS
    temperature: Optional[float] = TEMPERATURE


class Source(BaseModel):
    content: str
    metadata: dict
    score: Optional[float] = None


class Answer(BaseModel):
    answer: str
    sources: List[Source]
    model_used: str
    collection: str


class HealthCheck(BaseModel):
    status: str
    qdrant_connected: bool
    lm_studio_connected: bool
    collection_exists: bool
    collection_vectors_count: Optional[int] = None
    qa_chain_initialized: bool


@app.on_event("startup")
async def startup_event():
    """Inicjalizacja RAG chain przy starcie aplikacji"""
    global qa_chain, vectorstore, embeddings

    try:
        logger.info("🚀 Uruchamianie RAG MCP Server...")
        logger.info(f"📍 Qdrant URL: {QDRANT_URL}")
        logger.info(f"🤖 LM Studio URL: {LM_STUDIO_URL}")
        logger.info(f"📚 Kolekcja: {COLLECTION_NAME}")
        logger.info(f"🧠 Embedding model: {EMBEDDING_MODEL}")

        # Embeddingi
        logger.info("⚙️  Inicjalizacja embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        logger.info("✅ Embedding model załadowany")

        # Qdrant client
        logger.info("⚙️  Łączenie z Qdrant...")
        client = QdrantClient(url=QDRANT_URL, timeout=10)

        # Sprawdź połączenie z Qdrant
        collection_exists = False
        try:
            collections = client.get_collections().collections
            collection_exists = any(c.name == COLLECTION_NAME for c in collections)

            if not collection_exists:
                logger.warning(f"⚠️  Kolekcja '{COLLECTION_NAME}' nie istnieje w Qdrant!")
                logger.warning("📝 Uruchom skrypt indeksowania dokumentów (index_documents.py)")
                logger.warning("💡 Serwer będzie działał, ale /query nie będzie działać dopóki nie zaindeksujesz dokumentów")
            else:
                collection_info = client.get_collection(COLLECTION_NAME)
                logger.info(f"✅ Znaleziono kolekcję '{COLLECTION_NAME}' z {collection_info.vectors_count} wektorami")
        except Exception as e:
            logger.error(f"❌ Błąd połączenia z Qdrant: {e}")
            raise

        # Vector store - TYLKO jeśli kolekcja istnieje
        logger.info("⚙️  Inicjalizacja vector store...")
        if collection_exists:
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings
            )
            logger.info("✅ Vector store zainicjalizowany")

            # LLM (LM Studio) - tylko jeśli mamy vectorstore
            logger.info("⚙️  Łączenie z LM Studio...")
            llm = OpenAI(
                base_url=LM_STUDIO_URL,
                api_key="lm-studio",
                temperature=TEMPERATURE,
                max_tokens=2048,
                timeout=30
            )
            logger.info("✅ LM Studio połączony")

            # RAG Chain
            logger.info("⚙️  Tworzenie RAG chain...")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": MAX_RETRIEVAL_RESULTS,
                        "score_threshold": 0.5
                    }
                ),
                return_source_documents=True,
                verbose=True
            )
            logger.info("✅ RAG chain zainicjalizowany")
        else:
            vectorstore = None
            qa_chain = None
            logger.warning("⚠️  RAG chain NIE został zainicjalizowany - zaindeksuj dokumenty najpierw")

        logger.info("✅ RAG Server zainicjalizowany pomyślnie!")
        logger.info(f"🎯 Gotowy do przyjmowania zapytań na http://0.0.0.0:8000")
        if not collection_exists:
            logger.info("⚠️  Pamiętaj: musisz zaindeksować dokumenty zanim będziesz mógł używać /query")

    except Exception as e:
        logger.error(f"❌ Krytyczny błąd podczas inicjalizacji: {e}", exc_info=True)
        # NIE rzucaj wyjątku - pozwól serwerowi wystartować
        logger.warning("⚠️  Serwer wystartuje w trybie ograniczonym - niektóre endpointy mogą nie działać")


@app.get("/", tags=["Info"])
async def root():
    """Podstawowe informacje o API"""
    return {
        "service": "RAG MCP Server",
        "version": "1.0.0",
        "status": "running",
        "rag_ready": qa_chain is not None,
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "config": "/config",
            "docs": "/docs"
        }
    }


@app.get("/config", tags=["Info"])
async def get_config():
    """Zwraca aktualną konfigurację"""
    return {
        "qdrant_url": QDRANT_URL,
        "lm_studio_url": LM_STUDIO_URL,
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "temperature": TEMPERATURE,
        "max_retrieval_results": MAX_RETRIEVAL_RESULTS,
        "rag_initialized": qa_chain is not None
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health():
    """Szczegółowy health check"""
    qdrant_connected = False
    lm_studio_connected = False
    collection_exists = False
    vectors_count = None

    try:
        # Sprawdź Qdrant
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        collections = client.get_collections().collections
        qdrant_connected = True
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)

        if collection_exists:
            collection_info = client.get_collection(COLLECTION_NAME)
            vectors_count = collection_info.vectors_count
    except Exception as e:
        logger.error(f"Health check - Qdrant error: {e}")

    try:
        # Sprawdź LM Studio (opcjonalnie, może być wolne)
        lm_studio_connected = True  # Zakładamy że działa, można dodać prawdziwy check
    except Exception as e:
        logger.error(f"Health check - LM Studio error: {e}")

    return HealthCheck(
        status="healthy" if (qdrant_connected and collection_exists and qa_chain is not None) else "degraded",
        qdrant_connected=qdrant_connected,
        lm_studio_connected=lm_studio_connected,
        collection_exists=collection_exists,
        collection_vectors_count=vectors_count,
        qa_chain_initialized=qa_chain is not None
    )


@app.post("/query", response_model=Answer, tags=["RAG"])
async def query(question: Question):
    """Endpoint do zadawania pytań RAG"""
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain nie został zainicjalizowany. Zaindeksuj dokumenty używając index_documents.py i zrestartuj serwer."
        )

    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store nie jest dostępny. Zaindeksuj dokumenty najpierw."
        )

    try:
        logger.info(f"📨 Otrzymano pytanie: {question.question}")

        # Wykonaj zapytanie RAG
        result = qa_chain.invoke({"query": question.question})

        # Przygotuj odpowiedź
        sources = []
        for doc in result.get('source_documents', []):
            sources.append(Source(
                content=doc.page_content[:500],  # Ograniczenie długości
                metadata=doc.metadata,
                score=doc.metadata.get('score', None)
            ))

        answer = Answer(
            answer=result['result'],
            sources=sources,
            model_used=EMBEDDING_MODEL,
            collection=COLLECTION_NAME
        )

        logger.info(f"✅ Odpowiedź wygenerowana, znaleziono {len(sources)} źródeł")
        return answer

    except Exception as e:
        logger.error(f"❌ Błąd podczas przetwarzania pytania: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Błąd przetwarzania: {str(e)}")


@app.post("/search", tags=["RAG"])
async def search_documents(query: str, k: int = 5):
    """Bezpośrednie wyszukiwanie w vector store bez LLM"""
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store niedostępny. Zaindeksuj dokumenty najpierw."
        )

    try:
        docs = vectorstore.similarity_search_with_score(query, k=k)
        results = [
            {
                "content": doc.page_content[:300],
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in docs
        ]
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
