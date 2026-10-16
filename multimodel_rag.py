"""Persistent multimodal RAG using Qdrant + SQLite.

Pipeline preserved: Unstructured extraction -> text/table/image summaries ->
MultiVectorRetriever -> history-aware retrieval -> multimodal prompt -> LLM.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv
from PIL import Image
from unstructured.partition.pdf import partition_pdf

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.chains import create_history_aware_retriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

load_dotenv()

BASE_STORAGE_DIR = Path(os.getenv("RAG_STORAGE_DIR", "./rag_storage"))
PDF_STORAGE_DIR = BASE_STORAGE_DIR / "pdfs"
IMAGE_STORAGE_DIR = BASE_STORAGE_DIR / "images"
SQLITE_PATH = Path(os.getenv("SQLITE_DB_PATH", str(BASE_STORAGE_DIR / "rag_app.db")))
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(BASE_STORAGE_DIR / "qdrant_local"))
QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "multimodal_rag").strip() or "multimodal_rag"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

for directory in (BASE_STORAGE_DIR, PDF_STORAGE_DIR, IMAGE_STORAGE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS pdfs (
            pdf_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS documents (
            parent_id TEXT PRIMARY KEY,
            pdf_id TEXT NOT NULL REFERENCES pdfs(pdf_id) ON DELETE CASCADE,
            content_type TEXT NOT NULL,
            page_number TEXT,
            content_text TEXT,
            image_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS summaries (
            summary_id TEXT PRIMARY KEY,
            pdf_id TEXT NOT NULL REFERENCES pdfs(pdf_id) ON DELETE CASCADE,
            parent_id TEXT NOT NULL REFERENCES documents(parent_id) ON DELETE CASCADE,
            summary_type TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            qdrant_point_id TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            pdf_id TEXT NOT NULL UNIQUE REFERENCES pdfs(pdf_id) ON DELETE CASCADE,
            title TEXT NOT NULL DEFAULT 'PDF conversation',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            sequence_no INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content TEXT NOT NULL,
            images_json TEXT NOT NULL DEFAULT '[]',
            sources_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(conversation_id, sequence_no)
        );
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            job_id TEXT PRIMARY KEY,
            pdf_id TEXT NOT NULL REFERENCES pdfs(pdf_id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            started_at TEXT,
            finished_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_documents_pdf ON documents(pdf_id);
        CREATE INDEX IF NOT EXISTS idx_summaries_pdf ON summaries(pdf_id);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, sequence_no);
        """)

init_db()


class SourceString(str):
    """String content carrying source metadata through parent retrieval."""
    def __new__(cls, value: str, source_metadata: Optional[dict] = None):
        obj = str.__new__(cls, value)
        obj.source_metadata = source_metadata or {}
        return obj


class SQLiteDocStore(BaseStore[str, str]):
    """SQLite-backed LangChain BaseStore for persistent parent-document content."""
    def __init__(self, pdf_id: str):
        self.pdf_id = pdf_id

    def mget(self, keys: list[str]) -> list[Optional[str]]:
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        with db_connect() as conn:
            rows = conn.execute(
                f"SELECT parent_id, content_text, image_path, content_type, page_number, metadata_json "
                f"FROM documents WHERE pdf_id=? AND parent_id IN ({placeholders})",
                [self.pdf_id, *keys],
            ).fetchall()
        by_id = {row["parent_id"]: row for row in rows}
        result = []
        for key in keys:
            row = by_id.get(key)
            if row is None:
                result.append(None)
                continue
            content = row["content_text"] or ""
            if row["image_path"]:
                try:
                    content = base64.b64encode(Path(row["image_path"]).read_bytes()).decode("utf-8")
                except OSError:
                    content = ""
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata.setdefault("type", row["content_type"])
            metadata.setdefault("page_number", row["page_number"])
            result.append(SourceString(content, metadata))
        return result

    def mset(self, key_value_pairs: list[tuple[str, str]]) -> None:
        # Documents are written with their metadata in persist_parent_document().
        # This method exists for the expected MultiVectorRetriever store interface.
        with db_connect() as conn:
            for key, value in key_value_pairs:
                conn.execute(
                    "INSERT OR REPLACE INTO documents(parent_id,pdf_id,content_type,content_text,metadata_json) "
                    "VALUES(?,?,?,?,?)",
                    (key, self.pdf_id, "Text", str(value), "{}"),
                )

    def mdelete(self, keys: list[str]) -> None:
        if not keys:
            return
        with db_connect() as conn:
            conn.executemany("DELETE FROM documents WHERE parent_id=? AND pdf_id=?", [(k, self.pdf_id) for k in keys])

    def yield_keys(self, prefix: Optional[str] = None) -> Iterable[str]:
        with db_connect() as conn:
            rows = conn.execute("SELECT parent_id FROM documents WHERE pdf_id=? ORDER BY parent_id", (self.pdf_id,)).fetchall()
        for row in rows:
            key = row["parent_id"]
            if prefix is None or key.startswith(prefix):
                yield key


def calculate_pdf_id(pdf_path: str) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_name_for_pdf(pdf_id: str) -> str:
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", QDRANT_COLLECTION_PREFIX)[:32]
    # Per-PDF collections enforce PDF isolation without changing retrieval logic.
    return f"{safe_prefix}_{pdf_id[:32]}"


def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        if not QDRANT_API_KEY:
            raise RuntimeError("QDRANT_URL is configured but QDRANT_API_KEY is empty. Set both in .env for Qdrant Cloud.")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    return QdrantClient(path=QDRANT_LOCAL_PATH)


def get_pdf_storage_paths(pdf_id: str):
    return PDF_STORAGE_DIR / f"{pdf_id}.pdf", collection_name_for_pdf(pdf_id)


def rag_exists(pdf_id: str) -> bool:
    with db_connect() as conn:
        row = conn.execute("SELECT status, stored_path, collection_name FROM pdfs WHERE pdf_id=?", (pdf_id,)).fetchone()
    if not row or row["status"] != "completed" or not Path(row["stored_path"]).exists():
        return False
    client = get_qdrant_client()
    try:
        return client.collection_exists(row["collection_name"])
    finally:
        client.close()


def get_available_pdfs() -> list[dict]:
    with db_connect() as conn:
        rows = conn.execute("SELECT pdf_id, filename, status FROM pdfs WHERE status='completed' ORDER BY filename COLLATE NOCASE").fetchall()
    return [dict(row) for row in rows]


def register_pdf(pdf_id: str, filename: str, stored_path: Path, status: str = "processing") -> None:
    collection = collection_name_for_pdf(pdf_id)
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO pdfs(pdf_id,content_hash,filename,stored_path,collection_name,status) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(pdf_id) DO UPDATE SET filename=excluded.filename, stored_path=excluded.stored_path, "
            "collection_name=excluded.collection_name, status=excluded.status, error=NULL, updated_at=CURRENT_TIMESTAMP",
            (pdf_id, pdf_id, filename, str(stored_path), collection, status),
        )
        conn.execute("INSERT OR IGNORE INTO conversations(conversation_id,pdf_id,title) VALUES(?,?,?)", (str(uuid.uuid4()), pdf_id, filename))


def get_pdf_record(pdf_id: str):
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM pdfs WHERE pdf_id=?", (pdf_id,)).fetchone()
    return dict(row) if row else None


def get_conversation_id(pdf_id: str) -> Optional[str]:
    with db_connect() as conn:
        row = conn.execute("SELECT conversation_id FROM conversations WHERE pdf_id=?", (pdf_id,)).fetchone()
    return row[0] if row else None


def save_message(pdf_id: str, role: str, content: str, images: Optional[list] = None, sources: Optional[list] = None) -> None:
    with db_connect() as conn:
        conv = conn.execute("SELECT conversation_id FROM conversations WHERE pdf_id=?", (pdf_id,)).fetchone()
        if not conv:
            conversation_id = str(uuid.uuid4())
            conn.execute("INSERT INTO conversations(conversation_id,pdf_id,title) VALUES(?,?,?)", (conversation_id, pdf_id, "PDF conversation"))
        else:
            conversation_id = conv[0]
        seq = conn.execute("SELECT COALESCE(MAX(sequence_no),0)+1 FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()[0]
        conn.execute(
            "INSERT INTO messages(message_id,conversation_id,sequence_no,role,content,images_json,sources_json) VALUES(?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), conversation_id, seq, role, content, json.dumps(images or []), json.dumps(sources or [])),
        )
        conn.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=?", (conversation_id,))


def load_messages(pdf_id: str) -> list[dict]:
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT m.role,m.content,m.images_json,m.sources_json FROM messages m "
            "JOIN conversations c ON c.conversation_id=m.conversation_id WHERE c.pdf_id=? ORDER BY m.sequence_no",
            (pdf_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"], "images": json.loads(r["images_json"] or "[]"), "sources": json.loads(r["sources_json"] or "[]")} for r in rows]


def load_chat_history(pdf_id: str):
    from langchain_core.messages import HumanMessage, AIMessage
    messages = load_messages(pdf_id)
    result = []
    for msg in messages:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result


def clear_chat(pdf_id: str) -> None:
    with db_connect() as conn:
        conv = conn.execute("SELECT conversation_id FROM conversations WHERE pdf_id=?", (pdf_id,)).fetchone()
        if conv:
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv[0],))


def delete_pdf(pdf_id: str) -> None:
    record = get_pdf_record(pdf_id)
    if record:
        client = get_qdrant_client()
        try:
            if client.collection_exists(record["collection_name"]):
                client.delete_collection(record["collection_name"])
        finally:
            client.close()
        pdf_path = Path(record["stored_path"])
        if pdf_path.exists():
            pdf_path.unlink()
        with db_connect() as conn:
            image_rows = conn.execute("SELECT image_path FROM documents WHERE pdf_id=? AND image_path IS NOT NULL", (pdf_id,)).fetchall()
            conn.execute("DELETE FROM pdfs WHERE pdf_id=?", (pdf_id,))
        for row in image_rows:
            try:
                Path(row["image_path"]).unlink(missing_ok=True)
            except OSError:
                pass


def normalize_image_b64(image_b64: str) -> Optional[str]:
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return None


def get_page_number(metadata):
    if metadata is None:
        return None
    page = getattr(metadata, "page_number", None)
    if page is not None:
        return page
    try:
        return metadata.to_dict().get("page_number")
    except Exception:
        return None


def format_page_number(page_number):
    if page_number is None:
        return None
    if isinstance(page_number, (list, tuple, set)):
        return ", ".join(str(x) for x in page_number)
    return str(page_number)


def _store_parent(pdf_id: str, parent_id: str, value: str, source_metadata: dict) -> None:
    content_type = source_metadata.get("type", "Text")
    page = format_page_number(source_metadata.get("page_number"))
    image_path = None
    content_text = str(value)
    if content_type == "Image":
        image_path = IMAGE_STORAGE_DIR / f"{parent_id}.png"
        image_path.write_bytes(base64.b64decode(content_text))
        content_text = None
    with db_connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents(parent_id,pdf_id,content_type,page_number,content_text,image_path,metadata_json) VALUES(?,?,?,?,?,?,?)",
            (parent_id, pdf_id, content_type, page, content_text, str(image_path) if image_path else None, json.dumps(source_metadata)),
        )


def _upsert_summary_rows(pdf_id: str, records: list[dict]) -> None:
    with db_connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO summaries(summary_id,pdf_id,parent_id,summary_type,summary_text,qdrant_point_id) VALUES(?,?,?,?,?,?)",
            [(r["summary_id"], pdf_id, r["parent_id"], r["summary_type"], r["summary_text"], r["qdrant_point_id"]) for r in records],
        )


def _set_job_status(pdf_id: str, status: str, error: Optional[str] = None, job_id: Optional[str] = None):
    job_id = job_id or str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute("INSERT INTO ingestion_jobs(job_id,pdf_id,status,attempts,error,started_at,finished_at) VALUES(?,?,?,?,?,?,?)",
                     (job_id, pdf_id, status, 1, error, "CURRENT_TIMESTAMP" if status == "processing" else None, None))
        conn.execute("UPDATE pdfs SET status=?,error=?,updated_at=CURRENT_TIMESTAMP WHERE pdf_id=?", (status, error, pdf_id))


def build_multimodal_rag(pdf_path: str, pdf_id: Optional[str] = None, filename: Optional[str] = None):
    pdf_id = pdf_id or calculate_pdf_id(pdf_path)
    persistent_pdf_path, collection_name = get_pdf_storage_paths(pdf_id)
    if not persistent_pdf_path.exists():
        persistent_pdf_path.write_bytes(Path(pdf_path).read_bytes())
    register_pdf(pdf_id, filename or Path(pdf_path).name, persistent_pdf_path, "processing")
    job_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute("INSERT INTO ingestion_jobs(job_id,pdf_id,status,attempts) VALUES(?,?,?,?)", (job_id, pdf_id, "processing", 1))

    try:
        chunks = partition_pdf(
            filename=str(persistent_pdf_path), strategy="hi_res", infer_table_structure=True,
            extract_image_block_types=["Image"], extract_image_block_to_payload=True,
            chunking_strategy="by_title", max_characters=10000,
            combine_text_under_n_chars=2000, new_after_n_chars=6000,
        )
        texts, tables, images = [], [], []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                for element in getattr(chunk.metadata, "orig_elements", []) or []:
                    if "Image" in str(type(element)):
                        images.append({"base64": getattr(element.metadata, "image_base64", None), "page_number": get_page_number(element.metadata)})
            if "Table" in str(type(chunk)):
                tables.append(chunk)

        summarizer_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following content concisely. The content can be normal text or a table. "
            "For tables, identify what the table represents, preserve important column names and row values, "
            "and mention important relationships or comparisons.\n\nContent:\n{element}"
        )
        summarize_chain = ({"element": lambda x: str(x)} | summary_prompt | summarizer_llm | StrOutputParser())
        text_summaries = summarize_chain.batch(texts) if texts else []

        tables_html, table_metadata = [], []
        for table in tables:
            html = getattr(table.metadata, "text_as_html", None)
            if html:
                page = get_page_number(table.metadata)
                tables_html.append(SourceString(html, {"type": "Table", "page_number": page}))
                table_metadata.append({"type": "Table", "page_number": page})
        table_summaries = summarize_chain.batch(tables_html) if tables_html else []

        normalized_images, image_metadata = [], []
        for item in images:
            normalized = normalize_image_b64(item.get("base64"))
            if normalized:
                normalized_images.append(SourceString(normalized, {"type": "Image", "page_number": item.get("page_number")}))
                image_metadata.append({"type": "Image", "page_number": item.get("page_number")})

        image_prompt = ChatPromptTemplate.from_messages([("user", [
            {"type": "text", "text": "Describe this image in detail for retrieval. Include important objects, diagrams, labels, charts, relationships and visual information."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image}"}},
        ])])
        image_chain = image_prompt | ChatOpenAI(model=CHAT_MODEL, temperature=0) | StrOutputParser()
        image_summaries = image_chain.batch([{"image": str(img)} for img in normalized_images]) if normalized_images else []

        all_summaries = list(text_summaries) + list(table_summaries) + list(image_summaries)
        all_originals = list(texts) + list(tables_html) + list(normalized_images)
        metadata_list = ([{"type": "Text", "page_number": get_page_number(t.metadata)} for t in texts]
                         + table_metadata + image_metadata)
        if not all_summaries:
            raise ValueError("No indexable text, tables, or images were extracted from this PDF.")

        summary_docs, summary_rows = [], []
        for index, (summary, original, metadata) in enumerate(zip(all_summaries, all_originals, metadata_list)):
            parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{pdf_id}:{index}:{metadata.get('type')}:{hashlib.sha256(str(original).encode('utf-8')).hexdigest()}"))
            summary_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"summary:{parent_id}:{hashlib.sha256(str(summary).encode('utf-8')).hexdigest()}"))
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant:{summary_id}"))
            source_metadata = dict(metadata)
            source_metadata["pdf_id"] = pdf_id
            source_metadata["parent_id"] = parent_id
            _store_parent(pdf_id, parent_id, str(original), source_metadata)
            summary_docs.append(Document(page_content=str(summary), metadata={"doc_id": parent_id, "pdf_id": pdf_id, "summary_id": summary_id, "summary_type": metadata.get("type", "Text"), "page_number": format_page_number(metadata.get("page_number")) or ""}))
            summary_rows.append({"summary_id": summary_id, "parent_id": parent_id, "summary_type": metadata.get("type", "Text"), "summary_text": str(summary), "qdrant_point_id": point_id})

        # Remove an incomplete/previous collection first, then let the LangChain
        # Qdrant integration open its own client (important for local embedded mode).
        client = get_qdrant_client()
        try:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
        finally:
            client.close()
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = QdrantVectorStore.from_documents(
            documents=summary_docs, embedding=embedding, url=QDRANT_URL or None,
            api_key=QDRANT_API_KEY, path=None if QDRANT_URL else QDRANT_LOCAL_PATH,
            collection_name=collection_name, ids=[r["qdrant_point_id"] for r in summary_rows],
            distance=Distance.COSINE,
        )
        _upsert_summary_rows(pdf_id, summary_rows)
        with db_connect() as conn:
            conn.execute("UPDATE pdfs SET status='completed',error=NULL,updated_at=CURRENT_TIMESTAMP WHERE pdf_id=?", (pdf_id,))
            conn.execute("UPDATE ingestion_jobs SET status='completed',finished_at=CURRENT_TIMESTAMP WHERE job_id=?", (job_id,))
        return build_conversational_chain(MultiVectorRetriever(vectorstore=vectorstore, docstore=SQLiteDocStore(pdf_id), id_key="doc_id"))
    except Exception as exc:
        with db_connect() as conn:
            conn.execute("UPDATE pdfs SET status='failed',error=?,updated_at=CURRENT_TIMESTAMP WHERE pdf_id=?", (str(exc), pdf_id))
            conn.execute("UPDATE ingestion_jobs SET status='failed',error=?,finished_at=CURRENT_TIMESTAMP WHERE job_id=?", (str(exc), job_id))
        raise


def load_multimodal_rag(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record or record["status"] != "completed":
        raise FileNotFoundError(f"No completed RAG index found for PDF: {pdf_id}")
    client = get_qdrant_client()
    try:
        if not client.collection_exists(record["collection_name"]):
            raise FileNotFoundError(f"Qdrant collection missing: {record['collection_name']}")
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = QdrantVectorStore(client=client, collection_name=record["collection_name"], embedding=embedding)
        # Keep the client alive through the vectorstore's client reference.
    except Exception:
        client.close()
        raise
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=SQLiteDocStore(pdf_id), id_key="doc_id")
    return build_conversational_chain(retriever)


def is_base64_image(value) -> bool:
    if not isinstance(value, str):
        return False
    try:
        raw = base64.b64decode(value, validate=True)
        return raw.startswith((b"\x89PNG", b"\xff\xd8", b"GIF")) or (raw.startswith(b"RIFF") and b"WEBP" in raw[:20])
    except Exception:
        return False


def extract_sources(docs) -> list[dict]:
    sources = []
    for doc in docs:
        metadata = getattr(doc, "source_metadata", {}) if isinstance(doc, SourceString) else getattr(doc, "metadata", {}) or {}
        source_type = metadata.get("type", "Text")
        page = format_page_number(metadata.get("page_number"))
        source = {"type": source_type, "page": page, "document_id": metadata.get("parent_id"), "preview": str(doc)[:240] if source_type != "Image" else "Extracted image"}
        if not any(s["type"] == source["type"] and s["page"] == source["page"] for s in sources):
            sources.append(source)
    return sources


def build_conversational_chain(retriever):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and latest user question, reformulate the question into a standalone question if needed. Do NOT answer it. Return only the standalone question."),
        MessagesPlaceholder("chat_history"), ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    def parse_docs(docs):
        images_b64, texts = [], []
        for doc in docs:
            content = str(doc)
            if is_base64_image(content):
                images_b64.append(content)
            else:
                texts.append(doc)
        return {"images": images_b64, "texts": texts, "sources": extract_sources(docs)}

    def build_prompt(inputs):
        context, question = inputs["context"], inputs["input"]
        combined_text = "\n".join(str(item) for item in context["texts"])
        content = [{"type": "text", "text": f"Answer using only this context:\n\n{combined_text}\n\nQuestion: {question}"}]
        for image in context["images"]:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}})
        return ChatPromptTemplate.from_messages([("user", content)])

    def multimodal_qa(inputs):
        docs = history_aware_retriever.invoke({"input": inputs["input"], "chat_history": inputs.get("chat_history", [])})
        parsed = parse_docs(docs)
        prompt = build_prompt({"context": parsed, "input": inputs["input"]})
        answer = (prompt | llm | StrOutputParser()).invoke({})
        return {"answer": answer, "context": parsed}

    return RunnableLambda(multimodal_qa)
