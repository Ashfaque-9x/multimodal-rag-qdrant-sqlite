"""Streamlit UI for persistent multimodal RAG backed by Qdrant and SQLite."""
import base64
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from multimodel_rag import (
    BASE_STORAGE_DIR, PDF_STORAGE_DIR, SQLITE_PATH,
    build_multimodal_rag, load_multimodal_rag, rag_exists,
    get_available_pdfs, get_pdf_storage_paths, calculate_pdf_id,
    register_pdf, get_pdf_record, load_messages, load_chat_history,
    save_message, clear_chat, delete_pdf,
)

load_dotenv()
st.set_page_config(page_title="Multimodal RAG Chat", page_icon="📄", layout="wide")

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key and hasattr(st, "secrets"):
    openai_key = st.secrets.get("OPENAI_API_KEY")
if not openai_key:
    with st.sidebar:
        st.header("Configuration")
        st.warning("Set OPENAI_API_KEY in your .env file before using the application.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_key

st.title("📄 Multi-Modal RAG by Virtual Techbox")
st.caption(f"Vector store: Qdrant · Persistent state: SQLite ({SQLITE_PATH})")

if "selected_pdf_id" not in st.session_state:
    st.session_state.selected_pdf_id = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_sources(sources):
    with st.popover("Sources"):
        st.markdown("### Sources")
        if not sources:
            st.caption("No source metadata available.")
            return
        for index, source in enumerate(sources, 1):
            kind = source.get("type", "Document")
            icon = {"Text": "📄", "Table": "📊", "Image": "🖼️"}.get(kind, "📌")
            page = source.get("page")
            st.markdown(f"**{index}. {icon} {kind}**  \n" + (f"Page {page}" if page else "Page information unavailable"))
            if source.get("preview") and kind != "Image":
                st.caption(source["preview"])
            if index < len(sources):
                st.divider()


def select_pdf(pdf_id):
    st.session_state.selected_pdf_id = pdf_id
    st.session_state.rag_chain = load_multimodal_rag(pdf_id) if rag_exists(pdf_id) else None
    st.session_state.messages = load_messages(pdf_id)
    st.session_state.chat_history = load_chat_history(pdf_id)

with st.sidebar:
    st.header("📚 Your PDFs")
    pdfs = get_available_pdfs()
    pdf_map = {item["pdf_id"]: item["filename"] for item in pdfs}
    options = list(pdf_map)
    if options:
        current = st.session_state.selected_pdf_id
        index = options.index(current) if current in options else 0
        chosen = st.selectbox("Select PDF", options, index=index, format_func=lambda key: pdf_map.get(key, key[:12]))
        if chosen != st.session_state.selected_pdf_id:
            try:
                select_pdf(chosen)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load this PDF's RAG index: {exc}")
    else:
        st.info("No processed PDFs yet. Upload one below.")

    st.divider()
    if st.session_state.selected_pdf_id:
        if st.button("🧹 Clear Current PDF Chat", use_container_width=True):
            clear_chat(st.session_state.selected_pdf_id)
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        if st.button("🗑️ Delete Current PDF", use_container_width=True):
            try:
                delete_pdf(st.session_state.selected_pdf_id)
                st.session_state.selected_pdf_id = None
                st.session_state.rag_chain = None
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete PDF: {exc}")

uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])
if uploaded_file:
    pdf_bytes = uploaded_file.getvalue()
    pdf_id = __import__("hashlib").sha256(pdf_bytes).hexdigest()
    pdf_path, _ = get_pdf_storage_paths(pdf_id)
    if not pdf_path.exists():
        pdf_path.write_bytes(pdf_bytes)
    existing = get_pdf_record(pdf_id)
    if not existing or existing.get("status") != "completed" or not rag_exists(pdf_id):
        register_pdf(pdf_id, uploaded_file.name, pdf_path, "processing")
        with st.spinner("Extracting PDF and building the Qdrant index..."):
            try:
                chain = build_multimodal_rag(str(pdf_path), pdf_id=pdf_id, filename=uploaded_file.name)
                st.success("PDF processed and persisted in Qdrant and SQLite.")
            except Exception as exc:
                st.error(f"PDF processing failed: {exc}")
                st.stop()
    else:
        chain = load_multimodal_rag(pdf_id)
    if st.session_state.selected_pdf_id != pdf_id:
        st.session_state.selected_pdf_id = pdf_id
        st.session_state.rag_chain = chain
        st.session_state.messages = load_messages(pdf_id)
        st.session_state.chat_history = load_chat_history(pdf_id)
    else:
        st.session_state.rag_chain = chain

if st.session_state.selected_pdf_id and st.session_state.rag_chain is None:
    try:
        st.session_state.rag_chain = load_multimodal_rag(st.session_state.selected_pdf_id)
        st.session_state.messages = load_messages(st.session_state.selected_pdf_id)
        st.session_state.chat_history = load_chat_history(st.session_state.selected_pdf_id)
    except Exception as exc:
        st.error(f"Could not restore RAG: {exc}")

if st.session_state.selected_pdf_id:
    record = get_pdf_record(st.session_state.selected_pdf_id)
    st.caption(f"📄 Current PDF: **{record['filename'] if record else st.session_state.selected_pdf_id}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg.get("content", ""))
        for img_b64 in msg.get("images", []) or []:
            try:
                st.image(base64.b64decode(img_b64))
            except Exception:
                st.caption("An image could not be displayed.")
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

question = st.chat_input("Ask a question about the PDF...")
if question:
    if not st.session_state.selected_pdf_id or st.session_state.rag_chain is None:
        st.warning("Please upload or select a PDF first.")
        st.stop()
    pdf_id = st.session_state.selected_pdf_id
    with st.chat_message("user"):
        st.write(question)
    save_message(pdf_id, "user", question)
    st.session_state.messages.append({"role": "user", "content": question, "images": [], "sources": []})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
                answer = response.get("answer", "") if isinstance(response, dict) else str(response)
                context = response.get("context", {}) if isinstance(response, dict) else {}
                images = context.get("images", []) or []
                sources = context.get("sources", []) or []
                st.write(answer)
                if images:
                    st.markdown("### Retrieved Images")
                    for image_b64 in images:
                        try:
                            st.image(base64.b64decode(image_b64))
                        except Exception:
                            st.caption("A retrieved image could not be displayed.")
                render_sources(sources)
            except Exception as exc:
                st.error(f"RAG query failed: {exc}")
                st.stop()

    st.session_state.chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    st.session_state.messages.append({"role": "assistant", "content": answer, "images": images, "sources": sources})
    save_message(pdf_id, "assistant", answer, images=images, sources=sources)
