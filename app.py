"""Streamlit chat interface for DocuChat AI.

Features:
- Sidebar document uploader (supports multiple PDFs with size & format validation)
- Real-time ingestion pipeline status (PDF extraction -> chunking -> embeddings -> ChromaDB)
- Uploaded document list with page/chunk counts, document selection filter, and deletion
- Session-based document isolation via unique session IDs
- Modern Streamlit chat UI (`st.chat_message` / `st.chat_input`) with chat history & Clear Chat
- Expandable source citations (document name, page number, similarity distance, excerpt)
"""

import uuid
from typing import Dict, Any, List

import streamlit as st

from src.ingestion import DEFAULT_MAX_FILE_SIZE_BYTES, ingest_pdf_files
from src.vector_store import delete_document_embeddings
from src.generator import (
    is_api_key_configured,
    prepare_rag_context,
    stream_gemini_answer,
)

# ---------------- Configuration ----------------
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "docuchat"
MAX_FILE_SIZE_MB = DEFAULT_MAX_FILE_SIZE_BYTES // (1024 * 1024)
MAX_FILE_SIZE_BYTES = DEFAULT_MAX_FILE_SIZE_BYTES

st.set_page_config(
    page_title="DocuChat AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Session State Init ----------------
def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    if "indexed_docs" not in st.session_state:
        # Maps safe filename -> {"pages": int, "chunks": int}
        st.session_state.indexed_docs: Dict[str, Dict[str, int]] = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, Any]] = []


_init_session_state()


# ---------------- Document Ingestion Helper ----------------
def _ingest_uploaded_files(
    uploaded_files: List[Any],
    chunk_size: int,
    overlap: int,
) -> None:
    """Delegate uploaded PDF processing to ``src.ingestion.ingest_pdf_files``."""
    with st.status("Processing uploaded documents...", expanded=True) as status:
        file_payloads = [(f.name, bytes(f.getbuffer())) for f in uploaded_files]
        result = ingest_pdf_files(
            files=file_payloads,
            session_id=st.session_state.session_id,
            existing_docs=set(st.session_state.indexed_docs.keys()),
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            chunk_size=chunk_size,
            overlap=overlap,
            max_size_bytes=MAX_FILE_SIZE_BYTES,
            on_progress=st.write,
        )

        for warning_msg in result["warnings"]:
            st.warning(warning_msg)
        for error_msg in result["errors"]:
            st.error(error_msg)

        if result["indexed"]:
            st.session_state.indexed_docs.update(result["indexed"])
            status.update(
                label=(
                    f"Indexed {len(result['indexed'])} document(s) "
                    f"({result['total_chunks']} chunks)!"
                ),
                state="complete",
                expanded=False,
            )
        else:
            status.update(label="No new documents were indexed.", state="complete")


# ---------------- Sidebar: Upload & Document Management ----------------
with st.sidebar:
    st.title("📄 DocuChat AI")
    st.caption(f"Session ID: `{st.session_state.session_id}`")

    # API Key Status
    if is_api_key_configured():
        st.success("🟢 Gemini API Key configured", icon="✅")
    else:
        st.warning(
            "🟡 Gemini API key not set in `.env` — document upload & search still work!",
            icon="⚠️",
        )

    st.divider()

    # 1. Upload Form
    st.subheader("1. Upload PDFs")
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            f"Choose PDF file(s) (max {MAX_FILE_SIZE_MB} MB each)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        submit_upload = st.form_submit_button(
            "📥 Upload & Index PDFs",
            use_container_width=True,
            type="primary",
        )

    # Chunking & Retrieval Settings
    with st.expander("⚙️ RAG Settings"):
        top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=5)
        chunk_size = st.number_input(
            "Chunk Size (chars)", min_value=200, max_value=2000, value=500, step=50
        )
        overlap = st.number_input(
            "Chunk Overlap (chars)", min_value=0, max_value=300, value=50, step=10
        )

    if submit_upload:
        if not uploaded_files:
            st.warning("Please select at least one PDF file before clicking Upload.")
        elif overlap >= chunk_size:
            st.error("Chunk Overlap must be smaller than Chunk Size.")
        else:
            _ingest_uploaded_files(uploaded_files, int(chunk_size), int(overlap))

    st.divider()

    # 2. Uploaded Document List & Deletion
    st.subheader("2. Uploaded Documents")
    doc_names = list(st.session_state.indexed_docs.keys())

    if not doc_names:
        st.info("No documents uploaded yet.")
        selected_docs: List[str] = []
    else:
        selected_docs = st.multiselect(
            "Select documents to chat with:",
            options=doc_names,
            default=doc_names,
        )

        st.markdown("**Manage Files:**")
        for doc_name in doc_names:
            info = st.session_state.indexed_docs[doc_name]
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.markdown(f"📄 **{doc_name}**")
                st.caption(f"{info['pages']} page(s) · {info['chunks']} chunk(s)")
            with col_del:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"Remove {doc_name}"):
                    delete_document_embeddings(
                        collection_name=COLLECTION_NAME,
                        document_name=doc_name,
                        session_id=st.session_state.session_id,
                        persist_directory=PERSIST_DIR,
                    )
                    del st.session_state.indexed_docs[doc_name]
                    st.rerun()

    st.divider()

    # 3. Chat Controls
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------- Main Chat Interface ----------------
st.header("💬 Chat with Your Documents")

# Summary Metrics Bar
total_docs = len(st.session_state.indexed_docs)
total_chunks = sum(d["chunks"] for d in st.session_state.indexed_docs.values())
m1, m2, m3 = st.columns(3)
m1.metric("Uploaded Documents", total_docs)
m2.metric("Indexed Chunks", total_chunks)
m3.metric("Active in Chat", len(selected_docs))

st.divider()

# Empty state welcome message
if not st.session_state.indexed_docs:
    st.info(
        "👈 **Get started:** Use the sidebar on the left to upload one or more PDF documents "
        "(resumes, research papers, study notes, business docs, SAP manuals, etc.), then ask questions below!"
    )
elif not selected_docs:
    st.warning("Select at least one document in the sidebar to start chatting.")

def _render_sources(sources: List[Dict[str, Any]]) -> None:
    """Display retrieved document sources inside a collapsible expander."""
    if not sources:
        return
    with st.expander(f"📚 Retrieved Sources ({len(sources)})"):
        for idx, src in enumerate(sources, start=1):
            dist_text = (
                f" | Distance: `{src['distance']:.4f}`"
                if src.get("distance") is not None
                else ""
            )
            st.markdown(
                f"**{idx}. {src['document_name']} — Page {src['page_number']}**{dist_text}"
            )
            st.caption(src["text"])


# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))

# Chat Input
chat_disabled = not selected_docs
if query := st.chat_input(
    "Ask a question about your selected documents...",
    disabled=chat_disabled,
):
    # Append & display user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve & stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            ctx = prepare_rag_context(
                query=query,
                collection_name=COLLECTION_NAME,
                top_k=top_k,
                persist_directory=PERSIST_DIR,
                session_id=st.session_state.session_id,
                selected_documents=selected_docs,
            )
            prompt = ctx["prompt"]
            sources = ctx["sources"]

        if not prompt:
            answer = "I couldn't find any relevant excerpts in the selected documents."
            st.markdown(answer)
        elif not is_api_key_configured():
            answer = (
                "⚠️ **Gemini API key is not configured yet.** "
                "Your documents were searched and the matching excerpts are shown under **Sources** below. "
                "Add a valid `GEMINI_API_KEY` in `.env` to generate AI answers."
            )
            st.markdown(answer)
        else:
            answer = st.write_stream(stream_gemini_answer(prompt))

        _render_sources(sources)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
