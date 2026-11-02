""" Streamlit interface for model discovery, document ingestion, RAG chat, recovery, and audit. """

import streamlit as st
from services.container import get_services
from services.document_service import DocumentService
from services.retrieval_service import RetrievalService
from services.audit_service import AuditService
from services.ollama_service import OllamaError

st.set_page_config(page_title="Local AI Knowledge Manager", page_icon="🧠", layout="wide")
st.title("🧠 Local AI Knowledge Manager")
st.caption("Modular local RAG · Ollama · Qdrant · Dynamic model discovery")

try:
    services = get_services()
    ollama = services["ollama"]
    qdrant = services["qdrant"]
    audit = services["audit"]
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("System status")
    health = ollama.health_check()
    st.write("Ollama:", "🟢 Connected" if health["available"] else "🔴 Unavailable")
    if health.get("version"):
        st.caption(f"Ollama version: {health['version']}")
    try:
        st.write("Qdrant:", "🟢 Connected" if qdrant.health_check() else "🔴 Unavailable")
    except Exception as exc:
        st.write("Qdrant: 🔴 Unavailable")
        st.caption(str(exc))
    if st.button("Refresh model registry"):
        st.cache_data.clear()
        st.rerun()

@st.cache_data(ttl=30)
def load_models():
    """ Handle the load models operation. """
    return [m.to_dict() for m in ollama.discover_models()]

try:
    model_rows = load_models()
    chat_models = [m for m in model_rows if m["supports_chat"]]
    embedding_models = [m for m in model_rows if m["supports_embeddings"]]
except OllamaError as exc:
    model_rows, chat_models, embedding_models = [], [], []
    st.error(f"Model discovery failed: {exc}")

tab_models, tab_ingest, tab_chat, tab_docs, tab_audit, tab_recovery = st.tabs(
    ["🧩 Model registry", "📥 Ingest", "💬 Chat", "📚 Documents", "🔎 AI activity", "🛠 Recovery"]
)

with tab_models:
    st.subheader("Discovered Ollama models")
    st.caption("Installed models come from /api/tags; loaded models come from /api/ps; capabilities come from /api/show.")
    if model_rows:
        st.dataframe(
            [{
                "Model": m["name"],
                "Status": "Running" if m["running"] else "Installed",
                "Chat": m["supports_chat"],
                "Embeddings": m["supports_embeddings"],
                "Family": m["family"],
                "Parameters": m["parameter_size"],
                "Quantization": m["quantization"],
                "Capability metadata": m["capability_status"],
                "Size (GB)": round(m["size"] / (1024**3), 2),
            } for m in model_rows],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No installed models discovered. Pull a model with `ollama pull gemma3`.")

with tab_ingest:
    st.subheader("Add knowledge")
    collection = st.text_input("Qdrant collection", "technical_docs")
    title = st.text_input("Document title", "Untitled")
    tags_text = st.text_input("Tags (comma separated)", "general")
    upload = st.file_uploader("Upload PDF, DOCX, TXT or Markdown", type=["pdf", "docx", "txt", "md"])
    pasted = st.text_area("Or paste text", height=180)
    if embedding_models:
        embedding_names = [m["name"] for m in embedding_models]
        default_embedding = st.selectbox("Embedding model", embedding_names)
    else:
        default_embedding = None
        st.warning("No model advertises embedding capability. Install an embedding model such as nomic-embed-text.")
    chunk_size = st.slider("Chunk size (characters)", 500, 5000, 2400, 100)
    overlap = st.slider("Chunk overlap", 0, 800, 300, 50)
    if st.button("Preview chunks"):
        try:
            raw = DocumentService.extract(upload, services["documents"].max_upload_bytes) if upload else pasted
            DocumentService.validate_text_size(raw, services["documents"].max_text_bytes)
            if not raw.strip():
                raise ValueError("Upload a file or enter text.")
            if overlap >= chunk_size:
                raise ValueError("Overlap must be smaller than chunk size.")
            chunks = DocumentService.chunk(raw, chunk_size, overlap)
            st.session_state["pending_chunks"] = chunks
            st.session_state["pending_title"] = upload.name if upload else title
            st.session_state["pending_collection"] = collection.strip()
            st.session_state["pending_tags"] = [x.strip() for x in tags_text.split(",") if x.strip()]
            st.session_state["pending_embedding"] = default_embedding
            st.success(f"Created {len(chunks)} chunks.")
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get("pending_chunks"):
        st.write("Chunk preview")
        for idx, chunk in enumerate(st.session_state["pending_chunks"]):
            with st.expander(f"Chunk {idx + 1}"):
                st.text(chunk)
        if st.button("Save document to Qdrant"):
            try:
                if not st.session_state.get("pending_embedding"):
                    raise ValueError("Select a valid embedding model.")
                with st.spinner("Embedding and storing document..."):
                    result = services["documents"].ingest(
                        collection=st.session_state["pending_collection"],
                        title=st.session_state["pending_title"],
                        chunks=st.session_state["pending_chunks"],
                        tags=st.session_state["pending_tags"],
                        embedding_model=st.session_state["pending_embedding"],
                    )
                st.success(f"Stored document {result['document_id']} ({result['chunk_count']} chunks).")
                st.session_state.pop("pending_chunks", None)
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

with tab_chat:
    st.subheader("Ask your knowledge base")
    try:
        collections = qdrant.list_collections()
    except Exception as exc:
        collections = []
        st.error(f"Cannot list collections: {exc}")
    if not collections:
        st.info("Ingest a document first.")
    elif not chat_models:
        st.warning("No installed model advertises chat capability.")
    else:
        collection = st.selectbox("Collection", collections, key="chat_collection")
        chat_model = st.selectbox("Chat model", [m["name"] for m in chat_models])
        if embedding_models:
            query_embedding_model = st.selectbox("Query embedding model", [m["name"] for m in embedding_models])
        else:
            query_embedding_model = None
            st.warning("No embedding-capable model discovered.")
        top_k = st.slider("Retrieved chunks", 1, 15, 5)
        question = st.chat_input("Ask a question about your documents")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Retrieving context and querying Ollama..."):
                        answer, sources, trace = services["rag"].answer(
                            collection=collection,
                            question=question,
                            chat_model=chat_model,
                            top_k=top_k,
                            embedding_model=query_embedding_model,
                        )
                    st.markdown(answer)
                    st.caption(f"Model: {chat_model} · Request: {trace['request_id']} · {trace['duration_seconds']:.2f}s")
                    with st.expander("Retrieved source chunks"):
                        for source in sources:
                            st.markdown(
                                f"**{source['title']} · chunk {source['chunk_index']} · "
                                f"score {source['score']:.4f}**"
                            )
                            st.write(source["content"])
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    st.error(f"Chat request failed: {exc}")

with tab_docs:
    st.subheader("Stored documents")
    try:
        collections = qdrant.list_collections()
        if collections:
            selected = st.selectbox("Collection to inspect", collections, key="docs_collection")
            docs = qdrant.list_documents(selected)
            if not docs:
                st.info("No document records found.")
            for doc_id, data in docs.items():
                with st.expander(f"{data['title']} · {data['chunks']} chunks"):
                    st.code(doc_id)
                    st.write("Tags:", data.get("tags", []))
                    if st.button("Delete document", key=f"delete_{doc_id}"):
                        qdrant.delete_document(selected, doc_id)
                        st.success("Document deleted.")
                        st.rerun()
        else:
            st.info("No collections yet.")
    except Exception as exc:
        st.error(f"Document listing failed: {exc}")

with tab_audit:
    st.subheader("Inference audit trail")
    st.caption("These are application-side records of prompts/context submitted to Ollama. They do not expose model-internal reasoning.")
    records = audit.recent(limit=100)
    if records:
        st.metric("Recent recorded requests", len(records))
        for record in records:
            with st.expander(f"{record['created_at']} · {record['model']} · {record['status']}"):
                st.write("Request ID:", record["request_id"])
                st.write("Collection:", record["collection_name"])
                st.write("Question:", record["question"])
                st.write("Duration:", record["duration_seconds"])
                st.write("Error:", record["error"])
                st.json(record["sources"])
                with st.expander("Context sent to model"):
                    st.text(record["context_sent"])
                with st.expander("Model response"):
                    st.write(record["response"])
    else:
        st.info("No inference activity recorded yet.")


with tab_recovery:
    st.subheader("Interrupted ingestion recovery")
    st.caption(
        "Startup recovery checks stale PREPARING, RECOVERING and ROLLBACK_FAILED "
        "manifests. Manual recovery retries eligible stale manifests."
    )
    startup_report = services.get("recovery_report", {})
    if startup_report:
        st.write("Last startup recovery report")
        st.json(startup_report)

    stale_seconds = st.number_input(
        "Minimum manifest age before recovery (seconds)",
        min_value=0, max_value=86400, value=120, step=30
    )
    if st.button("Run recovery now"):
        try:
            with st.spinner("Reconciling SQLite manifests with Qdrant..."):
                report = services["recovery"].run(
                    stale_after_seconds=int(stale_seconds),
                    limit=500
                )
            st.success("Recovery run completed.")
            st.json(report)
        except Exception as exc:
            st.error(f"Recovery run failed: {exc}")

    manifests = audit.list_ingestion_manifests(limit=200)
    if manifests:
        st.dataframe(manifests, use_container_width=True, hide_index=True)
    else:
        st.info("No ingestion manifests recorded.")
