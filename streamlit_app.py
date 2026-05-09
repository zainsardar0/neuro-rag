import streamlit as st
import requests

# ─── Config ───────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="NeuroRAG — AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# ─── Helper Functions ──────────────────────────────────────

def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def api_query(query: str):
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"query": query},
            timeout=60
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return None


def api_upload(file_bytes, filename: str):
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=120
        )
        return r.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def api_documents():
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None


# ─── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("NeuroRAG")
    st.caption("AI Research Assistant")
    st.divider()

    # Check API status
    health = api_health()
    if health:
        st.success("🟢 API Connected")
        st.metric("Chunks in DB", health.get("total_chunks_in_db", 0))
        st.caption(f"Model: llama-3.3-70b-versatile")
        st.caption(f"Env: {health.get('environment', 'N/A')}")
    else:
        st.error("🔴 API Offline")
        st.warning("Start FastAPI server:\nuvicorn app.main:app --reload --port 8000")

    st.divider()
    page = st.radio(
        "Navigation",
        ["🔍 Query", "📄 Upload Documents", "📊 System Info"]
    )

# ─── Page 1: Query ─────────────────────────────────────────

if page == "🔍 Query":
    st.title("🔍 Ask a Research Question")
    st.caption("Query your ingested research documents with AI-powered answers and citations.")

    # Show ingested docs
    docs = api_documents()
    if docs and docs["documents"]:
        st.info(f"📚 Currently querying: {', '.join(docs['documents'])}")
    else:
        st.warning("No documents ingested yet. Go to Upload Documents to add PDFs.")

    st.divider()

    # Query input
    query = st.text_area(
        "Your Question",
        placeholder="e.g. What is the attention mechanism in transformers?",
        height=100
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("Ask NeuroRAG", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if submit:
        if not query.strip():
            st.error("Please enter a question.")
        elif not health:
            st.error("API is offline. Please start the FastAPI server.")
        else:
            with st.spinner("🧠 Thinking..."):
                result = api_query(query)

            if result:
                st.divider()

                # V2: Query Rewriting — show what was actually searched
                rewritten = result.get("rewritten_query", "").strip()
                if rewritten and rewritten != query.strip():
                    with st.expander("🔁 Query Rewriting (V2)", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Query**")
                            st.info(query.strip())
                        with col2:
                            st.markdown("**Rewritten for Retrieval**")
                            st.success(rewritten)

                # Answer
                st.subheader("📝 Answer")
                st.markdown(result["answer"])

                st.divider()

                # Sources
                if result["sources"]:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(
                            f"Source {i} — {source['file']} | Page {source['page']} | Score: {source['score']}"
                        ):
                            st.write(f"**File:** {source['file']}")
                            st.write(f"**Page:** {source['page']}")
                            st.write(f"**Relevance Score:** {source['score']}")
                else:
                    st.info("No sources found — fallback response used.")

                # Metadata
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", result["model"].split("-")[0] + "...")
                col2.metric("Used Fallback", "Yes" if result["used_fallback"] else "No")
                col3.metric("Retries", result["retry_count"])
            else:
                st.error("Failed to get response. Check if API is running.")

# ─── Page 2: Upload Documents ──────────────────────────────

elif page == "📄 Upload Documents":
    st.title("📄 Upload Research Documents")
    st.caption("Upload PDF files to add them to the knowledge base.")

    # Current documents
    st.subheader("📚 Currently Ingested Documents")
    docs = api_documents()
    if docs and docs["documents"]:
        for doc in docs["documents"]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {doc}")
        st.metric("Total Chunks in DB", docs["total_chunks"])
    else:
        st.info("No documents ingested yet.")

    st.divider()

    # Upload section
    st.subheader("➕ Add New Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Maximum file size: 10MB"
    )

    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        col1, col2 = st.columns(2)
        col1.metric("File Name", uploaded_file.name)
        col2.metric("File Size", f"{file_size_mb:.2f} MB")

        if file_size_mb > 10:
            st.error("File too large. Maximum size is 10MB.")
        else:
            if st.button("📥 Ingest Document", type="primary"):
                if not health:
                    st.error("API is offline. Please start the FastAPI server.")
                else:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = api_upload(
                            uploaded_file.getvalue(),
                            uploaded_file.name
                        )

                    if result.get("success"):
                        st.success(f"✅ {result['message']}")
                        st.metric("Chunks Added", result["chunks_added"])
                        st.metric("Total Chunks in DB", result["total_chunks"])
                        st.rerun()
                    else:
                        st.error(f"❌ {result.get('message', 'Upload failed')}")

# ─── Page 3: System Info ───────────────────────────────────

elif page == "📊 System Info":
    st.title("📊 System Information")

    if not health:
        st.error("API is offline. Please start the FastAPI server.")
    else:
        # System metrics
        st.subheader("⚙️ System Status")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", health["status"].upper())
        col2.metric("Environment", health["environment"])
        col3.metric("Total Chunks", health["total_chunks_in_db"])

        st.divider()

        # Model info
        st.subheader("🤖 Model Configuration")
        col1, col2, col3 = st.columns(3)
        col1.metric("LLM", "LLaMA 3.3 70B")
        col2.metric("Provider", "Groq")
        col3.metric("Embedding Model", "all-MiniLM-L6-v2")

        col1, col2, col3 = st.columns(3)
        col1.metric("Embedding Dims", "384")
        col2.metric("Chunk Size", "1000 chars")
        col3.metric("Chunk Overlap", "200 chars")

        st.divider()

        # Documents
        st.subheader("📚 Ingested Documents")
        docs = api_documents()
        if docs and docs["documents"]:
            for doc in docs["documents"]:
                st.write(f"📄 {doc}")
        else:
            st.info("No documents ingested yet.")

        st.divider()

        # Evaluation scores
        st.subheader("📈 Evaluation Results (V1)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pass Rate", "100%")
        col2.metric("Keyword Score", "93%")
        col3.metric("Fallback Accuracy", "100%")
        col4.metric("Citation Rate", "80%")

        st.divider()

        # V2 features
        st.subheader("🚀 V2 Features")
        col1, col2 = st.columns(2)
        col1.success("✅ Phase 1: Query Rewriting")
        col2.info("🔜 Phase 2: Reranking")

        st.divider()

        # About
        st.subheader("👤 About")
        st.markdown("""
        **NeuroRAG — AI Research Assistant**  
        Built by **Muhammad Zain Ul Abideen**

        [![GitHub](https://img.shields.io/badge/GitHub-zainsardar0-black?logo=github)](https://github.com/zainsardar0)
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Zain-blue?logo=linkedin)](https://www.linkedin.com/in/muhammad-zain-ul-abideen-1705032b3/)
        """)