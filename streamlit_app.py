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


def api_evaluate_ragas(test_cases: list = None):
    """V2 Phase 4: Call RAGAS evaluation endpoint."""
    try:
        payload = {"test_cases": test_cases if test_cases else []}
        r = requests.post(
            f"{API_BASE}/evaluate/ragas",
            json=payload,
            timeout=300        # RAGAS takes time — 5 min timeout
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return None


# ─── Sidebar ───────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("NeuroRAG")
    st.caption("AI Research Assistant")
    st.divider()

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
        ["🔍 Query", "📄 Upload Documents", "📊 System Info", "🧪 RAGAS Evaluation"]
    )

# ─── Page 1: Query ─────────────────────────────────────────

if page == "🔍 Query":
    st.title("🔍 Ask a Research Question")
    st.caption("Query your ingested research documents with AI-powered answers and citations.")

    docs = api_documents()
    if docs and docs["documents"]:
        st.info(f"📚 Currently querying: {', '.join(docs['documents'])}")
    else:
        st.warning("No documents ingested yet. Go to Upload Documents to add PDFs.")

    st.divider()

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

                # V2 Phase 1: Query Rewriting
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

                # V2 Phase 3: Retrieval method badge
                retrieval_method = result.get("retrieval_method", "hybrid")
                method_labels = {
                    "hybrid": "🔀 Hybrid Search (BM25 + Semantic)",
                    "semantic": "🧠 Semantic Search only",
                    "bm25": "🔑 BM25 Keyword Search only",
                    "none": "⚠️ No retrieval method"
                }
                st.caption(
                    f"Retrieval: {method_labels.get(retrieval_method, retrieval_method)}"
                )

                # Answer
                st.subheader("📝 Answer")
                st.markdown(result["answer"])

                st.divider()

                # Sources
                if result["sources"]:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        rerank_score = source.get("rerank_score", None)
                        cosine_score = source.get("score", 0)

                        with st.expander(
                            f"Source {i} — {source['file']} | "
                            f"Page {source['page']} | "
                            f"Rerank: {rerank_score} | "
                            f"Cosine: {cosine_score}"
                        ):
                            st.write(f"**File:** {source['file']}")
                            st.write(f"**Page:** {source['page']}")

                            score_col1, score_col2 = st.columns(2)
                            with score_col1:
                                st.metric(
                                    "Rerank Score",
                                    rerank_score if rerank_score is not None else "N/A",
                                    help="CrossEncoder relevance score (0-1)."
                                )
                            with score_col2:
                                st.metric(
                                    "Cosine Score",
                                    cosine_score,
                                    help="Vector similarity score from ChromaDB (0-1)."
                                )
                else:
                    st.info("No sources found — fallback response used.")

                # Metadata
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Model", result["model"].split("-")[0] + "...")
                col2.metric("Used Fallback", "Yes" if result["used_fallback"] else "No")
                col3.metric("Retries", result["retry_count"])
                col4.metric("Retrieval", retrieval_method.upper())
            else:
                st.error("Failed to get response. Check if API is running.")

# ─── Page 2: Upload Documents ──────────────────────────────

elif page == "📄 Upload Documents":
    st.title("📄 Upload Research Documents")
    st.caption("Upload PDF files to add them to the knowledge base.")

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
        st.subheader("⚙️ System Status")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", health["status"].upper())
        col2.metric("Environment", health["environment"])
        col3.metric("Total Chunks", health["total_chunks_in_db"])

        st.divider()

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

        st.subheader("📚 Ingested Documents")
        docs = api_documents()
        if docs and docs["documents"]:
            for doc in docs["documents"]:
                st.write(f"📄 {doc}")
        else:
            st.info("No documents ingested yet.")

        st.divider()

        st.subheader("📈 Evaluation Results (V1)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pass Rate", "100%")
        col2.metric("Keyword Score", "93%")
        col3.metric("Fallback Accuracy", "100%")
        col4.metric("Citation Rate", "80%")

        st.divider()

        st.subheader("🚀 V2 Features")
        col1, col2, col3, col4 = st.columns(4)
        col1.success("✅ Phase 1: Query Rewriting")
        col2.success("✅ Phase 2: Reranking")
        col3.success("✅ Phase 3: Hybrid Search")
        col4.success("✅ Phase 4: RAGAS Evaluation")

        st.divider()

        st.subheader("👤 About")
        st.markdown("""
        **NeuroRAG — AI Research Assistant**
        Built by **Muhammad Zain Ul Abideen**

        [![GitHub](https://img.shields.io/badge/GitHub-zainsardar0-black?logo=github)](https://github.com/zainsardar0)
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Zain-blue?logo=linkedin)](https://www.linkedin.com/in/muhammad-zain-ul-abideen-1705032b3/)
        """)

# ─── Page 4: RAGAS Evaluation ──────────────────────────────

elif page == "🧪 RAGAS Evaluation":
    st.title("🧪 RAGAS Evaluation Dashboard")
    st.caption("Industry-standard RAG evaluation using Faithfulness, Response Relevancy, and Context Precision.")

    if not health:
        st.error("API is offline. Please start the FastAPI server.")
    else:
        # Metric explanations
        with st.expander("ℹ️ What do these metrics mean?", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🎯 Faithfulness**")
                st.markdown("Is the answer grounded in the retrieved context? Detects hallucinations. Score 0-1.")
            with col2:
                st.markdown("**💬 Response Relevancy**")
                st.markdown("Does the answer actually address the question asked? Score 0-1.")
            with col3:
                st.markdown("**🔍 Context Precision**")
                st.markdown("Are the retrieved chunks relevant to the question? Score 0-1.")

        st.divider()

        # Custom test cases
        st.subheader("📝 Test Cases")
        st.caption("Leave empty to use default BERT paper test cases, or add your own.")

        custom_queries = st.text_area(
            "Custom queries (one per line)",
            placeholder="What is BERT?\nHow does MLM work?\nWhat is fine-tuning?",
            height=120
        )

        test_cases = []
        if custom_queries.strip():
            test_cases = [q.strip() for q in custom_queries.strip().split("\n") if q.strip()]
            st.info(f"Using {len(test_cases)} custom test cases.")
        else:
            st.info("Using 5 default BERT paper test cases.")

        st.divider()

        # Run evaluation
        if st.button("🚀 Run RAGAS Evaluation", type="primary"):
            docs = api_documents()
            if not docs or not docs["documents"]:
                st.error("No documents ingested. Please upload documents first.")
            else:
                with st.spinner("🧪 Running RAGAS evaluation... This may take 2-5 minutes."):
                    results = api_evaluate_ragas(test_cases if test_cases else None)

                if results:
                    st.success("✅ Evaluation complete!")
                    st.divider()

                    # Overall score
                    overall = results.get("overall_score", 0)
                    st.subheader("📊 Overall RAGAS Score")

                    score_color = "green" if overall >= 0.7 else "orange" if overall >= 0.5 else "red"
                    st.markdown(
                        f"<h1 style='text-align:center; color:{score_color}'>{overall:.2%}</h1>",
                        unsafe_allow_html=True
                    )

                    st.divider()

                    # Per-metric scores
                    st.subheader("📈 Metric Scores")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "🎯 Faithfulness",
                        f"{results.get('faithfulness', 0):.2%}",
                        help="Higher = less hallucination"
                    )
                    col2.metric(
                        "💬 Response Relevancy",
                        f"{results.get('response_relevancy', 0):.2%}",
                        help="Higher = more relevant answers"
                    )
                    col3.metric(
                        "🔍 Context Precision",
                        f"{results.get('context_precision', 0):.2%}",
                        help="Higher = better retrieval"
                    )

                    st.divider()

                    # Per-query results
                    st.subheader("🔬 Per-Query Results")
                    per_query = results.get("per_query_results", [])
                    if per_query:
                        for i, row in enumerate(per_query, 1):
                            with st.expander(f"Query {i}: {row.get('user_input', '')[:60]}..."):
                                q_col1, q_col2, q_col3 = st.columns(3)
                                q_col1.metric(
                                    "Faithfulness",
                                    f"{row.get('faithfulness', 0):.2%}"
                                )
                                q_col2.metric(
                                    "Relevancy",
                                    f"{row.get('response_relevancy', 0):.2%}"
                                )
                                q_col3.metric(
                                    "Precision",
                                    f"{row.get('llm_context_precision_without_reference', 0):.2%}"
                                )
                else:
                    st.error("RAGAS evaluation failed. Check FastAPI terminal for details.")