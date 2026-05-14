from app.embedding.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.core.logger import app_logger
from app.core.exceptions import RetrievalError

# Minimum similarity score to consider a semantic chunk relevant
SIMILARITY_THRESHOLD = 0.3

# Fetch more candidates for reranker to work with
RETRIEVAL_TOP_K = 10

# Reciprocal Rank Fusion constant — standard value
RRF_K = 60


class Retriever:
    """
    Handles retrieval of relevant document chunks.
    V2 Phase 3: Hybrid Search — BM25 + semantic + RRF fusion + reranking.
    """

    def __init__(self, top_k: int = 5):
        """
        Initialize retriever with embedder, vector store, reranker, BM25.

        Args:
            top_k: Number of final chunks to return after reranking
        """
        try:
            self.top_k = top_k
            self.embedder = Embedder()
            self.vector_store = VectorStore()
            self.reranker = Reranker()

            # V3 Phase 3: Build BM25 index from all chunks in ChromaDB
            all_chunks = self.vector_store.get_all_chunks()
            self.bm25 = BM25Retriever(all_chunks)

            app_logger.info(f"Retriever initialized with top_k={top_k}")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize retriever: {str(e)}")

    def _reciprocal_rank_fusion(
        self,
        semantic_chunks: list[dict],
        bm25_chunks: list[dict],
        k: int = RRF_K
    ) -> list[dict]:
        """
        Combine semantic and BM25 results using Reciprocal Rank Fusion.

        RRF score = 1/(k + rank) for each list.
        Final score = sum of RRF scores across both lists.
        Chunks appearing in both lists get boosted.

        Args:
            semantic_chunks: Results from ChromaDB sorted by cosine score
            bm25_chunks: Results from BM25 sorted by BM25 score
            k: RRF constant (default 60)

        Returns:
            Merged and sorted list of unique chunks
        """
        # Build a unique key for each chunk
        def chunk_key(chunk):
            return f"{chunk['source']}::{chunk['page']}::{chunk['text'][:50]}"

        # Track RRF scores per chunk key
        rrf_scores = {}
        chunk_map = {}

        # Process semantic results
        for rank, chunk in enumerate(semantic_chunks, start=1):
            key = chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

        # Process BM25 results
        for rank, chunk in enumerate(bm25_chunks, start=1):
            key = chunk_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            if key not in chunk_map:
                # BM25 chunk needs cosine score placeholder
                chunk["score"] = 0.0
                chunk_map[key] = chunk

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        # Build final list with rrf_score attached
        fused = []
        for key in sorted_keys:
            chunk = chunk_map[key]
            chunk["rrf_score"] = round(rrf_scores[key], 6)
            fused.append(chunk)

        app_logger.info(
            f"RRF fusion: {len(semantic_chunks)} semantic + "
            f"{len(bm25_chunks)} BM25 → {len(fused)} unique chunks"
        )

        return fused

    def retrieve(self, query: str) -> list[dict]:
        """
        Hybrid retrieval: BM25 + semantic search fused with RRF, then reranked.

        Flow:
            1. Embed query
            2. Semantic search — ChromaDB top 10
            3. BM25 search — keyword top 10
            4. RRF fusion — combine both ranked lists
            5. Filter fused results by similarity threshold
            6. Rerank with CrossEncoder
            7. Return top 5

        Args:
            query: Rewritten query string

        Returns:
            List of dicts with 'text', 'page', 'source',
            'score', 'rerank_score', 'rrf_score' keys

        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")

        try:
            app_logger.info(f"Hybrid retrieval for query: {query[:50]}...")

            # Step 1 — Embed query
            query_embedding = self.embedder.embed_query(query)

            # Step 2 — Semantic search
            semantic_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=RETRIEVAL_TOP_K
            )
            # Filter semantic by threshold
            semantic_filtered = [
                r for r in semantic_results
                if r["score"] >= SIMILARITY_THRESHOLD
            ]
            app_logger.info(
                f"Semantic: {len(semantic_filtered)} chunks above threshold"
            )

            # Step 3 — BM25 search
            bm25_results = self.bm25.retrieve(query, top_k=RETRIEVAL_TOP_K)
            app_logger.info(f"BM25: {len(bm25_results)} chunks retrieved")

            # Step 4 — RRF fusion
            if semantic_filtered and bm25_results:
                fused = self._reciprocal_rank_fusion(
                    semantic_filtered, bm25_results
                )
                retrieval_method = "hybrid"
            elif semantic_filtered:
                # BM25 returned nothing — use semantic only
                fused = semantic_filtered
                retrieval_method = "semantic"
                app_logger.warning("BM25 empty — using semantic only")
            elif bm25_results:
                # Semantic returned nothing — use BM25 only
                for chunk in bm25_results:
                    chunk["score"] = 0.0
                fused = bm25_results
                retrieval_method = "bm25"
                app_logger.warning("Semantic empty — using BM25 only")
            else:
                app_logger.warning("Both semantic and BM25 returned no results")
                return []

            # Step 5 — Take top candidates for reranking
            candidates = fused[:RETRIEVAL_TOP_K]

            # Step 6 — Rerank with CrossEncoder
            reranked = self.reranker.rerank(query, candidates)

            # Attach retrieval method to each chunk
            for chunk in reranked:
                chunk["retrieval_method"] = retrieval_method

            # Log top result
            if reranked:
                app_logger.info(
                    f"Top result — Method: {retrieval_method} | "
                    f"Rerank: {reranked[0]['rerank_score']} | "
                    f"Source: {reranked[0]['source']} | "
                    f"Page: {reranked[0]['page']}"
                )

            return reranked

        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    def retrieve_with_fallback(self, query: str) -> tuple[list[dict], bool]:
        """
        Hybrid retrieve with fallback flag for LangGraph.

        Args:
            query: User's question string

        Returns:
            Tuple of (results, needs_fallback)
        """
        results = self.retrieve(query)
        needs_fallback = len(results) == 0
        return results, needs_fallback