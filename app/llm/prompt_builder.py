def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a grounded RAG prompt from query and retrieved chunks.

    Args:
        query: User's question
        chunks: Retrieved chunks with 'text', 'page', 'source', 'score'

    Returns:
        Formatted prompt string for the LLM
    """

    # Format each chunk with its citation info
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"[Source {i}] File: {chunk['source']} | Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""You are NeuroRAG, an AI research assistant. 
Your job is to answer questions strictly based on the provided context.

STRICT RULES:
1. Only use information from the provided context below
2. Always cite your sources using [Source N] format
3. If the context does not contain enough information, say: 
   "I cannot find sufficient information in the provided documents."
4. Never use outside knowledge or make up information
5. Be precise and academic in your responses

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (with citations):"""

    return prompt


def build_fallback_prompt(query: str) -> str:
    """
    Build a fallback prompt when no relevant chunks are found.

    Args:
        query: User's question

    Returns:
        Fallback prompt string
    """
    prompt = f"""You are NeuroRAG, an AI research assistant.
    
A user asked: "{query}"

No relevant documents were found in the knowledge base for this query.
Politely inform the user that you cannot find relevant information 
in the currently loaded documents, and suggest they may need to 
upload documents containing information about this topic.

Keep your response brief and helpful."""

    return prompt


def build_rewrite_prompt(query: str) -> str:
    """
    Build a prompt to rewrite a user query for better vector retrieval.

    The goal is to expand vague or conversational queries into
    keyword-rich, semantically dense queries suitable for embedding search.

    Args:
        query: Original user query

    Returns:
        Prompt string for the rewriter LLM
    """
    prompt = f"""You are a query optimization expert for a scientific research retrieval system.

Your task: Rewrite the user's query to maximize retrieval quality from a vector database 
containing academic research papers.

REWRITING RULES:
1. Expand abbreviations and vague terms into specific technical language
2. Add relevant domain-specific keywords and synonyms
3. Keep the core intent of the original query intact
4. Output ONLY the rewritten query — no explanation, no preamble, no quotes
5. Length: 1-2 sentences maximum

EXAMPLES:
Original: "what does the paper say about memory?"
Rewritten: memory consolidation mechanisms working memory neural networks cognitive processes

Original: "how does attention work?"
Rewritten: attention mechanism self-attention transformer architecture neural network focus

Original: "side effects of the drug"
Rewritten: adverse effects pharmacological side effects drug toxicity clinical outcomes

Now rewrite this query:
Original: "{query}"
Rewritten:"""

    return prompt