from typing import List, Dict, Any
import asyncio
import time
class ChatService:

    def __init__(
        self,
        knowledge_client,
        llm_client,
        logger=None,
        max_context_chars: int = 6000,
        llm_timeout: float = 15.0
    ):
        self.knowledge = knowledge_client
        self.llm = llm_client
        self.logger = logger
        self.max_context_chars = max_context_chars
        self.llm_timeout = llm_timeout


    async def ask(
        self,
        user_id: int,
        query: str,
        k: int = 5
    ) -> Dict[str, Any]:

        try:
            if not query or not query.strip():
                return {"answer": "Query cannot be empty.", "sources": []}

            original_query = query.strip()

            # 1️⃣ Clarification detection
            clarification = await self._detect_and_generate_clarification(original_query)
            if clarification:
                return {
                    "answer": clarification,
                    "sources": [],
                    "needs_clarification": True
                }

            #  Rewrite for retrieval
            refined_query = await self._rewrite_query(original_query)

            if self.logger:
                self.logger.info(f"[QUERY] Original: {original_query}")
                self.logger.info(f"[QUERY] Refined: {refined_query}")

            #  Vector retrieval
            chunks, vector_time = await self.knowledge.search(
                user_id=user_id,
                query=refined_query,
                k=k * 2  # retrieve more for reranking
            )

            if not chunks:
                return {
                    "answer": "I could not find relevant information.",
                    "sources": []
                }

            #  Rerank using LLM
            time0=time.time()
            reranked_chunks = await self._rerank_chunks(original_query, chunks)
            reranking_time=time.time()-time0

            # Keep top-k after reranking
            top_chunks = reranked_chunks[:k]

            #  Build context
            context = self._build_context(top_chunks)

            if not context.strip():
                return {
                    "answer": "I could not find relevant information.",
                    "sources": []
                }

            #  Final answer generation
            prompt = self._build_prompt(context, original_query)
            time1=time.time()
            answer = await asyncio.wait_for(
                self.llm.generate(prompt),
                timeout=self.llm_timeout
            )
            generation_time=time.time()-time1

            return {
                "answer": answer.strip(),
                "sources": [
                    {
                        "chunk_id": c["chunk_id"],
                        "pdf_id": c["pdf_id"],
                        "page_no": c["page_no"],
                        "text": c["text"],
                        "confidence": c.get("score"),
                        "rerank_score": c.get("rerank_score"),
                        "embedding_time":c.get("embedding_time")
                    }
                    for c in top_chunks
                ],
                "vector_db_time": vector_time,
                "reranking_time":reranking_time,
                "generation_time":generation_time
            }

        except asyncio.TimeoutError:
            return {
                "answer": "The request timed out. Please try again.",
                "sources": []
            }

        except Exception as e:
            if self.logger:
                self.logger.exception(f"[CHAT SERVICE ERROR] {e}")
            return {
                "answer": "Internal error occurred.",
                "sources": []
            }

   

    async def _detect_and_generate_clarification(self, query: str) -> str | None:

        if len(query.split()) >= 6:
            return None

        clarification_prompt = f"""
Determine whether this query is too vague for document-based QA.

If vague:
Ask ONE concise clarification question.
If clear:
Respond exactly with: CLEAR

Query:
{query}
"""

        try:
            response = await asyncio.wait_for(
                self.llm.generate(clarification_prompt),
                timeout=5.0
            )
            response = response.strip()

            if response.upper() == "CLEAR":
                return None

            return response

        except Exception:
            return None

   

    async def _rewrite_query(self, query: str) -> str:

        rewrite_prompt = f"""
Rewrite this into a concise search query for vector retrieval.
Do NOT answer it.
Maximum 25 words.

Query:
{query}
"""

        try:
            rewritten = await asyncio.wait_for(
                self.llm.generate(rewrite_prompt),
                timeout=5.0
            )

            rewritten = rewritten.strip()

            if not rewritten or len(rewritten) > 200:
                return query

            return rewritten

        except Exception:
            return query

   

    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict]
    ) -> List[Dict]:

        chunk_blocks = []
        for idx, c in enumerate(chunks):
            chunk_blocks.append(
                f"Chunk {idx}:\n{c['text'][:800]}\n"
            )

        rerank_prompt = f"""
You are a relevance scoring system.

Score each chunk from 0 to 100 based on relevance to the question.

Return strictly:

Chunk 0: <score>
Chunk 1: <score>
...

Question:
{query}

Chunks:
{chr(10).join(chunk_blocks)}
"""

        try:
            response = await asyncio.wait_for(
                self.llm.generate(rerank_prompt),
                timeout=10.0
            )

            scores = self._parse_rerank_scores(response, len(chunks))

            for i, c in enumerate(chunks):
                c["rerank_score"] = scores.get(i, 0)

            return sorted(
                chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )

        except Exception:
            return sorted(
                chunks,
                key=lambda x: x.get("score", 0),
                reverse=True
            )

    def _parse_rerank_scores(
        self,
        response: str,
        expected_chunks: int
    ) -> Dict[int, int]:

        scores = {}

        for line in response.splitlines():
            line = line.strip()
            if line.startswith("Chunk"):
                try:
                    parts = line.split(":")
                    idx = int(parts[0].split()[1])
                    score = int(parts[1].strip())
                    scores[idx] = score
                except Exception:
                    continue

        for i in range(expected_chunks):
            if i not in scores:
                scores[i] = 0

        return scores

   

    def _build_context(self, chunks: List[Dict]) -> str:

        combined = []
        total_chars = 0

        for c in chunks:
            block = (
                f"[PDF: {c['pdf_id']} | Page: {c['page_no']}]\n"
                f"{c['text']}\n"
            )

            if total_chars + len(block) > self.max_context_chars:
                break

            combined.append(block)
            total_chars += len(block)

        return "\n".join(combined)

   

    def _build_prompt(self, context: str, question: str) -> str:

        return f"""
You are a helpful assistant.

Follow these steps:
1. Identify any numbers related to units or residences.
2. If total residences are mentioned and no other unit types are specified,
   treat that as the total number of units.
3. Provide the answer clearly.

Only say "I do not have enough information." if no relevant number exists.

Context:
{context}

Question:
{question}

Answer:
"""
