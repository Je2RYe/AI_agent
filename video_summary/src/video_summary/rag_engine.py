"""
rag_engine.py — Business-Aware Hybrid RAG Engine

高内聚门面模块，专为 *视频摘要* 场景量身定制。
弃用 LangChain 黑盒，纯 Python 原生实现以下四大支柱：

  1. RecursiveChunker   — 递归退坡切分 (\\n\\n → \\n → . → 空格)，带 overlap
  2. HybridRetriever    — Dense (SBERT+FAISS) + Sparse (BM25) + RRF 倒数秩融合
  3. ContextOrchestrator— Head-Tail Pinning (首尾固钉) + Temporal Re-ordering (时序重排)
  4. run_rag_pipeline()  — 一站式入口供 crew.py 调用

参考：DESIGN.md §14 (v1.4)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import EVAL_SBERT_DEFAULT, RAG_CONFIG, MAP_REDUCE_SEGMENT_TOKENS


# ═══════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """表示 transcript 中的一个语义块。"""
    text: str
    index: int                         # 在原文中的顺序（用于 Temporal Re-ordering）
    token_estimate: int
    metadata: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 1. RecursiveChunker — 递归退坡分块器
# ═══════════════════════════════════════════════════════════════════════════

class RecursiveChunker:
    """
    递归退坡分块 (Recursive Back-off Splitting)。

    分割优先级：
      Level-0: \\n\\n (段落)
      Level-1: \\n   (行)
      Level-2: .     (句号)
      Level-3: ' '   (空格 / 兜底)

    每级若产出的碎片仍超 chunk_size，递归用下一级继续拆；
    最终对所有碎片做 sliding-window 聚合 + overlap 保留。
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int = RAG_CONFIG["chunk_size"],
        chunk_overlap: int = RAG_CONFIG["chunk_overlap"],
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---------- public API ----------

    def chunk_transcript(self, transcript: str) -> List[Chunk]:
        """将 transcript 切分为带 overlap 的 Chunk 列表。"""
        raw_pieces = self._recursive_split(transcript, level=0)
        chunks = self._merge_with_overlap(raw_pieces)
        return chunks

    # ---------- 递归拆分 ----------

    def _recursive_split(self, text: str, level: int) -> List[str]:
        """按当前级别分隔符拆分；若碎片仍超长则递归下一级。"""
        if level >= len(self.SEPARATORS):
            # 兜底：硬截断
            return self._hard_split(text)

        sep = self.SEPARATORS[level]
        pieces = text.split(sep)

        result: List[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            est = self._estimate_tokens(piece)
            if est <= self.chunk_size:
                result.append(piece)
            else:
                # 碎片仍超长 → 递归下一级
                result.extend(self._recursive_split(piece, level + 1))
        return result

    def _hard_split(self, text: str) -> List[str]:
        """最终兜底：按 word 级别硬截断。"""
        words = text.split()
        pieces: List[str] = []
        current: List[str] = []
        count = 0
        for w in words:
            t = self._estimate_tokens(w)
            if count + t > self.chunk_size and current:
                pieces.append(" ".join(current))
                current = []
                count = 0
            current.append(w)
            count += t
        if current:
            pieces.append(" ".join(current))
        return pieces

    # ---------- Sliding-Window 聚合 + Overlap ----------

    def _merge_with_overlap(self, pieces: List[str]) -> List[Chunk]:
        """
        将碎片聚合为 ≤ chunk_size 的 Chunk，并在相邻 Chunk 间
        保持 overlap tokens 的重叠。
        """
        if not pieces:
            return []

        chunks: List[Chunk] = []
        buffer: List[str] = []
        buf_tokens = 0

        for piece in pieces:
            est = self._estimate_tokens(piece)
            if buf_tokens + est > self.chunk_size and buffer:
                # 产出一个 chunk
                chunk_text = " ".join(buffer)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    token_estimate=buf_tokens,
                ))
                # 保留 overlap 尾部
                buffer, buf_tokens = self._retain_overlap(buffer)

            buffer.append(piece)
            buf_tokens += est

        # 剩余尾巴
        if buffer:
            chunks.append(Chunk(
                text=" ".join(buffer),
                index=len(chunks),
                token_estimate=buf_tokens,
            ))

        # 补充 metadata: position_pct
        total = max(len(chunks), 1)
        for c in chunks:
            c.metadata["position_pct"] = round(c.index / total, 2)

        return chunks

    def _retain_overlap(self, buffer: List[str]) -> Tuple[List[str], int]:
        """从 buffer 尾部保留约 overlap tokens 的碎片。"""
        retained: List[str] = []
        tokens = 0
        for piece in reversed(buffer):
            t = self._estimate_tokens(piece)
            if tokens + t > self.chunk_overlap:
                break
            retained.insert(0, piece)
            tokens += t
        return retained, tokens

    # ---------- token 估算 ----------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗估 token 数（1 word ≈ 1.3 tokens）。"""
        return max(1, int(len(text.split()) * 1.3))


# ═══════════════════════════════════════════════════════════════════════════
# 2. HybridRetriever — Dense + Sparse + RRF
# ═══════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    双路召回 + Reciprocal Rank Fusion (RRF)。

    Dense  路 (语义): SBERT → FAISS IndexFlatIP
    Sparse 路 (关键词): BM25Okapi (rank_bm25)
    融合算法: RRF  score(d) = Σ 1 / (K + rank_i(d))
    """

    def __init__(
        self,
        chunks: List[Chunk],
        sbert_model_name: str = EVAL_SBERT_DEFAULT,
        rrf_k: int = RAG_CONFIG["rrf_k"],
    ):
        self.chunks = chunks
        self.rrf_k = rrf_k

        # ---- 延迟加载重量级依赖 ----
        self._sbert_model = None
        self._sbert_model_name = sbert_model_name
        self._faiss_index = None
        self._bm25 = None

        self._build_indices()

    # ---------- 构建索引 ----------

    def _build_indices(self) -> None:
        """同时构建 Dense (FAISS) 和 Sparse (BM25) 索引。"""
        texts = [c.text for c in self.chunks]

        # ---- Dense: SBERT + FAISS ----
        import faiss
        from sentence_transformers import SentenceTransformer

        self._sbert_model = SentenceTransformer(self._sbert_model_name)
        embeddings = self._sbert_model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)

        # ---- Sparse: BM25 ----
        from rank_bm25 import BM25Okapi

        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    # ---------- 检索 ----------

    def retrieve(self, query: str, top_k: int = RAG_CONFIG["top_k_retrieval"]) -> List[Tuple[Chunk, float]]:
        """
        混合检索：双路召回 → RRF 融合 → 返回 top_k (Chunk, score)。
        """
        if not self.chunks:
            return []

        n = len(self.chunks)

        # ---- Dense 排序 ----
        import faiss
        q_emb = self._sbert_model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        dense_scores, dense_ids = self._faiss_index.search(q_emb, n)
        dense_rank: Dict[int, int] = {}
        for rank, idx in enumerate(dense_ids[0]):
            if idx >= 0:
                dense_rank[int(idx)] = rank  # rank 0 = best

        # ---- Sparse 排序 ----
        bm25_scores = self._bm25.get_scores(query.lower().split())
        sparse_order = np.argsort(-bm25_scores)   # 高分在前
        sparse_rank: Dict[int, int] = {int(idx): rank for rank, idx in enumerate(sparse_order)}

        # ---- RRF 融合 ----
        rrf_scores: Dict[int, float] = {}
        for idx in range(n):
            score = 0.0
            if idx in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[idx])
            if idx in sparse_rank:
                score += 1.0 / (self.rrf_k + sparse_rank[idx])
            rrf_scores[idx] = score

        # 按 RRF 分数降序
        sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]

        return [(self.chunks[i], rrf_scores[i]) for i in top_indices]


# ═══════════════════════════════════════════════════════════════════════════
# 3. ContextOrchestrator — Head-Tail Pinning + Temporal Re-ordering
# ═══════════════════════════════════════════════════════════════════════════

class ContextOrchestrator:
    """
    业务感知上下文编排器。

    策略：
      1. Head-Tail Pinning: 强制注入首 chunk (视频引言) 和尾 chunk (视频结论)
      2. Temporal Re-ordering: 将检索到的碎片按原文 chunk.index 时序排列
      3. Token Budget: 控制总 context 不超过 max_context_tokens
    """

    def __init__(self, max_context_tokens: int = RAG_CONFIG["max_context_tokens"]):
        self.max_context_tokens = max_context_tokens

    def build_context(
        self,
        all_chunks: List[Chunk],
        retrieved: List[Tuple[Chunk, float]],
    ) -> Dict:
        """
        编排最终上下文。

        Returns:
            {
                "context": str,           # 拼装后的文本
                "n_chunks_used": int,
                "n_chunks_total": int,
                "rag_coverage_pct": float, # 使用/总 chunk 比例
                "pinned_head": bool,
                "pinned_tail": bool,
            }
        """
        if not all_chunks:
            return {
                "context": "",
                "n_chunks_used": 0,
                "n_chunks_total": 0,
                "rag_coverage_pct": 0.0,
                "pinned_head": False,
                "pinned_tail": False,
            }

        head_chunk = all_chunks[0]
        tail_chunk = all_chunks[-1] if len(all_chunks) > 1 else None

        # 收集被检索到的 chunks (去重)
        retrieved_indices = {chunk.index for chunk, _ in retrieved}
        selected_map: Dict[int, Chunk] = {}

        # 先固钉 head / tail
        pinned_head = True
        selected_map[head_chunk.index] = head_chunk
        pinned_tail = False
        if tail_chunk and tail_chunk.index != head_chunk.index:
            selected_map[tail_chunk.index] = tail_chunk
            pinned_tail = True

        # 加入检索结果
        for chunk, _ in retrieved:
            if chunk.index not in selected_map:
                selected_map[chunk.index] = chunk

        # Temporal Re-ordering: 按原文顺序排序
        ordered = sorted(selected_map.values(), key=lambda c: c.index)

        # Token budget 截断
        final_chunks: List[Chunk] = []
        token_count = 0
        for chunk in ordered:
            if token_count + chunk.token_estimate > self.max_context_tokens:
                # 首尾即使超 budget 也保留
                if chunk.index == head_chunk.index or (tail_chunk and chunk.index == tail_chunk.index):
                    final_chunks.append(chunk)
                    token_count += chunk.token_estimate
                continue
            final_chunks.append(chunk)
            token_count += chunk.token_estimate

        # 最终排序（以防截断打乱顺序）
        final_chunks.sort(key=lambda c: c.index)

        context_text = "\n\n---\n\n".join(c.text for c in final_chunks)

        return {
            "context": context_text,
            "n_chunks_used": len(final_chunks),
            "n_chunks_total": len(all_chunks),
            "rag_coverage_pct": round(len(final_chunks) / max(len(all_chunks), 1), 4),
            "pinned_head": pinned_head,
            "pinned_tail": pinned_tail,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Pipeline 入口
# ═══════════════════════════════════════════════════════════════════════════

def run_rag_pipeline(transcript: str, user_query: str = "") -> Dict:
    """
    一站式 RAG Pipeline 入口。

    Args:
        transcript: 完整转录文本。
        user_query: 用户查询 / 摘要指令（可选；空字符串时自动生成内部查询）。

    Returns:
        {
            "rag_used": True,
            "context": str,          # 编排后传给 Summarizer 的精选上下文
            "n_chunks_used": int,
            "n_chunks_total": int,
            "rag_coverage_pct": float,
            "pinned_head": bool,
            "pinned_tail": bool,
        }
    """
    # ---- Step 1: Recursive Chunking ----
    chunker = RecursiveChunker()
    chunks = chunker.chunk_transcript(transcript)

    if len(chunks) <= 2:
        # 太短，不需要 RAG
        return {
            "rag_used": False,
            "context": transcript,
            "n_chunks_used": len(chunks),
            "n_chunks_total": len(chunks),
            "rag_coverage_pct": 1.0,
            "pinned_head": False,
            "pinned_tail": False,
        }

    # ---- Step 2: Hybrid Retrieval ----
    if not user_query:
        # 自动生成内部查询：提取前 200 字 + "summarize key points"
        user_query = f"Summarize the key points: {transcript[:300]}"

    retriever = HybridRetriever(chunks)
    retrieved = retriever.retrieve(user_query, top_k=RAG_CONFIG["top_k_retrieval"])

    # ---- Step 3: Context Orchestration ----
    orchestrator = ContextOrchestrator()
    result = orchestrator.build_context(chunks, retrieved)

    result["rag_used"] = True
    return result


def should_use_rag(transcript: str) -> bool:
    """判断是否需要启用 RAG（基于估算 token 数）。"""
    est_tokens = len(transcript.split()) * 1.3
    return est_tokens > RAG_CONFIG["rag_threshold_tokens"]


def build_enhanced_prompt(rag_result: Dict) -> str:
    """
    将 RAG 结果包装为 Summarizer Agent 的增强 Prompt。

    Args:
        rag_result: run_rag_pipeline() 的返回值。

    Returns:
        增强后的 prompt 文本。
    """
    if not rag_result.get("rag_used"):
        return rag_result.get("context", "")

    coverage = rag_result["rag_coverage_pct"]
    n_used = rag_result["n_chunks_used"]
    n_total = rag_result["n_chunks_total"]

    return f"""## 视频全文统计
- 总段落数: {n_total}
- 已检索关键段落数: {n_used}（覆盖率 {coverage:.0%}）
- 首尾固钉: Head={'✓' if rag_result['pinned_head'] else '✗'}, Tail={'✓' if rag_result['pinned_tail'] else '✗'}

## 关键段落（按原文时序排列，已通过 Hybrid Search + RRF 融合筛选）

{rag_result['context']}

## 指令
基于以上关键段落生成结构化摘要。注意：
1. 这些段落是从完整视频中通过 Dense+Sparse 混合检索筛选的最重要部分
2. 段落已按视频原始顺序排列，请保持叙事逻辑
3. 首段和尾段分别是视频的引言和结论，请确保摘要覆盖这些锚点
4. 若段落间有逻辑关联，请在摘要中体现
"""


# ═══════════════════════════════════════════════════════════════════════════
# 5. Map-Reduce 辅助 — 超长视频分段工具
# ═══════════════════════════════════════════════════════════════════════════

def split_for_map_reduce(
    transcript: str,
    segment_size_tokens: int = MAP_REDUCE_SEGMENT_TOKENS,
) -> List[str]:
    """
    将超长 transcript 拆分为若干段，供 Map-Reduce 摘要使用。

    设计原则：
    - 优先在段落边界（\\n\\n）切分，保留语义完整性
    - 若段落本身超过 segment_size，降级到行级别切分
    - 每段约 segment_size_tokens token（1 word ≈ 1.3 tokens）

    这是摘要生成路径的分段工具，与 Chat QA 用的 RecursiveChunker 不同：
    - Map-Reduce 段：粒度大（~20K tokens），追求覆盖率
    - Chat QA chunk：粒度小（~512 tokens），追求检索精度

    Args:
        transcript: 完整转录文本。
        segment_size_tokens: 每段目标 token 数（默认 MAP_REDUCE_SEGMENT_TOKENS）。

    Returns:
        List[str]，每段为可独立摘要的文本块。
    """
    words_per_segment = int(segment_size_tokens / 1.3)

    # 尝试段落级切分
    paragraphs = transcript.split("\n\n")
    segments: List[str] = []
    current_words: List[str] = []
    current_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = para.split()
        para_count = len(para_words)

        # 单段落本身超过限制 → 降级到行级切分
        if para_count > words_per_segment:
            # 先把已有 buffer 产出
            if current_words:
                segments.append(" ".join(current_words))
                current_words = []
                current_count = 0
            # 行级切分该段落
            lines = para.split("\n")
            for line in lines:
                line_words = line.split()
                if current_count + len(line_words) > words_per_segment and current_words:
                    segments.append(" ".join(current_words))
                    current_words = []
                    current_count = 0
                current_words.extend(line_words)
                current_count += len(line_words)
            continue

        if current_count + para_count > words_per_segment and current_words:
            segments.append(" ".join(current_words))
            current_words = []
            current_count = 0

        current_words.extend(para_words)
        current_count += para_count

    if current_words:
        segments.append(" ".join(current_words))

    return segments if segments else [transcript]
