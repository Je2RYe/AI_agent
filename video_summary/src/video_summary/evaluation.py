import re
import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np


@lru_cache(maxsize=1)
def _get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    return [s.strip() for s in sentences if s.strip()]


def _chunk_sentences(sentences: List[str], chunk_size: int) -> List[str]:
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


class ObjectiveEvaluator:
    def __init__(self, model_name: Optional[str] = None) -> None:
        selected_model = model_name or os.getenv("EVAL_SBERT_MODEL", "all-MiniLM-L6-v2")
        self._model = _get_embedding_model(selected_model)

    def factual_consistency(self, summary: str, transcript: str, threshold: float = 0.5) -> Optional[float]:
        if not summary.strip() or not transcript.strip():
            return None

        from sklearn.metrics.pairwise import cosine_similarity

        summary_sentences = _split_sentences(summary)
        transcript_sentences = _split_sentences(transcript)
        if not summary_sentences or not transcript_sentences:
            return None

        transcript_chunks = _chunk_sentences(transcript_sentences, chunk_size=3)
        summary_embeds = self._model.encode(summary_sentences)
        transcript_embeds = self._model.encode(transcript_chunks)

        sim_matrix = cosine_similarity(summary_embeds, transcript_embeds)
        max_sims = sim_matrix.max(axis=1)
        return float((max_sims > threshold).mean())

    @staticmethod
    def _get_dynamic_top_k(n_sentences: int, cap: int = 50) -> int:
        """
        自适应关键句数量 — 平滑曲线，避免长视频关键句爆炸。

        设计思想（参考 Ragas / LlamaIndex 工业标准）：
          - 短文本 (≤100 句, ~3 min):  20% 灵活跟随
          - 中文本 (101-500 句, 3-15 min): 15% 适度压缩
          - 长文本 (>500 句, 15+ min): 5% 严格筛选，封顶 cap 句
        """
        if n_sentences <= 100:
            return max(10, int(n_sentences * 0.20))
        elif n_sentences <= 500:
            return max(20, int(n_sentences * 0.15))
        else:
            return max(30, min(int(n_sentences * 0.05), cap))

    def completeness(self, summary: str, transcript: str, threshold: float = None) -> Optional[float]:
        """
        多粒度语义覆盖 (Multi-Granular Semantic Coverage)。

        算法：Sentence-level Max-Recall + 双向验证
          1. 动态 Top-K 选出 transcript 的关键句（TF-IDF 排序）
          2. 将 summary 拆为句子级向量
          3. 构建 (N_summary × M_key) 相似度矩阵
          4. Recall:    每个关键句被任意 summary 句子覆盖的比例
             Precision: 每个 summary 句子有 transcript 支撑的比例
          5. 加权组合（Recall 为主，防信息遗漏）
        """
        if not summary.strip() or not transcript.strip():
            return None

        from config import COMPLETENESS_CONFIG
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        cfg = COMPLETENESS_CONFIG
        if threshold is None:
            threshold = cfg["threshold"]

        # ---- Step 1: 动态 Top-K 关键句 ----
        transcript_sentences = _split_sentences(transcript)
        if not transcript_sentences:
            return None

        top_k = self._get_dynamic_top_k(
            len(transcript_sentences), cap=cfg["max_key_sentences"]
        )

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(transcript_sentences)
        importance_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        key_indices = np.argsort(importance_scores)[-top_k:]
        key_sentences = [transcript_sentences[i] for i in key_indices]

        # ---- Step 2: 句子级编码 ----
        summary_sentences = _split_sentences(summary)
        if not summary_sentences:
            return 0.0

        key_embeds = self._model.encode(key_sentences)          # (M, dim)
        summary_embeds = self._model.encode(summary_sentences)  # (N, dim)

        # ---- Step 3: N × M 相似度矩阵 ----
        sim_matrix = cosine_similarity(summary_embeds, key_embeds)  # (N, M)

        # ---- Step 4: 双向验证 ----
        # Recall: 每个关键句的最大匹配度 → 被覆盖比例
        max_sim_per_key = sim_matrix.max(axis=0)                    # (M,)
        recall = float((max_sim_per_key > threshold).mean())

        # Precision: 每个 summary 句子的最大匹配度 → 有据比例
        max_sim_per_summary = sim_matrix.max(axis=1)                # (N,)
        precision = float((max_sim_per_summary > threshold).mean())

        # ---- Step 5: 加权组合（Recall 为主） ----
        w_r = cfg["recall_weight"]
        w_p = cfg["precision_weight"]
        score = w_r * recall + w_p * precision
        return float(score)

    def coherence(self, summary: str) -> Optional[float]:
        if not summary.strip():
            return None

        from sklearn.metrics.pairwise import cosine_similarity

        sentences = _split_sentences(summary)
        if len(sentences) < 2:
            return 1.0

        embeddings = self._model.encode(sentences)
        coherence_scores = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            coherence_scores.append(sim)

        avg_coherence = float(np.mean(coherence_scores))
        return float((avg_coherence + 1) / 2)


def compute_objective_metrics(summary: str, transcript: str) -> Dict[str, Optional[float]]:
    evaluator = ObjectiveEvaluator()
    return {
        "factual_consistency": evaluator.factual_consistency(summary, transcript),
        "completeness": evaluator.completeness(summary, transcript),
        "coherence": evaluator.coherence(summary),
    }


def compute_answer_grounding(
    answer: str,
    retrieved_context: str,
    model_name: Optional[str] = None,
) -> Optional[float]:
    """
    答案锚定分数 (Answer Grounding Score)。

    量化"回答有多少来自检索的原文内容"，即 RAG 上下文被实际利用的程度。

    算法：SBERT cosine_similarity(answer_embedding, context_embedding)
      - 复用评估模块已加载的模型（零额外开销）
      - 范围 [0, 1]；归一化后约 [0.3, 0.9] 为常见区间

    解读：
      > 0.65  高锚定：回答明确来源于检索内容  ✅
      0.45~0.65 中等：部分利用                ⚠️
      < 0.45  低锚定：回答几乎未使用检索内容   ❌

    复合解读（与 retrieval_confidence 结合）：
      高置信 + 高锚定 → RAG 完整有效
      高置信 + 低锚定 → 检索到但 LLM 忽略了（提示词问题）
      低置信 + 任意   → 问题本身不适合 RAG

    Args:
        answer:            Chat Agent 的完整回答文本。
        retrieved_context: run_rag_pipeline 或 retriever.retrieve() 拼出的 context 字符串。
        model_name:        SBERT 模型名（默认复用全局 cache）。

    Returns:
        float in [0,1]，或 None（输入为空时）。
    """
    if not answer.strip() or not retrieved_context.strip():
        return None

    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    selected_model = model_name or "all-MiniLM-L6-v2"
    model = _get_embedding_model(selected_model)

    a_emb = model.encode([answer])
    c_emb = model.encode([retrieved_context])

    raw_sim = float(sk_cosine(a_emb, c_emb)[0][0])
    # 线性归一化 [-1,1] → [0,1]（SBERT 输出通常已在 [-1,1]）
    return float((raw_sim + 1.0) / 2.0)
