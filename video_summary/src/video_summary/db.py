"""
db.py — DuckDB 持久化层（LLMOps 可观测性基础）

职责：
  1. 初始化 DuckDB 数据库 + Schema（reports / eval_runs）
  2. 插入 report / eval_run 记录
  3. 查询历史（支持 soft delete 过滤）
  4. Soft delete 操作
  5. 聚合查询（趋势、模型分布、成本累计）

参考：DESIGN.md §16
"""

import os
import uuid
from datetime import datetime
from typing import Optional

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DB_PATH = "video_summary.duckdb"

# ---------------------------------------------------------------------------
# 连接 & Schema
# ---------------------------------------------------------------------------

def get_conn() -> duckdb.DuckDBPyConnection:
    """获取 DuckDB 连接，自动初始化 schema。"""
    conn = duckdb.connect(DB_PATH)
    _init_tables(conn)
    return conn


def _init_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """建表（幂等）。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id       VARCHAR PRIMARY KEY,
            created_at      TIMESTAMP NOT NULL,
            model_used      VARCHAR NOT NULL,
            latency_sec     DOUBLE,
            total_tokens    INTEGER,
            cost_usd        DOUBLE,
            report_path     VARCHAR,
            transcript_path VARCHAR,
            summary_preview VARCHAR,
            is_deleted      BOOLEAN DEFAULT FALSE,
            deleted_at      TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            eval_id                  VARCHAR PRIMARY KEY,
            report_id                VARCHAR NOT NULL,
            created_at               TIMESTAMP NOT NULL,
            model_used               VARCHAR NOT NULL,
            factual_consistency      DOUBLE,
            completeness             DOUBLE,
            coherence                DOUBLE,
            objective_weighted_score DOUBLE,
            chat_depth               INTEGER DEFAULT 0,
            re_asks                  INTEGER DEFAULT 0,
            copy_clicks              INTEGER DEFAULT 0,
            expand_clicks            INTEGER DEFAULT 0,
            time_spent_sec           DOUBLE DEFAULT 0.0,
            is_feedback_collected    BOOLEAN DEFAULT FALSE,
            user_feedback_score      DOUBLE,
            rag_used                 BOOLEAN DEFAULT FALSE,
            rag_total_chunks         INTEGER,
            rag_used_chunks          INTEGER,
            rag_coverage_pct         DOUBLE
        )
    """)
    # 向后兼容：为旧库增加 RAG 列（ALTER 幂等处理）
    for col_def in [
        ("rag_used", "BOOLEAN DEFAULT FALSE"),
        ("rag_total_chunks", "INTEGER"),
        ("rag_used_chunks", "INTEGER"),
        ("rag_coverage_pct", "DOUBLE"),
        # Phase 2B：Judge-LLM 副轨
        ("judge_score", "DOUBLE"),
        ("judge_confidence", "DOUBLE"),
        ("judge_flags", "VARCHAR"),   # JSON 字符串: {hallucination_risk, contradiction_hint, ...}
        ("judge_reasoning", "VARCHAR"),
        ("judge_model", "VARCHAR"),
        ("judge_latency_sec", "DOUBLE"),
    ]:
        try:
            conn.execute(f"ALTER TABLE eval_runs ADD COLUMN {col_def[0]} {col_def[1]}")
        except Exception:
            pass  # 列已存在

    # chat_logs 表：每一轮 Chat QA 的 RAG 可观测性记录
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            log_id                  VARCHAR PRIMARY KEY,
            session_id              VARCHAR,          -- Streamlit session 标识
            created_at              TIMESTAMP NOT NULL,
            report_id               VARCHAR,          -- 关联的摘要报告
            question                VARCHAR,
            answer_preview          VARCHAR,          -- 回答前 200 字
            rag_used                BOOLEAN DEFAULT FALSE,
            n_chunks_retrieved      INTEGER,          -- 实际检索到的 chunk 数
            retrieval_confidence    DOUBLE,           -- top-K 平均 RRF 分数 (代理"找到了多少")
            answer_grounding_score  DOUBLE,           -- SBERT cos_sim(answer, retrieved_ctx) (代理"用了多少")
            model_used              VARCHAR           -- chat 使用的模型
        )
    """)


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------

def generate_report_id() -> str:
    """生成 UUID report_id。"""
    return str(uuid.uuid4())


def insert_report(
    conn: duckdb.DuckDBPyConnection,
    report_id: str,
    model_used: str,
    latency_sec: float,
    total_tokens: int,
    cost_usd: float,
    report_path: str = "",
    transcript_path: str = "",
    summary_preview: str = "",
) -> None:
    """插入一条报告记录。"""
    conn.execute(
        """
        INSERT INTO reports (
            report_id, created_at, model_used, latency_sec,
            total_tokens, cost_usd, report_path, transcript_path,
            summary_preview, is_deleted, deleted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, NULL)
        """,
        [
            report_id,
            datetime.now(),
            model_used,
            round(latency_sec, 2),
            total_tokens,
            round(cost_usd, 6),
            report_path,
            transcript_path,
            summary_preview[:200] if summary_preview else "",
        ],
    )


def insert_eval(
    conn: duckdb.DuckDBPyConnection,
    report_id: str,
    model_used: str,
    factual_consistency: Optional[float] = None,
    completeness: Optional[float] = None,
    coherence: Optional[float] = None,
    objective_weighted_score: Optional[float] = None,
    chat_depth: int = 0,
    re_asks: int = 0,
    copy_clicks: int = 0,
    expand_clicks: int = 0,
    time_spent_sec: float = 0.0,
    is_feedback_collected: bool = False,
    user_feedback_score: Optional[float] = None,
    rag_used: bool = False,
    rag_total_chunks: Optional[int] = None,
    rag_used_chunks: Optional[int] = None,
    rag_coverage_pct: Optional[float] = None,
) -> str:
    """插入一条评估记录，返回 eval_id。"""
    eval_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO eval_runs (
            eval_id, report_id, created_at, model_used,
            factual_consistency, completeness, coherence,
            objective_weighted_score,
            chat_depth, re_asks, copy_clicks, expand_clicks,
            time_spent_sec, is_feedback_collected, user_feedback_score,
            rag_used, rag_total_chunks, rag_used_chunks, rag_coverage_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            eval_id,
            report_id,
            datetime.now(),
            model_used,
            factual_consistency,
            completeness,
            coherence,
            objective_weighted_score,
            chat_depth,
            re_asks,
            copy_clicks,
            expand_clicks,
            round(time_spent_sec, 1),
            is_feedback_collected,
            user_feedback_score,
            rag_used,
            rag_total_chunks,
            rag_used_chunks,
            rag_coverage_pct,
        ],
    )
    return eval_id


def update_eval_judge(
    conn: duckdb.DuckDBPyConnection,
    report_id: str,
    judge_score: Optional[float],
    judge_confidence: Optional[float],
    judge_flags: str = "{}",
    judge_reasoning: str = "",
    judge_model: str = "",
    judge_latency_sec: float = 0.0,
) -> None:
    """
    Phase 2B：将 Judge-LLM 评分结果异步回写到对应 eval_run 行。

    由 judge_llm.py 的后台线程在 Judge 完成后调用；
    通过 report_id 匹配，不阻塞主链路。
    """
    conn.execute(
        """
        UPDATE eval_runs
        SET judge_score        = ?,
            judge_confidence   = ?,
            judge_flags        = ?,
            judge_reasoning    = ?,
            judge_model        = ?,
            judge_latency_sec  = ?
        WHERE report_id = ?
        """,
        [
            round(judge_score, 4) if judge_score is not None else None,
            round(judge_confidence, 4) if judge_confidence is not None else None,
            judge_flags[:2000] if judge_flags else "{}",
            judge_reasoning[:1000] if judge_reasoning else "",
            judge_model,
            round(judge_latency_sec, 2),
            report_id,
        ],
    )


def insert_chat_log(
    conn: duckdb.DuckDBPyConnection,
    question: str,
    answer: str,
    rag_used: bool,
    n_chunks_retrieved: int = 0,
    retrieval_confidence: Optional[float] = None,
    answer_grounding_score: Optional[float] = None,
    session_id: str = "",
    report_id: str = "",
    model_used: str = "",
) -> str:
    """
    记录一轮 Chat QA 的 RAG 可观测性指标。

    核心指标语义：
      retrieval_confidence  — avg RRF score of top-K chunks:
                              衡量"检索到了多少相关内容"（与问题无关 = 低置信）
      answer_grounding_score — SBERT cosine_sim(answer, retrieved_context):
                              衡量"回答有多少来自检索内容"（忽略上下文 = 低分）

    两指标组合解读：
      高置信 + 高锚定 → RAG 有效（检索到 + 实际使用了）   ✅
      高置信 + 低锚定 → 检索到但 LLM 未使用（prompt 问题）⚠️
      低置信 + 任意   → 问题类型不适合 RAG               ℹ️
      rag_used=False  → 基线（无 RAG 的回答，grounding=None）
    """
    log_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO chat_logs (
            log_id, session_id, created_at, report_id,
            question, answer_preview, rag_used,
            n_chunks_retrieved, retrieval_confidence,
            answer_grounding_score, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            log_id,
            session_id,
            datetime.now(),
            report_id,
            question[:500],
            answer[:200],
            rag_used,
            n_chunks_retrieved,
            round(retrieval_confidence, 4) if retrieval_confidence is not None else None,
            round(answer_grounding_score, 4) if answer_grounding_score is not None else None,
            model_used,
        ],
    )
    return log_id


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def get_reports(
    conn: duckdb.DuckDBPyConnection,
    include_deleted: bool = False,
) -> pd.DataFrame:
    """查询报告列表（默认排除已删除），按时间倒序。"""
    if include_deleted:
        query = "SELECT * FROM reports ORDER BY created_at DESC"
    else:
        query = "SELECT * FROM reports WHERE is_deleted = FALSE ORDER BY created_at DESC"
    return conn.execute(query).fetchdf()


def get_evals(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """查询全部评估记录（含已删除报告的评估），按时间倒序。"""
    return conn.execute(
        "SELECT * FROM eval_runs ORDER BY created_at DESC"
    ).fetchdf()


def get_evals_with_report(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    JOIN 查询：eval_runs LEFT JOIN reports，
    包含 is_deleted 标记供 UI 区分。
    """
    return conn.execute("""
        SELECT
            e.eval_id,
            e.report_id,
            e.created_at,
            e.model_used,
            e.factual_consistency,
            e.completeness,
            e.coherence,
            e.objective_weighted_score,
            e.chat_depth,
            e.time_spent_sec,
            e.user_feedback_score,
            r.report_path,
            r.summary_preview,
            COALESCE(r.is_deleted, FALSE) AS report_deleted
        FROM eval_runs e
        LEFT JOIN reports r ON e.report_id = r.report_id
        ORDER BY e.created_at DESC
    """).fetchdf()


def get_report_by_id(
    conn: duckdb.DuckDBPyConnection,
    report_id: str,
) -> Optional[dict]:
    """按 report_id 查询单条报告。"""
    df = conn.execute(
        "SELECT * FROM reports WHERE report_id = ?", [report_id]
    ).fetchdf()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Soft Delete
# ---------------------------------------------------------------------------

def soft_delete_report(
    conn: duckdb.DuckDBPyConnection,
    report_id: str,
) -> bool:
    """软删除报告（标记 is_deleted=TRUE），评估数据保留。"""
    conn.execute(
        """
        UPDATE reports
        SET is_deleted = TRUE, deleted_at = ?
        WHERE report_id = ?
        """,
        [datetime.now(), report_id],
    )
    return True


def restore_report(
    conn: duckdb.DuckDBPyConnection,
    report_id: str,
) -> bool:
    """恢复软删除的报告。"""
    conn.execute(
        """
        UPDATE reports
        SET is_deleted = FALSE, deleted_at = NULL
        WHERE report_id = ?
        """,
        [report_id],
    )
    return True


# ---------------------------------------------------------------------------
# 聚合查询（Dashboard 用）
# ---------------------------------------------------------------------------

def get_model_distribution(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """模型使用分布统计。"""
    return conn.execute("""
        SELECT model_used, COUNT(*) AS run_count,
               ROUND(AVG(objective_weighted_score), 3) AS avg_score,
               ROUND(SUM(COALESCE(e.cost_usd_calc, 0)), 4) AS total_cost
        FROM eval_runs e
        LEFT JOIN (
            SELECT report_id, cost_usd AS cost_usd_calc FROM reports
        ) r ON e.report_id = r.report_id
        GROUP BY model_used
        ORDER BY run_count DESC
    """).fetchdf()


def get_score_trend(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """按时间排列的 weighted score 趋势（含 judge_score 供对比）。"""
    return conn.execute("""
        SELECT
            created_at,
            model_used,
            factual_consistency,
            completeness,
            coherence,
            objective_weighted_score,
            judge_score,
            judge_confidence,
            judge_flags
        FROM eval_runs
        ORDER BY created_at ASC
    """).fetchdf()


def get_total_cost(conn: duckdb.DuckDBPyConnection) -> float:
    """累计总成本。"""
    result = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM reports"
    ).fetchone()
    return float(result[0]) if result else 0.0


def get_cost_quality_df(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Phase 2B Dashboard：Cost vs Quality 散点数据。
    JOIN reports + eval_runs，供 Plotly 散点图使用。
    """
    return conn.execute("""
        SELECT
            r.created_at,
            r.model_used,
            r.cost_usd,
            r.total_tokens,
            e.objective_weighted_score,
            e.factual_consistency,
            e.completeness,
            e.coherence,
            e.judge_score,
            CASE WHEN r.total_tokens > 0
                 THEN ROUND(e.objective_weighted_score / NULLIF(r.total_tokens, 0) * 1000, 6)
                 ELSE NULL END AS token_efficiency
        FROM reports r
        LEFT JOIN eval_runs e ON r.report_id = e.report_id
        WHERE r.is_deleted = FALSE
          AND e.objective_weighted_score IS NOT NULL
        ORDER BY r.created_at ASC
    """).fetchdf()


def get_chat_logs(
    conn: duckdb.DuckDBPyConnection,
    limit: int = 200,
) -> pd.DataFrame:
    """
    查询最近 N 条 chat QA 日志（按时间倒序）。
    包含 RAG 可观测性指标，供 Metrics 面板展示。
    """
    return conn.execute(
        """
        SELECT
            created_at,
            question,
            rag_used,
            n_chunks_retrieved,
            ROUND(retrieval_confidence, 4)   AS retrieval_confidence,
            ROUND(answer_grounding_score, 4) AS answer_grounding_score,
            model_used,
            answer_preview
        FROM chat_logs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        [limit],
    ).fetchdf()


def get_chat_rag_summary(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    生成 RAG A/B 对比摘要：
      - RAG ON 组 vs RAG OFF 组的平均 grounding score
      - 总轮次分布
    供 Metrics 面板一眼看出 RAG 是否在帮忙。
    """
    df = conn.execute("""
        SELECT
            rag_used,
            COUNT(*)                                AS turns,
            ROUND(AVG(answer_grounding_score), 4)   AS avg_grounding,
            ROUND(AVG(retrieval_confidence), 4)     AS avg_confidence,
            ROUND(AVG(n_chunks_retrieved), 1)       AS avg_chunks
        FROM chat_logs
        GROUP BY rag_used
        ORDER BY rag_used DESC
    """).fetchdf()
    return df.to_dict("records") if not df.empty else []


# ---------------------------------------------------------------------------
# CSV 数据迁移（一次性）
# ---------------------------------------------------------------------------

def migrate_csv_to_duckdb(
    csv_path: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> int:
    """
    将现有 system_metrics.csv 数据迁移到 DuckDB。
    返回迁移的行数。跳过已存在的 report_id。
    """
    if not os.path.exists(csv_path):
        return 0

    close_conn = False
    if conn is None:
        conn = get_conn()
        close_conn = True

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0

        count = 0
        for _, row in df.iterrows():
            rid = generate_report_id()
            ts = row.get("timestamp", datetime.now().isoformat())

            # 插入 reports
            conn.execute(
                """
                INSERT INTO reports (
                    report_id, created_at, model_used, latency_sec,
                    total_tokens, cost_usd, report_path, transcript_path,
                    summary_preview, is_deleted, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', FALSE, NULL)
                """,
                [
                    rid,
                    ts,
                    str(row.get("model_used", "")),
                    float(row.get("latency_sec", 0)),
                    int(row.get("total_tokens", 0)),
                    float(row.get("cost_usd", 0)),
                    str(row.get("report_path", "")),
                    str(row.get("transcript_path", "")),
                ],
            )

            # 插入 eval_runs
            fc = row.get("factual_consistency")
            cp = row.get("completeness")
            ch = row.get("coherence")
            ws = row.get("objective_weighted_score")

            conn.execute(
                """
                INSERT INTO eval_runs (
                    eval_id, report_id, created_at, model_used,
                    factual_consistency, completeness, coherence,
                    objective_weighted_score,
                    chat_depth, re_asks, copy_clicks, expand_clicks,
                    time_spent_sec, is_feedback_collected, user_feedback_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(uuid.uuid4()),
                    rid,
                    ts,
                    str(row.get("model_used", "")),
                    None if pd.isna(fc) else float(fc),
                    None if pd.isna(cp) else float(cp),
                    None if pd.isna(ch) else float(ch),
                    None if pd.isna(ws) else float(ws),
                    int(row.get("chat_depth", 0)),
                    int(row.get("re_asks", 0)),
                    int(row.get("copy_clicks", 0)),
                    int(row.get("expand_clicks", 0)),
                    float(row.get("time_spent_sec", 0)),
                    bool(row.get("is_feedback_collected", False)),
                    None
                    if pd.isna(row.get("user_feedback_score"))
                    else float(row.get("user_feedback_score")),
                ],
            )
            count += 1

        return count
    finally:
        if close_conn:
            conn.close()
