"""
config.py — 参数中心（Single Source of Truth for all magic numbers）

所有 MAPE-K 决策阈值、权重、模型配置集中在此。
修改任何参数只需编辑本文件，无需改动业务代码。
参考：DESIGN.md §4.1 / §11.1
"""

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------
MODEL_TIERS = [
    "gemini-2.0-flash-lite-001",  # Tier 0: 最廉价
    "gemini-2.5-flash",           # Tier 1: 均衡
    "gemini-2.5-pro",             # Tier 2: 最强
]

PRICING_TABLE = {
    "gemini-2.0-flash-lite-001": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash":          {"input": 0.30,  "output": 1.20},
    "gemini-2.5-pro": {
        "input_low": 1.25,   # <= cutoff tokens
        "input_high": 2.50,  # >  cutoff tokens
        "cutoff": 200_000,
        "output": 5.00,
    },
}

# ---------------------------------------------------------------------------
# Objective Metrics 权重
# ---------------------------------------------------------------------------
METRIC_WEIGHTS = {
    "factual_consistency": 0.60,
    "completeness":        0.25,
    "coherence":           0.15,
}

# ---------------------------------------------------------------------------
# MAPE-K 决策阈值（satisfaction 区间 → 动作）
# ---------------------------------------------------------------------------
SATISFACTION_EXCELLENT  = 0.85   # >= 0.85: 质量充足，可降级省钱
SATISFACTION_GOOD       = 0.75   # 0.75-0.85: 维持当前模型
SATISFACTION_ACCEPTABLE = 0.65   # 0.65-0.75: 边界，资源充足时升级
SATISFACTION_POOR       = 0.50   # < 0.50: 必须升级

# ---------------------------------------------------------------------------
# 资源约束
# ---------------------------------------------------------------------------
COST_LIMIT_USD   = 0.10   # 总预算上限
RESOURCE_ABUNDANT = 0.70  # 剩余 >= 70% 视为充足
RESOURCE_MODERATE = 0.30  # 剩余 30%-70% 视为中等
RESOURCE_SCARCE   = 0.05  # 剩余 <= 5% 视为稀缺 / 破产

# ---------------------------------------------------------------------------
# 抗震荡参数（DESIGN §11.1）
# ---------------------------------------------------------------------------
WINDOW_SIZE          = 5     # 滑动窗口大小（最近 N 条有效记录）
TIME_DECAY_LAMBDA    = 0.35  # 指数衰减系数 λ
MODEL_COOLDOWN_RUNS  = 2     # 模型切换后至少等待 N 次再切换
SWITCH_MARGIN        = 0.03  # 切换门槛余量

# ---------------------------------------------------------------------------
# 行为信号权重（二阶优化，当前全部为 0）
# ---------------------------------------------------------------------------
BEHAVIOR_WEIGHTS = {
    "chat_depth":    0.0,
    "re_asks":       0.0,
    "copy_clicks":   0.0,
    "expand_clicks": 0.0,
}

# ---------------------------------------------------------------------------
# Embedding / 评估模型
# ---------------------------------------------------------------------------
EVAL_SBERT_DEFAULT = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Completeness 评估配置（DESIGN §评估优化）
# 多粒度语义覆盖 (Multi-Granular Semantic Coverage)
# ---------------------------------------------------------------------------
COMPLETENESS_CONFIG = {
    "threshold": 0.5,           # 句子级相似度阈值（比单向量0.4严格，比0.6宽松）
    "max_key_sentences": 50,    # 长视频关键句封顶（避免关键句爆炸）
    "recall_weight": 0.7,       # 双向验证中 Recall 权重（信息覆盖为主）
    "precision_weight": 0.3,    # 双向验证中 Precision 权重（防幻觉为辅）
}

# ---------------------------------------------------------------------------
# RAG 混合检索配置（Chat QA 专用；不再用于摘要生成）
# DESIGN §15.4：RAG for QA，全量 for Summary
# ---------------------------------------------------------------------------
RAG_CONFIG = {
    # NOTE: rag_threshold_tokens 已废弃（摘要生成不再触发 RAG）
    # 保留此字段仅为向后兼容，实际路由由 MAP_REDUCE_THRESHOLD_TOKENS 控制
    "rag_threshold_tokens": 999_999,  # 设极大值，等效于"永不触发"
    "chunk_size": 512,             # tokens per chunk（Chat QA 分块粒度）
    "chunk_overlap": 64,           # overlap tokens (Sliding Window)
    "top_k_retrieval": 5,          # Chat QA 检索返回 top-K chunks
    "max_context_tokens": 8000,    # 单次 Chat context 上限
    "rrf_k": 60,                   # RRF 常数 K (标准值 60)
}

# ---------------------------------------------------------------------------
# Map-Reduce 超长视频配置（DESIGN §15.4 轨道A）
# 摘要生成路由：
#   transcript < MAP_REDUCE_THRESHOLD_TOKENS → 直接全量输入（最准确）
#   transcript >= MAP_REDUCE_THRESHOLD_TOKENS → Map-Reduce 分段汇总
# ---------------------------------------------------------------------------
MAP_REDUCE_THRESHOLD_TOKENS = 50_000   # ~38K words ≈ ~60 分钟视频
MAP_REDUCE_SEGMENT_TOKENS   = 20_000   # 每段约 15K words

# ---------------------------------------------------------------------------
# 日志文件
# ---------------------------------------------------------------------------
LOG_FILE      = "system_metrics.csv"
FEEDBACK_FILE = "feedback_log.csv"

# CSV 列定义（与 DESIGN §3.1 一致）
CSV_COLUMNS = [
    "timestamp", "model_used", "latency_sec", "total_tokens", "cost_usd",
    "report_path", "transcript_path",
    "factual_consistency", "completeness", "coherence", "objective_weighted_score",
    "chat_depth", "re_asks", "copy_clicks", "expand_clicks", "time_spent_sec",
    "is_feedback_collected", "user_feedback_score",
]

FEEDBACK_COLUMNS = [
    "timestamp", "model_used", "report_path",
    "user_feedback_score", "user_feedback_text",
    "metrics_fc", "metrics_cp", "metrics_ch", "metrics_weighted_score",
    "user_action",
]
